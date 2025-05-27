const std = @import("std");
const Allocator = std.mem.Allocator;
const Thread = std.Thread;
const time = std.time;

const BakaCorasick = @import("baka_corasick.zig").BakaCorasick;
const fineweb = @import("data_loader.zig").FinewebDataLoader;
const VocabLearnerModule = @import("vocab_learner.zig");
const SampleStats = VocabLearnerModule.SampleStats;
const TokenStats = VocabLearnerModule.TokenStats;
const MatchInfo = VocabLearnerModule.MatchInfo;
const BoundedQueue = @import("spsc.zig").BoundedQueue;

const ProcessDocumentMessage = struct {
    id: usize,
    document: []const u8,
};

const CountTokensInDocumentMessage = struct {
    id: usize,
    document: []const u8,
};

const DocumentProcessedMessage = struct {
    id: usize,
};

const TokenCountMessage = struct {
    id: usize,
    token_count: u64,
};

const ErrorMessage = struct {
    worker_id: usize,
    message: []const u8,
};

const ResetMessage = struct {
    candidates: []SampleStats,
    automaton: *BakaCorasick,
};


const SubmissionQueueEntry = union(enum) {
    Reset: ResetMessage,
    ProcessDocument: ProcessDocumentMessage,
    CountTokensInDocument: CountTokensInDocumentMessage,
    Shutdown: usize, // worker id
};

const CompletionQueueEntry = union(enum) {
    DocumentProcessed: DocumentProcessedMessage,
    TokenCount: TokenCountMessage,
    Error: ErrorMessage,
};

// Worker implementation
pub const Worker = struct {
    allocator: Allocator,
    id: usize,
    vocab_learner: *VocabLearnerModule.VocabLearner,
    candidate_stats: []SampleStats,
    candidate_automaton: *BakaCorasick, // Now just a reference to shared automaton
    token_idx_to_least_end_pos: []u32,
    lookbacks: std.ArrayList(u64),
    dp_solution: std.ArrayList(u32),
    matches: std.ArrayList(MatchInfo),
    submission_queue: *BoundedQueue(SubmissionQueueEntry, 256),
    completion_queue: *BoundedQueue(CompletionQueueEntry, 256),
    thread: ?Thread = null,
    running: bool = false,
    candidates_length: usize,
    debug: bool,
    token_count: u64 = 0,
    current_n_candidates: usize = 0,

    pub fn init(
        allocator: Allocator,
        id: usize,
        vocab_learner: *VocabLearnerModule.VocabLearner,
        candidate_automaton: *BakaCorasick, // Now taking shared automaton
        candidates_length: usize,
        submission_queue: *BoundedQueue(SubmissionQueueEntry, 256),
        completion_queue: *BoundedQueue(CompletionQueueEntry, 256),
        debug: bool,
    ) !Worker {
        // Initialize data structures (each worker has its own)
        const lookbacks = std.ArrayList(u64).init(allocator);
        const dp_solution = std.ArrayList(u32).init(allocator);
        const matches = std.ArrayList(MatchInfo).init(allocator);

        const token_idx_to_least_end_pos = try allocator.alloc(u32, candidates_length);
        const candidate_stats = try allocator.alloc(SampleStats, candidates_length);

        return Worker{
            .allocator = allocator,
            .id = id,
            .vocab_learner = vocab_learner,
            .candidate_stats = candidate_stats,
            .candidate_automaton = candidate_automaton, // Using shared automaton
            .token_idx_to_least_end_pos = token_idx_to_least_end_pos,
            .lookbacks = lookbacks,
            .dp_solution = dp_solution,
            .matches = matches,
            .submission_queue = submission_queue,
            .completion_queue = completion_queue,
            .candidates_length = candidates_length,
            .debug = debug,
        };
    }

    pub fn deinit(self: *Worker) void {
        if (self.thread) |thread| {
            thread.join();
            self.thread = null;
        }

        // Don't deinit shared automaton, just our private data structures
        self.lookbacks.deinit();
        self.dp_solution.deinit();
        self.matches.deinit();
        self.allocator.free(self.token_idx_to_least_end_pos);
    }

    pub fn start(self: *Worker) !void {
        if (self.thread != null) return;

        self.running = true;
        self.thread = try Thread.spawn(.{}, Worker.run, .{self});

        if (self.debug) {
            std.debug.print("[Worker {d}] Started\n", .{self.id});
        }
    }

    pub fn stop(self: *Worker) void {
        self.running = false;
    }

    fn processDocument(self: *Worker, document: []const u8, doc_id: usize) !void {
        // Process the document using the evaluateCandidatesOnDocumentDP logic
        try self.vocab_learner.evaluateCandidatesOnDocumentDP(
            self.candidate_stats[0..self.current_n_candidates],
            self.candidate_automaton,
            document,
            &self.lookbacks,
            &self.dp_solution,
            &self.matches,
            self.token_idx_to_least_end_pos[0..self.current_n_candidates],
        );

        // Send results
        const result_msg = CompletionQueueEntry{
            .DocumentProcessed = DocumentProcessedMessage{
                .id = doc_id,
            },
        };

        var pushed = self.completion_queue.push(result_msg);
        while (!pushed) {
            std.time.sleep(1 * std.time.ns_per_ms);
            pushed = self.completion_queue.push(result_msg);
        }
    }

    fn countTokensInDocument(
        self: *Worker,
        document: []const u8,
        doc_id: usize,
    ) !void {
        const token_count = try self.vocab_learner.getDocumentTokenCount(
            document,
            &self.lookbacks,
            &self.dp_solution);

        const result_msg = CompletionQueueEntry{
            .TokenCount = TokenCountMessage{
                .id = doc_id,
                .token_count = token_count,
            },
        };

        var pushed = self.completion_queue.push(result_msg);
        while (!pushed) {
            std.time.sleep(1 * std.time.ns_per_ms);
            pushed = self.completion_queue.push(result_msg);
        }
    }

    fn run(self: *Worker) !void {
        if (self.debug) {
            std.debug.print("[Worker {d}] Running\n", .{self.id});
        }

        while (self.running) {
            if (self.submission_queue.pop()) |msg| {
                switch (msg) {
                    .Reset => |resset_message| {
                        const coordinator_stats = resset_message.candidates;
                        const automaton = resset_message.automaton;
                        self.current_n_candidates = coordinator_stats.len;
                        if (self.candidate_stats.len < self.current_n_candidates) {
                            @panic("oh no!1");
                        }
                        for (self.candidate_stats[0..self.current_n_candidates], 0..) |*stats, idx| {
                            stats.sampled_occurrences = 0;
                            stats.sampled_savings = 0;
                            stats.token_id = coordinator_stats[idx].token_id;
                        }
                        self.candidate_automaton = automaton;
                    },
                    .ProcessDocument => |process_data| {
                        self.processDocument(process_data.document, process_data.id) catch |err| {
                            const error_msg = try std.fmt.allocPrint(self.allocator, "Error processing document: {s}", .{@errorName(err)});
                            const error_message = CompletionQueueEntry{
                                .Error = ErrorMessage{
                                    .worker_id = self.id,
                                    .message = error_msg,
                                },
                            };
                            _ = self.completion_queue.push(error_message);
                        };
                    },
                    .CountTokensInDocument => |count_tokens_in_document| {
                        self.countTokensInDocument(count_tokens_in_document.document, count_tokens_in_document.id) catch |err| {
                            const error_msg = try std.fmt.allocPrint(self.allocator, "Error counting tokens in document: {s}", .{@errorName(err)});
                            const error_message = CompletionQueueEntry{
                                .Error = ErrorMessage{
                                    .worker_id = self.id,
                                    .message = error_msg,
                                },
                            };
                            _ = self.completion_queue.push(error_message);
                        };
                    },
                    .Shutdown => {
                        if (self.debug) {
                            std.debug.print("[Worker {d}] Shutting down\n", .{self.id});
                        }
                        self.running = false;
                    },
                    // else => {
                    //     if (self.debug) {
                    //         std.debug.print("[Worker {d}] Unexpected message type\n", .{self.id});
                    //     }
                    // },
                }
            } else {
                // No messages, sleep a bit
                std.time.sleep(1 * std.time.ns_per_ms);
            }
        }

        if (self.debug) {
            std.debug.print("[Worker {d}] Stopped\n", .{self.id});
        }
    }
};

// Coordinator implementation
pub const ParallelDP = struct {
    allocator: Allocator,
    vocab_learner: *VocabLearnerModule.VocabLearner,
    submission_queues: []BoundedQueue(SubmissionQueueEntry, 256),
    completion_queues: []BoundedQueue(CompletionQueueEntry, 256),
    workers: []Worker,
    num_workers: usize,
    debug: bool,
    next_document_id: usize = 0,
    n_outstanding_jobs: []usize,
    pending_documents: [][]const u8,
    pending_documents_free_list: []usize,
    n_free_pending_documents: usize,
    queue_depth_for_tokenize: usize,
    queue_depth_for_dp: usize,
    processed_count: usize = 0,
    started_workers: bool = false,

    pub fn init(
        allocator: Allocator,
        vocab_learner: *VocabLearnerModule.VocabLearner,
        debug: bool,
    ) !ParallelDP {
        // Determine number of workers based on CPU cores
        const num_workers = try Thread.getCpuCount();

        // Create queues
        var submission_queues = try allocator.alloc(BoundedQueue(SubmissionQueueEntry, 256), num_workers);
        errdefer allocator.free(submission_queues);

        var completion_queues = try allocator.alloc(BoundedQueue(CompletionQueueEntry, 256), num_workers);
        errdefer allocator.free(completion_queues);

        // Initialize queues
        for (0..num_workers) |i| {
            submission_queues[i] = try BoundedQueue(SubmissionQueueEntry, 256).init(allocator);
            completion_queues[i] = try BoundedQueue(CompletionQueueEntry, 256).init(allocator);
        }

        // Allocate worker array
        const workers = try allocator.alloc(Worker, num_workers);
        errdefer allocator.free(workers);

        const queue_depth_for_tokenize = 20;
        const queue_depth_for_dp = 20;
        const larger_queue_depth = @max(queue_depth_for_tokenize, queue_depth_for_dp);
        const pending_documents = try allocator.alloc([]const u8, num_workers * larger_queue_depth);
        errdefer allocator.free(pending_documents);
        for (pending_documents) |*ptr| {
            ptr.* = &[_]u8{};
        }
        const pending_documents_free_list = try allocator.alloc(usize, num_workers * larger_queue_depth);
        errdefer allocator.free(pending_documents_free_list);
        for (pending_documents_free_list, 0..) |*ptr, i| {
            ptr.* = i;
        }

        const n_outstanding_jobs = try allocator.alloc(usize, num_workers);
        errdefer allocator.free(n_outstanding_jobs);
        @memset(n_outstanding_jobs, 0);

        const automaton = try BakaCorasick.init(allocator);
        errdefer automaton.deinit();

        if (debug) {
            std.debug.print("[ParallelDP] Initialized with {d} workers\n", .{num_workers});
        }

        return ParallelDP{
            .allocator = allocator,
            .vocab_learner = vocab_learner,
            .submission_queues = submission_queues,
            .completion_queues = completion_queues,
            .workers = workers,
            .num_workers = num_workers,
            .debug = debug,
            .pending_documents = pending_documents,
            .pending_documents_free_list = pending_documents_free_list,
            .n_free_pending_documents = num_workers * larger_queue_depth,
            .queue_depth_for_tokenize = queue_depth_for_tokenize,
            .queue_depth_for_dp = queue_depth_for_dp,
            .n_outstanding_jobs = n_outstanding_jobs,
        };
    }

    pub fn deinit(self: *ParallelDP) void {
        // Stop and deinit all workers
        for (0..self.num_workers) |i| {
            self.workers[i].stop();
            self.workers[i].deinit();

            // Deinit queues
            self.submission_queues[i].deinit(self.allocator);
            self.completion_queues[i].deinit(self.allocator);
        }

        // Free arrays
        self.allocator.free(self.pending_documents);
        self.allocator.free(self.pending_documents_free_list);
        self.allocator.free(self.submission_queues);
        self.allocator.free(self.completion_queues);
        self.allocator.free(self.workers);
        self.allocator.free(self.n_outstanding_jobs);
    }

    /// Initialize workers with the given candidates to evaluate
    pub fn initWorkers(
        self: *ParallelDP,
        candidates: []SampleStats,
        automaton: *BakaCorasick,
    ) !void {
        if (self.debug) {
            std.debug.print("[ParallelDP] Initializing {d} workers with {d} candidates\n", .{ self.num_workers, candidates.len });
        }
        const automaton_start = time.nanoTimestamp();

        if (self.debug) {
            std.debug.print("[ParallelDP] Using pre-built automaton with {d} states\n", .{automaton.len});
        }

        const automaton_elapsed = time.nanoTimestamp() - automaton_start;
        if (self.debug) {
            std.debug.print("[ParallelDP] Using automaton, elapsed: {d:.2}ms\n", .{@as(f64, @floatFromInt(automaton_elapsed)) / time.ns_per_ms});
        }

        // Create and start workers, sharing the automaton
        if (!self.started_workers) {
            for (0..self.num_workers) |i| {
                self.workers[i] = try Worker.init(self.allocator, i, self.vocab_learner, automaton, candidates.len, &self.submission_queues[i], &self.completion_queues[i], self.debug);

                try self.workers[i].start();
            }
            self.started_workers = true;
        }

        for (0..self.num_workers) |i| {
            const msg = SubmissionQueueEntry{
                .Reset = .{
                    .candidates = candidates,
                    .automaton = automaton,
                },
            };
            _ = self.sendMessageToWorker(i, msg, false);
        }
    }

    pub fn initWorkersForCorpusTokenCount(
        self: *ParallelDP,
    ) !void {
        if (self.debug) {
            std.debug.print("[ParallelDP] Initializing {d} workers for corpus token count\n", .{self.num_workers});
        }

        if (!self.started_workers) {
            return error.FixMePlz;
        }
    }

    fn addToPendingDocuments(self: *ParallelDP, document: []const u8) usize {
        //std.debug.print("Adding document to pending documents list {d}\n", .{self.n_free_pending_documents});
        self.n_free_pending_documents -= 1;
        const index = self.pending_documents_free_list[self.n_free_pending_documents];
        self.pending_documents[index] = document;
        return index;
    }

    fn removeFromPendingDocuments(self: *ParallelDP, index: usize) void {
        //std.debug.print("Removing document from pending documents list {d}\n", .{self.n_free_pending_documents});
        const document = self.pending_documents[index];
        self.allocator.free(document);
        self.pending_documents_free_list[self.n_free_pending_documents] = index;
        self.pending_documents[index] = &[_]u8{};
        self.n_free_pending_documents += 1;
    }

    /// Send a message to a worker and track it if a response is expected
    fn sendMessageToWorker(self: *ParallelDP, worker_id: usize, msg: SubmissionQueueEntry, expect_response: bool) bool {
        const result = self.submission_queues[worker_id].push(msg);
        if (result and expect_response) {
            self.n_outstanding_jobs[worker_id] += 1;
        }
        return result;
    }

    /// Process a batch of documents in parallel
    pub fn processDocuments(
        self: *ParallelDP,
        loader: *fineweb,
        sample_size: usize,
        candidates: []SampleStats,
        candidate_automaton: *BakaCorasick,
    ) !void {
        const start_time = time.nanoTimestamp();
        // Initialize workers with shared automaton
        try self.initWorkers(candidates, candidate_automaton);

        // Process documents
        var documents_processed: usize = 0;
        var documents_submitted: usize = 0;
        var running = true;

        while (running) {
            var did_anything = false;

            // Loop through each worker's output queue
            for (0..self.num_workers) |i| {
                // Process all available messages from this worker
                if (self.completion_queues[i].pop()) |msg| {
                    self.n_outstanding_jobs[i] -= 1;
                    did_anything = true;
                    switch (msg) {
                        .DocumentProcessed => |doc_processed_msg| {
                            self.removeFromPendingDocuments(doc_processed_msg.id);
                            documents_processed += 1;
                        },
                        .Error => |error_data| {
                            std.debug.print("[Coordinator] Worker {d} error: {s}\n", .{ i, error_data.message });
                            @panic("oh no!2");
                        },
                        else => {
                            std.debug.print("[Coordinator] Worker {d} error: Unknown message type\n", .{ i });
                            @panic("oh no!3");
                        },
                    }
                }
                if (documents_submitted < sample_size and self.n_outstanding_jobs[i] < self.queue_depth_for_dp) {
                    const doc = try loader.nextDocumentStringLoop();
                    did_anything = true;
                    // Add to pending documents
                    const doc_idx = self.addToPendingDocuments(doc);

                    // Create a message for the worker
                    const msg = SubmissionQueueEntry{
                        .ProcessDocument = ProcessDocumentMessage{
                            .id = doc_idx,
                            .document = doc,
                        },
                    };

                    _ = self.sendMessageToWorker(i, msg, true); // Expect response
                    documents_submitted += 1;
                    //std.debug.print("Submitted document {d} to worker {d}, now self.n_outstanding_jobs[i] = {d} and documents_submitted = {d}\n", .{ doc_idx, i, self.n_outstanding_jobs[i], documents_submitted });
                }
            }

            if (documents_processed == sample_size) {
                // copy the stats from all workers
                for (self.workers) |worker| {
                    for (worker.candidate_stats, 0..) |stats, idx| {
                        candidates[idx].sampled_occurrences += stats.sampled_occurrences;
                        candidates[idx].sampled_savings += stats.sampled_savings;
                    }
                }
                // if we haven't seen each token at least 5 times, keep going
                // var keep_going = false;
                // for (candidates) |stats| {
                //     if (stats.sampled_occurrences < 5) {
                //         keep_going = true;
                //         break;
                //     }
                // }
                // if (keep_going) {
                //     documents_submitted = 0;
                //     documents_processed = 0;
                // } else {
                running = false;
                // }
            }

            // If we didn't process any messages, sleep a bit to avoid spinning
            if (!did_anything) {
                std.time.sleep(1);
            }
        }

        if (self.debug) {
            std.debug.print("[ParallelDP] Initially submitted {d} documents to workers\n", .{documents_submitted});
        }

        const elapsed = time.nanoTimestamp() - start_time;
        if (self.debug) {
            std.debug.print("[ParallelDP] Processed {d} documents in {d:.2}ms\n", .{ documents_processed, @as(f64, @floatFromInt(elapsed)) / time.ns_per_ms });
        }
    }

    pub fn updateMaxSavingsByTokenizingCandidates(
        self: *ParallelDP,
        candidates: []u32,
    ) !void {
        const start_time = time.nanoTimestamp();
        // Initialize workers with shared automaton
        try self.initWorkersForCorpusTokenCount();

        // Process documents
        var documents_processed: usize = 0;
        var documents_submitted: usize = 0;

        while (true) {
            var did_anything = false;

            // Loop through each worker's output queue
            for (0..self.num_workers) |i| {
                // Process all available messages from this worker
                if (self.completion_queues[i].pop()) |msg| {
                    self.n_outstanding_jobs[i] -= 1;
                    did_anything = true;
                    switch (msg) {
                        .TokenCount => |token_count_msg| {
                            const token_id: u32 = @intCast(token_count_msg.id);
                            const token_count: u16 = @intCast(token_count_msg.token_count);
                            const stats: *TokenStats = &self.vocab_learner.candidate_stats[token_id];
                            const old_token_count = stats.len_in_tokens;
                            if (token_count < old_token_count) {
                                stats.len_in_tokens = token_count;
                                const old_est_savings = stats.est_total_savings;
                                const occurrence_count = stats.n_nonoverlapping_occurrences;
                                const new_savings: f64 = @floatFromInt(occurrence_count * (token_count - 1));
                                stats.est_total_savings = @min(old_est_savings, new_savings);
                            }
                            documents_processed += 1;
                        },
                        .Error => |error_data| {
                            std.debug.print("[Coordinator] Worker {d} error: {s}\n", .{ i, error_data.message });
                            @panic("oh no!4");
                        },
                        else => {
                            std.debug.print("[Coordinator] Worker {d} error: Unknown message type\n", .{ i });
                            @panic("oh no!5");
                        },
                    }
                }
                if (self.n_outstanding_jobs[i] < self.queue_depth_for_tokenize) {
                    if (documents_submitted < candidates.len) {
                        const doc = self.vocab_learner.getTokenStr(candidates[documents_submitted]);
                        did_anything = true;

                        // Create a message for the worker
                        const msg = SubmissionQueueEntry{
                            .CountTokensInDocument = CountTokensInDocumentMessage{
                                .id = documents_submitted,
                                .document = doc,
                            },
                        };

                        _ = self.sendMessageToWorker(i, msg, true); // Expect response
                        documents_submitted += 1;
                    }
                }
            }

            if (documents_processed == candidates.len) {
                const elapsed = time.nanoTimestamp() - start_time;
                if (self.debug) {
                    std.debug.print("[ParallelDP] Tokenized {d} candidates in {d:.2}ms\n", .{ documents_processed, @as(f64, @floatFromInt(elapsed)) / time.ns_per_ms });
                }
                return;
            }

            // If we didn't process any messages, sleep a bit to avoid spinning
            if (!did_anything) {
                std.time.sleep(1);
            }

        }
    }

    /// Process a batch of documents in parallel
    pub fn getCorpusTokenCount(
        self: *ParallelDP,
    ) !u64 {
        const start_time = time.nanoTimestamp();
        // Initialize workers with shared automaton
        try self.initWorkersForCorpusTokenCount();

        // Process documents
        var documents_processed: usize = 0;
        var documents_submitted: usize = 0;

        const loader = self.vocab_learner.loader.?;
        try loader.rewind();
        var maybe_doc = try loader.nextDocumentString();
        var ret: u64 = 0;

        while (true) {
            var did_anything = false;

            // Loop through each worker's output queue
            for (0..self.num_workers) |i| {
                // Process all available messages from this worker
                if (self.completion_queues[i].pop()) |msg| {
                    self.n_outstanding_jobs[i] -= 1;
                    did_anything = true;
                    switch (msg) {
                        .TokenCount => |token_count_msg| {
                            self.removeFromPendingDocuments(token_count_msg.id);
                            ret += token_count_msg.token_count;
                            documents_processed += 1;
                        },
                        .Error => |error_data| {
                            std.debug.print("[Coordinator] Worker {d} error: {s}\n", .{ i, error_data.message });
                            @panic("oh no!6");
                        },
                        else => {
                            std.debug.print("[Coordinator] Worker {d} error: Unknown message type\n", .{ i });
                            @panic("oh no!7");
                        },
                    }
                }
                if (self.n_outstanding_jobs[i] < self.queue_depth_for_tokenize) {
                    if (maybe_doc) |doc| {
                        did_anything = true;
                        // Add to pending documents
                        const doc_idx = self.addToPendingDocuments(doc);

                        // Create a message for the worker
                        const msg = SubmissionQueueEntry{
                            .CountTokensInDocument = CountTokensInDocumentMessage{
                                .id = doc_idx,
                                .document = doc,
                            },
                        };

                        _ = self.sendMessageToWorker(i, msg, true); // Expect response
                        documents_submitted += 1;
                        maybe_doc = try loader.nextDocumentString();
                        //std.debug.print("Submitted document {d} to worker {d}, now self.n_outstanding_jobs[i] = {d} and documents_submitted = {d}\n", .{ doc_idx, i, self.n_outstanding_jobs[i], documents_submitted });
                    }
                }
            }

            if (maybe_doc == null and documents_processed == documents_submitted) {
                const elapsed = time.nanoTimestamp() - start_time;
                if (self.debug) {
                    std.debug.print("[ParallelDP] Processed {d} documents in {d:.2}ms\n", .{ documents_processed, @as(f64, @floatFromInt(elapsed)) / time.ns_per_ms });
                }
                return ret;
            }

            // If we didn't process any messages, sleep a bit to avoid spinning
            if (!did_anything) {
                std.time.sleep(1);
            }
        }
    }
};
