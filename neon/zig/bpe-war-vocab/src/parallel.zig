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

// SPSC Queue implementation - reusing from provided files
const cache_line_length = switch (@import("builtin").target.cpu.arch) {
    .aarch64, .powerpc64 => 128,
    .arm, .mips, .mips64, .riscv64 => 32,
    .s390x => 256,
    else => 64,
};

pub fn BoundedQueue(comptime T: type, comptime capacity: comptime_int) type {
    std.debug.assert(std.math.isPowerOfTwo(capacity));
    const mask = capacity - 1;

    return struct {
        const Self = @This();

        // align to 2*cache_line_length to fight the next-line prefetcher
        buffer: *[capacity]T align(2 * cache_line_length),
        head: usize align(2 * cache_line_length),
        tail: usize align(2 * cache_line_length),

        pub fn init(gpa: Allocator) !Self {
            const buffer = try gpa.create([capacity]T);
            return Self{ .buffer = buffer, .head = 0, .tail = 0 };
        }

        pub fn deinit(self: *Self, gpa: Allocator) void {
            gpa.destroy(self.buffer);
        }

        pub fn push(self: *Self, value: T) bool {
            const head = @atomicLoad(usize, &self.head, .acquire);
            const tail = self.tail;
            if (head +% 1 -% tail > capacity) return false;
            self.buffer[head & mask] = value;
            @atomicStore(usize, &self.head, head +% 1, .release);
            return true;
        }

        pub fn pop(self: *Self) ?T {
            const tail = @atomicLoad(usize, &self.tail, .acquire);
            const head = self.head;
            if (tail -% head == 0) return null;
            const value = self.buffer[tail & mask];
            @atomicStore(usize, &self.tail, tail +% 1, .release);
            return value;
        }
    };
}

// Message types
const MessageType = enum {
    ProcessDocument,
    DocumentProcessed,
    Shutdown,
    Error,
};

const ProcessDocumentMessage = struct {
    id: usize,
    document: []const u8,
};

const DocumentProcessedMessage = struct {
    id: usize,
    worker_id: usize,
    document_stats: []SampleStats,
};

const ErrorMessage = struct {
    worker_id: usize,
    message: []const u8,
};

const Message = union(MessageType) {
    ProcessDocument: ProcessDocumentMessage,
    DocumentProcessed: DocumentProcessedMessage,
    Shutdown: usize, // worker id
    Error: ErrorMessage,
};

fn deinitBakaCorasick(automaton: *BakaCorasick, allocator: Allocator) void {
    allocator.free(automaton.transitions[0..automaton.capacity]);
    allocator.free(automaton.info[0..automaton.capacity]);
}

// Worker implementation
pub const Worker = struct {
    allocator: Allocator,
    id: usize,
    vocab_learner: *VocabLearnerModule.VocabLearner,
    candidate_automaton: *BakaCorasick, // Now just a reference to shared automaton
    token_idx_to_least_end_pos: []u32,
    lookbacks: std.ArrayList(u64),
    dp_solution: std.ArrayList(u32),
    matches: std.ArrayList(MatchInfo),
    input_queue: *BoundedQueue(Message, 256),
    output_queue: *BoundedQueue(Message, 256),
    thread: ?Thread = null,
    running: bool = false,
    candidates_length: usize,
    debug: bool,

    pub fn init(
        allocator: Allocator,
        id: usize,
        vocab_learner: *VocabLearnerModule.VocabLearner,
        candidate_automaton: *BakaCorasick, // Now taking shared automaton
        candidates_length: usize,
        input_queue: *BoundedQueue(Message, 256),
        output_queue: *BoundedQueue(Message, 256),
        debug: bool,
    ) !*Worker {
        const worker = try allocator.create(Worker);
        errdefer allocator.destroy(worker);

        // Initialize data structures (each worker has its own)
        const lookbacks = std.ArrayList(u64).init(allocator);
        const dp_solution = std.ArrayList(u32).init(allocator);
        const matches = std.ArrayList(MatchInfo).init(allocator);

        const token_idx_to_least_end_pos = try allocator.alloc(u32, candidates_length);

        worker.* = .{
            .allocator = allocator,
            .id = id,
            .vocab_learner = vocab_learner,
            .candidate_automaton = candidate_automaton, // Using shared automaton
            .token_idx_to_least_end_pos = token_idx_to_least_end_pos,
            .lookbacks = lookbacks,
            .dp_solution = dp_solution,
            .matches = matches,
            .input_queue = input_queue,
            .output_queue = output_queue,
            .candidates_length = candidates_length,
            .debug = debug,
        };

        return worker;
    }

    pub fn deinit(self: *Worker) void {
        if (self.thread) |thread| {
            thread.join();
        }

        // Don't deinit shared automaton, just our private data structures
        self.lookbacks.deinit();
        self.dp_solution.deinit();
        self.matches.deinit();
        self.allocator.free(self.token_idx_to_least_end_pos);
        self.allocator.destroy(self);
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
        // Create local copies of candidate stats for this document
        var local_stats = try self.allocator.alloc(SampleStats, self.candidates_length);
        defer self.allocator.free(local_stats);

        // Initialize local stats with zero counts
        for (0..self.candidates_length) |i| {
            local_stats[i] = .{
                .token_id = @intCast(i), // Store index for coordinator to map back
                .sampled_occurrences = 0,
                .sampled_savings = 0,
            };
        }

        // Process the document using the evaluateCandidatesOnDocumentDP logic
        try self.vocab_learner.evaluateCandidatesOnDocumentDP(
            local_stats,
            self.candidate_automaton,
            document,
            &self.lookbacks,
            &self.dp_solution,
            &self.matches,
            self.token_idx_to_least_end_pos,
        );

        // Create copy of results to send back to coordinator
        const result_stats = try self.allocator.dupe(SampleStats, local_stats);

        // Send results
        const result_msg = Message{
            .DocumentProcessed = DocumentProcessedMessage{
                .id = doc_id,
                .worker_id = self.id,
                .document_stats = result_stats,
            },
        };

        var pushed = false;
        while (!pushed) {
            pushed = self.output_queue.push(result_msg);
            if (!pushed) {
                std.time.sleep(1 * std.time.ns_per_ms);
            }
        }
    }

    fn run(self: *Worker) !void {
        if (self.debug) {
            std.debug.print("[Worker {d}] Running\n", .{self.id});
        }

        while (self.running) {
            if (self.input_queue.pop()) |msg| {
                switch (msg) {
                    .ProcessDocument => |process_data| {
                        self.processDocument(process_data.document, process_data.id) catch |err| {
                            const error_msg = try std.fmt.allocPrint(self.allocator, "Error processing document: {s}", .{@errorName(err)});
                            const error_message = Message{
                                .Error = ErrorMessage{
                                    .worker_id = self.id,
                                    .message = error_msg,
                                },
                            };
                            _ = self.output_queue.push(error_message);
                        };
                    },
                    .Shutdown => {
                        if (self.debug) {
                            std.debug.print("[Worker {d}] Shutting down\n", .{self.id});
                        }
                        self.running = false;
                    },
                    else => {
                        if (self.debug) {
                            std.debug.print("[Worker {d}] Unexpected message type\n", .{self.id});
                        }
                    },
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
    input_queues: []BoundedQueue(Message, 256),
    output_queues: []BoundedQueue(Message, 256),
    workers: []*Worker,
    shared_automaton: BakaCorasick, // Now we keep a shared automaton
    num_workers: usize,
    debug: bool,
    next_document_id: usize = 0,
    pending_documents: std.AutoHashMap(usize, []const u8),
    processed_count: usize = 0,

    pub fn init(
        allocator: Allocator,
        vocab_learner: *VocabLearnerModule.VocabLearner,
        debug: bool,
    ) !*ParallelDP {
        const coordinator = try allocator.create(ParallelDP);
        errdefer allocator.destroy(coordinator);

        // Determine number of workers based on CPU cores
        const num_workers = try Thread.getCpuCount();

        // Create queues
        var input_queues = try allocator.alloc(BoundedQueue(Message, 256), num_workers);
        errdefer allocator.free(input_queues);

        var output_queues = try allocator.alloc(BoundedQueue(Message, 256), num_workers);
        errdefer allocator.free(output_queues);

        // Initialize queues
        for (0..num_workers) |i| {
            input_queues[i] = try BoundedQueue(Message, 256).init(allocator);
            output_queues[i] = try BoundedQueue(Message, 256).init(allocator);
        }

        // Allocate worker array
        const workers = try allocator.alloc(*Worker, num_workers);
        errdefer allocator.free(workers);

        // Initialize coordinator state
        coordinator.* = .{
            .allocator = allocator,
            .vocab_learner = vocab_learner,
            .input_queues = input_queues,
            .output_queues = output_queues,
            .workers = workers,
            .shared_automaton = undefined, // Will be initialized in initWorkers
            .num_workers = num_workers,
            .debug = debug,
            .pending_documents = std.AutoHashMap(usize, []const u8).init(allocator),
        };

        if (debug) {
            std.debug.print("[ParallelDP] Initialized with {d} workers\n", .{num_workers});
        }

        return coordinator;
    }

    pub fn deinit(self: *ParallelDP) void {
        // Stop and deinit all workers
        for (0..self.num_workers) |i| {
            self.workers[i].stop();
            self.workers[i].deinit();

            // Deinit queues
            self.input_queues[i].deinit(self.allocator);
            self.output_queues[i].deinit(self.allocator);
        }

        // Free the shared automaton
        deinitBakaCorasick(&self.shared_automaton, self.allocator);

        // Free remaining pending documents
        var it = self.pending_documents.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.value_ptr.*);
        }
        self.pending_documents.deinit();

        // Free arrays
        self.allocator.free(self.input_queues);
        self.allocator.free(self.output_queues);
        self.allocator.free(self.workers);

        // Free self
        self.allocator.destroy(self);
    }

    /// Initialize workers with the given candidates to evaluate
    pub fn initWorkers(
        self: *ParallelDP,
        candidates: []SampleStats,
        candidates_length: usize,
    ) !void {
        if (self.debug) {
            std.debug.print("[ParallelDP] Initializing {d} workers with {d} candidates\n", .{ self.num_workers, candidates_length });
        }

        // Build a SINGLE automaton on the main thread
        const automaton_start = time.nanoTimestamp();

        // Initialize shared automaton
        self.shared_automaton = try BakaCorasick.init(self.allocator);

        // Add candidate tokens to the shared automaton
        for (candidates[0..candidates_length], 0..) |stats, idx| {
            const my_idx: u32 = @intCast(idx);
            const token_id = stats.token_id;
            const token_str = self.vocab_learner.getTokenStr(token_id);
            try self.shared_automaton.insert(token_str, my_idx);
        }

        // Compute suffix links for the shared automaton
        try self.shared_automaton.computeSuffixLinks();

        const automaton_elapsed = time.nanoTimestamp() - automaton_start;
        if (self.debug) {
            std.debug.print("[ParallelDP] Built shared automaton in {d:.2}ms\n", .{@as(f64, @floatFromInt(automaton_elapsed)) / time.ns_per_ms});
        }

        // Create and start workers, sharing the automaton
        for (0..self.num_workers) |i| {
            self.workers[i] = try Worker.init(self.allocator, i, self.vocab_learner, &self.shared_automaton, // Share the automaton
                candidates_length, &self.input_queues[i], &self.output_queues[i], self.debug);

            try self.workers[i].start();
        }
    }

    /// Process a batch of documents in parallel
    pub fn processDocuments(
        self: *ParallelDP,
        loader: *fineweb,
        sample_size: usize,
        candidates: []SampleStats,
        candidates_length: usize,
    ) !void {
        const start_time = time.nanoTimestamp();
        // At the start of processDocuments in ParallelDP:
        if (self.debug) {
            const status = loader.getFileStatus();
            std.debug.print("[ParallelDP] Document loader status: {d} files\n", .{status.total_files});

            // Try to read a test document
            const doc = try loader.nextDocumentStringLoop();
            if (doc.len > 0) {
                std.debug.print("[ParallelDP] Successfully read test document of length {d}\n", .{doc.len});
                // Read another document to test the loop
                const doc2 = try loader.nextDocumentStringLoop();
                std.debug.print("[ParallelDP] Successfully read second test document of length {d}\n", .{doc2.len});
            } else {
                std.debug.print("[ParallelDP] Read empty document - loader may have issues\n", .{});
            }

            // Important: Rewind the loader to reset position after this test
            try loader.rewind();
        }
        // Clear candidate stats before starting
        for (candidates[0..candidates_length]) |*stats| {
            stats.sampled_occurrences = 0;
            stats.sampled_savings = 0;
        }

        // Initialize workers with shared automaton
        try self.initWorkers(candidates, candidates_length);

        // Process documents
        var documents_processed: usize = 0;
        var documents_submitted: usize = 0;
        var worker_idx: usize = 0;

        // Initial document feeding
        const feed_per_worker = 4; // Number of documents to initially feed each worker
        for (0..self.num_workers * feed_per_worker) |_| {
            if (documents_submitted >= sample_size) break;

            if (try loader.nextDocumentString()) |doc| {
                const doc_copy = try self.allocator.dupe(u8, doc);
                const doc_id = self.next_document_id;
                self.next_document_id += 1;

                try self.pending_documents.put(doc_id, doc_copy);

                const msg = Message{
                    .ProcessDocument = ProcessDocumentMessage{
                        .id = doc_id,
                        .document = doc_copy,
                    },
                };

                // Try to push to a worker's queue
                var pushed = false;
                var attempts: usize = 0;
                while (!pushed and attempts < self.num_workers) {
                    pushed = self.input_queues[worker_idx].push(msg);
                    if (!pushed) {
                        worker_idx = (worker_idx + 1) % self.num_workers;
                        attempts += 1;
                    }
                }

                if (pushed) {
                    documents_submitted += 1;
                } else {
                    // Couldn't push to any worker, free the document
                    self.allocator.free(doc_copy);
                    _ = self.pending_documents.remove(doc_id);
                    if (self.debug) {
                        std.debug.print("[ParallelDP] Warning: Couldn't submit document to any worker\n", .{});
                    }
                }
            } else {
                // No more documents
                break;
            }
        }

        if (self.debug) {
            std.debug.print("[ParallelDP] Initially submitted {d} documents to workers\n", .{documents_submitted});
        }

        // Process results and feed more documents
        while (documents_processed < sample_size and documents_processed < documents_submitted) {
            // Check for results from all workers
            var got_results = false;

            for (0..self.num_workers) |i| {
                if (self.output_queues[i].pop()) |msg| {
                    got_results = true;

                    switch (msg) {
                        .DocumentProcessed => |processed| {
                            // Update candidate statistics - using actual token IDs
                            for (processed.document_stats, 0..) |stats, idx| {
                                if (idx < candidates_length) {
                                    candidates[idx].sampled_occurrences += stats.sampled_occurrences;
                                    candidates[idx].sampled_savings += stats.sampled_savings;
                                }
                            }

                            // Free the result stats
                            self.allocator.free(processed.document_stats);

                            // Remove document from pending
                            if (self.pending_documents.fetchRemove(processed.id)) |entry| {
                                self.allocator.free(entry.value);
                            }

                            documents_processed += 1;

                            // Submit another document if we haven't reached the limit
                            if (documents_submitted < sample_size) {
                                if (try loader.nextDocumentString()) |doc| {
                                    const doc_copy = try self.allocator.dupe(u8, doc);
                                    const doc_id = self.next_document_id;
                                    self.next_document_id += 1;

                                    try self.pending_documents.put(doc_id, doc_copy);

                                    const new_msg = Message{
                                        .ProcessDocument = ProcessDocumentMessage{
                                            .id = doc_id,
                                            .document = doc_copy,
                                        },
                                    };

                                    if (self.input_queues[i].push(new_msg)) {
                                        documents_submitted += 1;
                                    } else {
                                        // Could not push, free the document
                                        self.allocator.free(doc_copy);
                                        _ = self.pending_documents.remove(doc_id);
                                    }
                                }
                            }

                            // Log progress
                            if (self.debug and documents_processed % 1000 == 0) {
                                std.debug.print("[ParallelDP] Processed {d}/{d} documents\n", .{ documents_processed, sample_size });
                            }
                        },
                        .Error => |error_data| {
                            std.debug.print("[ParallelDP] Worker {d} error: {s}\n", .{ error_data.worker_id, error_data.message });
                            self.allocator.free(error_data.message);
                        },
                        else => {
                            std.debug.print("[ParallelDP] Unexpected message from worker\n", .{});
                        },
                    }
                }
            }

            if (!got_results) {
                // No results, sleep a bit
                std.time.sleep(1 * std.time.ns_per_ms);
            }

            // Check if we've processed all submitted documents and there are no more to submit
            if (documents_processed == documents_submitted and documents_submitted < sample_size) {
                if (try loader.nextDocumentString()) |doc| {
                    const doc_copy = try self.allocator.dupe(u8, doc);
                    const doc_id = self.next_document_id;
                    self.next_document_id += 1;

                    try self.pending_documents.put(doc_id, doc_copy);

                    const new_msg = Message{
                        .ProcessDocument = ProcessDocumentMessage{
                            .id = doc_id,
                            .document = doc_copy,
                        },
                    };

                    // Try to push to any worker
                    var pushed = false;
                    for (0..self.num_workers) |i| {
                        if (self.input_queues[i].push(new_msg)) {
                            pushed = true;
                            documents_submitted += 1;
                            break;
                        }
                    }

                    if (!pushed) {
                        // Couldn't push to any worker, free the document
                        self.allocator.free(doc_copy);
                        _ = self.pending_documents.remove(doc_id);
                    }
                } else if (documents_processed == documents_submitted) {
                    // No more documents and all processed
                    break;
                }
            }
        }

        // Shutdown all workers
        for (0..self.num_workers) |i| {
            const msg = Message{
                .Shutdown = i,
            };

            // Try to push shutdown message
            var pushed = false;
            while (!pushed) {
                pushed = self.input_queues[i].push(msg);
                if (!pushed) {
                    std.time.sleep(1 * std.time.ns_per_ms);
                }
            }
        }

        // Wait for all workers to join
        for (0..self.num_workers) |i| {
            if (self.workers[i].thread) |thread| {
                thread.join();
                self.workers[i].thread = null;
            }
        }

        self.processed_count += documents_processed;

        const elapsed = time.nanoTimestamp() - start_time;
        if (self.debug) {
            std.debug.print("[ParallelDP] Processed {d} documents in {d:.2}ms\n", .{ documents_processed, @as(f64, @floatFromInt(elapsed)) / time.ns_per_ms });
        }
    }
};
