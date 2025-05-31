const std = @import("std");
const Allocator = std.mem.Allocator;
const Thread = std.Thread;
const time = std.time;
const fs = std.fs;

const message = @import("message.zig");
const message_queue = @import("message_queue.zig");
const worker_mod = @import("worker.zig");
const fineweb = @import("data_loader.zig").FinewebDataLoader;
const CMS_F = @import("count_min_sketch.zig").CountMinSketch;

const EMPTY_SLICE: []const u8 = "";

/// Multi-threaded document processing coordinator
pub fn Coordinator(
    comptime cms_width: usize,
    comptime cms_depth: usize,
    comptime MY_LEN: comptime_int,
    comptime top_k: usize,
) type {
    // Import worker type
    const Worker = worker_mod.Worker(cms_width, cms_depth, MY_LEN, top_k, false);
    // Import CMS type
    const CMS = CMS_F(cms_width, cms_depth, MY_LEN);

    return struct {
        const Self = @This();

        /// Coordinator state
        const State = enum {
            /// Not started
            Idle,
            /// Loading documents and dispatching to workers
            FirstPass,
            /// Merging CMS from workers
            MergingCMS,
            /// Copying merged CMS to workers
            CopyingCMS,
            /// Start second pass
            StartSecondPass,
            /// Finding top K strings using shared CMS
            SecondPass,
            /// Merging results from workers
            MergingResults,
            /// Completed processing
            Complete,
            /// Encountered an error
            Error,
        };

        /// Allocator
        allocator: Allocator,

        /// Number of worker threads
        num_workers: usize,

        /// Worker threads - using non-nullable array
        workers: []*Worker,

        /// Input queues for workers - using value-based arrays
        input_queues: []message_queue.CoordinatorMessageQueue,

        /// Output queues from workers - using value-based arrays
        output_queues: []message_queue.WorkerMessageQueue,

        n_outstanding_jobs: []usize,

        /// Data loader
        data_loader: *fineweb,

        /// Global merged CMS
        global_cms: ?*CMS = null,

        /// Current state
        state: State = .Idle,

        /// Number of documents processed in first pass
        first_pass_count: usize = 0,

        /// stride for cms merge steps
        cms_merge_stride: usize = 1,

        /// Number of documents processed in second pass
        second_pass_count: usize = 0,

        /// Max documents to process for faster testing (set to 0 for unlimited)
        max_documents: usize = 0, // Process all available documents (no limit)

        /// Running flag
        running: bool = false,

        /// Debug flag
        debug: bool,

        /// Error message if any
        error_msg: ?[]const u8 = null,

        /// Queue depth - documents to maintain per worker
        queue_depth: usize,

        /// Documents waiting for acknowledgment from workers
        pending_documents: [][]const u8,

        /// free list for pending documents
        pending_documents_free_list: []usize,
        n_free_pending_documents: usize,

        /// Flag to track if we've reached the end of documents
        end_of_documents_reached: bool = false,

        /// Number of active workers
        active_workers: usize = 0,

        /// Initialize a new coordinator
        pub fn init(
            allocator: Allocator,
            num_workers: usize,
            data_loader: *fineweb,
            debug: bool,
        ) !*Self {
            const start_time = time.nanoTimestamp();

            // Create the coordinator
            const self = try allocator.create(Self);
            errdefer allocator.destroy(self);

            // Allocate arrays
            const workers = try allocator.alloc(*Worker, num_workers);
            errdefer allocator.free(workers);

            const input_queues = try allocator.alloc(message_queue.CoordinatorMessageQueue, num_workers);
            errdefer allocator.free(input_queues);

            const output_queues = try allocator.alloc(message_queue.WorkerMessageQueue, num_workers);
            errdefer allocator.free(output_queues);

            const n_outstanding_jobs = try allocator.alloc(usize, num_workers);
            errdefer allocator.free(n_outstanding_jobs);
            @memset(n_outstanding_jobs, 0);

            const queue_depth = 80;

            // Create pending documents list
            const pending_documents = try allocator.alloc([]const u8, num_workers * queue_depth);
            errdefer allocator.free(pending_documents);
            for (pending_documents) |*ptr| {
                ptr.* = EMPTY_SLICE;
            }

            const pending_documents_free_list = try allocator.alloc(usize, num_workers * queue_depth);
            errdefer allocator.free(pending_documents_free_list);
            for (pending_documents_free_list, 0..) |*ptr, i| {
                ptr.* = i;
            }

            // Initialize the coordinator with placeholder values
            self.* = .{
                .allocator = allocator,
                .num_workers = num_workers,
                .workers = workers,
                .input_queues = input_queues,
                .output_queues = output_queues,
                .data_loader = data_loader,
                .n_outstanding_jobs = n_outstanding_jobs,
                .debug = debug,
                .queue_depth = queue_depth,
                .pending_documents = pending_documents,
                .pending_documents_free_list = pending_documents_free_list,
                .n_free_pending_documents = num_workers * queue_depth,
                .max_documents = 0, // Process all available documents (no limit)
                .end_of_documents_reached = false,
                .active_workers = 0,
            };

            if (debug) {
                const elapsed = time.nanoTimestamp() - start_time;
                std.debug.print("[Coordinator] init: {d:.2}ms (processing all available documents)\n", .{
                    @as(f64, @floatFromInt(elapsed)) / time.ns_per_ms,
                });
            }

            return self;
        }

        /// Clean up resources
        pub fn deinit(self: *Self) void {
            const start_time = time.nanoTimestamp();

            // Clean up workers
            for (0..self.num_workers) |i| {
                self.workers[i].stop();
                self.workers[i].deinit();

                // Clean up queues
                self.input_queues[i].deinit();
                self.output_queues[i].deinit();
            }

            // Free the arrays
            self.allocator.free(self.workers);
            self.allocator.free(self.input_queues);
            self.allocator.free(self.output_queues);
            self.allocator.free(self.n_outstanding_jobs);

            // Free the global CMS if it exists
            if (self.global_cms) |cms| {
                cms.deinit();
                self.global_cms = null;
            }

            // Free any error message
            if (self.error_msg) |msg| {
                self.allocator.free(msg);
                self.error_msg = null;
            }

            // Free pending documents
            for (self.pending_documents) |doc| {
                if (doc.len > 0) {
                    self.allocator.free(doc);
                }
            }
            self.allocator.free(self.pending_documents);
            self.allocator.free(self.pending_documents_free_list);

            // Free the coordinator itself
            self.allocator.destroy(self);

            if (self.debug) {
                const elapsed = time.nanoTimestamp() - start_time;
                std.debug.print("[Coordinator] deinit: {d:.2}ms\n", .{@as(f64, @floatFromInt(elapsed)) / time.ns_per_ms});
            }
        }

        /// Start the coordinator
        pub fn start(self: *Self) !void {
            //const start_time = time.nanoTimestamp();

            // Always initialize workers - since we're making a single pass through the code
            try self.initWorkers();

            // Reset end-of-documents flag when starting a new pass
            self.end_of_documents_reached = false;

            // Set running flag
            self.running = true;

            if (self.debug) {
                //const elapsed = time.nanoTimestamp() - start_time;
                //std.debug.print("[Coordinator] start: {d:.2}ms\n", .{@as(f64, @floatFromInt(elapsed)) / time.ns_per_ms});
            }

            // Instead of spawning a thread, run the coordinator logic directly
            try self.run();
        }

        /// Initialize the workers
        fn initWorkers(self: *Self) !void {
            //const start_time = time.nanoTimestamp();

            // Create and start each worker
            for (0..self.num_workers) |i| {
                // Create message queues as values
                self.input_queues[i] = try message_queue.CoordinatorMessageQueue.init(self.allocator);
                self.output_queues[i] = try message_queue.WorkerMessageQueue.init(self.allocator);

                // Create worker with references to the queues
                self.workers[i] = try Worker.init(self.allocator, i, &self.input_queues[i], &self.output_queues[i]);

                // Start worker
                try self.workers[i].start();
            }

            if (self.debug) {
                //const elapsed = time.nanoTimestamp() - start_time;
                //std.debug.print("[Coordinator] initWorkers ({d} workers): {d:.2}ms\n", .{ self.num_workers, @as(f64, @floatFromInt(elapsed)) / time.ns_per_ms });
            }
        }

        /// Stop the coordinator
        pub fn stop(self: *Self) void {
            self.running = false;

            if (self.debug) {
                std.debug.print("[Coordinator] stopping\n", .{});
            }
        }

        /// Wait for the coordinator to complete - now a no-op since we run directly
        pub fn wait(self: *Self) void {
            // Previously would join the thread, now we run directly
            _ = self;
        }

        /// Handle a message from a worker
        fn handleWorkerMessage(self: *Self, msg: message.WorkerMessage) !void {
            const start_time = time.nanoTimestamp();

            // Add detailed message info logging
            if (self.debug) {
                //std.debug.print("[Coordinator] [DEBUG] Handling message from Worker, type: {s}\n", .{@tagName(msg)});
            }

            defer message.freeWorkerMessage(self.allocator, &msg);

            switch (msg) {
                .DocumentProcessed => |processed_data| {
                    const document = processed_data.document;
                    const pass = processed_data.pass;

                    // Remove from pending documents
                    // Check pending documents array
                    const index = processed_data.document_id;
                    self.allocator.free(document);
                    self.pending_documents_free_list[self.n_free_pending_documents] = index;
                    self.pending_documents[index] = EMPTY_SLICE;
                    self.n_free_pending_documents += 1;

                    // Count processed documents
                    if (pass == 1) {
                        self.first_pass_count += 1;

                        if (self.debug and (self.first_pass_count % 10000 == 0 or self.first_pass_count < 10)) {
                            std.debug.print("[Coordinator] First pass processed: {d} documents\n", .{self.first_pass_count});
                            const n_pending = self.pending_documents_free_list.len - self.n_free_pending_documents;
                            std.debug.print("[Coordinator] [DEBUG] Removed document from pending list, {d} remaining\n", .{n_pending});
                        }
                    } else if (pass == 2) {
                        self.second_pass_count += 1;

                        if (self.debug and (self.second_pass_count % 10000 == 0 or self.second_pass_count < 10)) {
                            std.debug.print("[Coordinator] Second pass processed: {d} documents\n", .{self.second_pass_count});
                            const n_pending = self.pending_documents_free_list.len - self.n_free_pending_documents;
                            std.debug.print("[Coordinator] [DEBUG] Removed document from pending list, {d} remaining\n", .{n_pending});
                        }
                    }
                },
                .MergedCounts => {
                    // Worker has completed merging Counts data
                    if (self.getTotalOutstandingJobs() == 0) {
                        self.cms_merge_stride *= 2;
                        if (!self.sendMergeCountsMessages()) {
                            self.state = .Complete;
                            self.running = false;
                        }
                    }
                },
                .MergedCMS => |_| {
                    if (self.getTotalOutstandingJobs() == 0) {
                        self.cms_merge_stride *= 2;
                        if (!self.sendMergeCMSMessages()) {
                            self.state = .CopyingCMS;
                            self.sendCopyCMSMessages();
                        }
                    }
                },
                .CopiedCMS => |_| {
                    // If all workers have copied their CMS data, skip second pass and go to Complete
                    if (self.getTotalOutstandingJobs() == 0) {
                        self.state = .Complete;
                        self.running = false;
                    }
                },
                .StateDumped => {
                    // Worker has dumped its state
                    if (self.debug) {
                        std.debug.print("[Coordinator] Worker dumped state\n", .{});
                    }
                },
                .Error => |error_data| {
                    // Worker encountered an error
                    const worker_id = error_data.worker_id; // We still need worker_id for error reporting

                    if (error_data.getErrorMessage()) |error_text| {
                        const error_copy = try std.fmt.allocPrint(self.allocator, "Worker {d} error: {s}", .{ worker_id, error_text });

                        if (self.error_msg) |old_error| {
                            self.allocator.free(old_error);
                        }

                        self.error_msg = error_copy;
                        self.state = .Error;

                        if (self.debug) {
                            std.debug.print("[Coordinator] {s}\n", .{error_copy});
                        }
                    }
                },
            }

            const elapsed = time.nanoTimestamp() - start_time;
            if (self.debug and elapsed > 50 * time.ns_per_ms) { // Log if handling took more than 50ms
                std.debug.print("[Coordinator] [DEBUG] handleWorkerMessage for {s} took {d:.2}ms\n", .{ @tagName(msg), @as(f64, @floatFromInt(elapsed)) / time.ns_per_ms });
            }
        }

        /// Start the second pass
        fn startSecondPass(self: *Self) !void {
            if (self.debug) {
                std.debug.print("[Coordinator] Starting second pass\n", .{});
            }

            // OPTIMIZATION: Rewind the existing data loader instead of creating a new one
            const rewind_start = time.nanoTimestamp();
            try self.data_loader.rewind();
            const rewind_time = time.nanoTimestamp() - rewind_start;

            if (self.debug) {
                std.debug.print("[Coordinator] Data loader rewound in {d:.2}ms\n", .{@as(f64, @floatFromInt(rewind_time)) / time.ns_per_ms});
            }

            // Change state
            self.state = .SecondPass;

            // Reset document count and end-of-documents flag for second pass
            self.second_pass_count = 0;
            self.end_of_documents_reached = false;

            // Note: Initial document feeding happens in the main run loop
        }

        /// Get the next document from the data loader
        fn getNextDocument(self: *Self) !?[]const u8 {
            // Check if we've reached the document limit
            const current_count = if (self.state == .FirstPass) self.first_pass_count else self.second_pass_count;

            // If max_documents is set and we've reached the limit, stop
            if (self.max_documents > 0 and current_count >= self.max_documents) {
                if (self.debug) {
                    std.debug.print("[Coordinator] Reached document limit ({d}), stopping\n", .{self.max_documents});
                }
                return null;
            }

            // Add logging for tracking progression through large datasets
            if (current_count > 0 and current_count % 50000 == 0) {
                std.debug.print("[Coordinator] Processed {d} documents in current pass\n", .{current_count});
            }

            const doc = try self.data_loader.nextDocumentString();
            if (doc == null) {
                if (self.debug) {
                    std.debug.print("[Coordinator] End of documents reached after {d} documents\n", .{current_count});
                }
                self.end_of_documents_reached = true;
            }
            return doc;
        }

        fn sendCopyCMSMessages(self: *Self) void {
            for (1..self.num_workers) |i| {
                const msg = message.createCopyCMSMessage(i, self.workers[0].sfm.cms);
                _ = self.sendMessageToWorker(i, msg, true); // Expect response
            }
        }

        fn sendMergeCMSMessages(self: *Self) bool {
            const mask = self.cms_merge_stride * 2 - 1;
            const stride = self.cms_merge_stride;
            var i: usize = 0;
            var any_sent = false;
            while (i < self.num_workers) : (i += 1) {
                if (i & mask == 0 and i + stride < self.num_workers) {
                    const msg = message.createCMSMergeMessage(i, self.workers[i + stride].sfm.cms);
                    _ = self.sendMessageToWorker(i, msg, true); // Expect response
                    any_sent = true;
                    if (self.debug) {
                        std.debug.print("[Coordinator] Sent CMS merge message to worker {d}, {d}\n", .{ i, i + stride });
                    }
                }
            }
            return any_sent;
        }

        fn sendMergeCountsMessages(self: *Self) bool {
            const mask = self.cms_merge_stride * 2 - 1;
            const stride = self.cms_merge_stride;
            var i: usize = 0;
            var any_sent = false;
            while (i < self.num_workers) : (i += 1) {
                if (i & mask == 0 and i + stride < self.num_workers) {
                    const msg = message.createMergeCountsMessage(i, self.workers[i + stride].sfm);
                    _ = self.sendMessageToWorker(i, msg, true); // Expect response
                    any_sent = true;
                    if (self.debug) {
                        std.debug.print("[Coordinator] Sent Counts merge message to worker {d}, {d}\n", .{ i, i + stride });
                    }
                }
            }
            return any_sent;
        }

        fn getTotalOutstandingJobs(self: *Self) usize {
            var total: usize = 0;
            for (self.n_outstanding_jobs) |n| {
                total += n;
            }
            return total;
        }

        fn transitionToMergingCMS(self: *Self) !void {
            if (self.debug) {
                std.debug.print("[Coordinator] [DEBUG] Starting transition to MergingCMS state\n", .{});
            }

            // All documents processed in first pass, transition to merging CMS
            self.state = .MergingCMS;

            if (self.debug) {
                std.debug.print("[Coordinator] First pass complete ({d} documents), starting CMS merges\n", .{self.first_pass_count});
                const n_pending = self.pending_documents_free_list.len - self.n_free_pending_documents;
                std.debug.print("[Coordinator] [DEBUG] Pending documents: {d}, State: {s}\n", .{ n_pending, @tagName(self.state) });
            }

            _ = self.sendMergeCMSMessages();
        }

        /// Transition to the next state if needed
        fn transitionStateIfNeeded(self: *Self) !void {
            const old_state = self.state;

            switch (self.state) {
                .Idle => {
                    // Start first pass
                    self.state = .FirstPass;

                    if (self.debug) {
                        std.debug.print("[Coordinator] Starting first pass\n", .{});
                    }
                },
                .FirstPass => {
                    // All documents processed in first pass, transition to merging CMS
                    try self.transitionToMergingCMS();
                },
                .MergingCMS => {
                    // In MergingCMS state we don't transition automatically - we wait for all
                    // workers to provide their CMS data which is handled via the
                    // ProvideCMS message handler
                    if (self.debug) {
                        std.debug.print("[Coordinator] CMS merging in progress\n", .{});
                    }
                },
                .CopyingCMS => {
                    if (self.debug) {
                        std.debug.print("[Coordinator] Copying CMS in progress\n", .{});
                    }
                },
                .StartSecondPass => {
                    // Start second pass
                    self.state = .SecondPass;

                    if (self.debug) {
                        std.debug.print("[Coordinator] Starting second pass\n", .{});
                    }
                },
                .SecondPass => {
                    // All documents processed in second pass, transition to merging results
                    self.state = .MergingResults;

                    if (self.debug) {
                        std.debug.print("[Coordinator] Second pass complete ({d} documents), merging results\n", .{self.second_pass_count});
                    }

                    self.cms_merge_stride = 1;
                    _ = self.sendMergeCountsMessages();
                },
                .MergingResults => {
                    // All results merged, transition to complete
                    self.state = .Complete;

                    if (self.debug) {
                        std.debug.print("[Coordinator] Processing complete\n", .{});
                    }

                    // Stop the coordinator
                    self.running = false;
                },
                .Complete, .Error => {
                    // Already in a terminal state
                },
            }

            // Log state transitions
            if (self.state != old_state) {
                if (self.debug) {
                    std.debug.print("[Coordinator] State transition: {s} -> {s}\n", .{ @tagName(old_state), @tagName(self.state) });
                }
            }
        }

        fn addToPendingDocuments(self: *Self, document: []const u8) usize {
            self.n_free_pending_documents -= 1;
            const index = self.pending_documents_free_list[self.n_free_pending_documents];
            self.pending_documents[index] = document;
            return index;
        }

        pub fn runSecondPass(self: *Self) !void {
            self.state = .StartSecondPass;
            self.end_of_documents_reached = false;
            self.running = true;
            self.second_pass_count = 0;

            // Run the coordinator
            try self.run();

            // After run completes, ensure proper shutdown
            if (self.state != .Complete and self.state != .Error) {
                self.state = .Complete;
            }
        }

        /// Send a message to a worker and track it if a response is expected
        fn sendMessageToWorker(self: *Self, worker_id: usize, msg: message.CoordinatorMessage, expect_response: bool) bool {
            const result = self.input_queues[worker_id].push(msg);
            if (result and expect_response) {
                self.n_outstanding_jobs[worker_id] += 1;
            }
            return result;
        }

        /// Send shutdown messages to all workers
        pub fn shutdownWorkers(self: *Self) void {
            for (0..self.num_workers) |i| {
                while (self.n_outstanding_jobs[i] > 0) {
                    if (self.output_queues[i].pop()) |worker_msg| {
                        self.n_outstanding_jobs[i] -= 1;
                        message.freeWorkerMessage(self.allocator, &worker_msg);
                    } else {
                        std.time.sleep(1 * std.time.ns_per_ms);
                    }
                }
            }

            for (0..self.num_workers) |i| {
                const msg = message.createShutdownMessage(i);
                _ = self.sendMessageToWorker(i, msg, false); // Don't expect response
            }
        }

        /// Main coordinator loop
        fn run(self: *Self) !void {
            if (self.debug) {
                std.debug.print("[Coordinator] started running\n", .{});
            }

            // Reset state if we're restarting
            if (self.state == .Complete or self.state == .Error) {
                self.state = .Idle;
                self.first_pass_count = 0;
                self.second_pass_count = 0;
                self.end_of_documents_reached = false;
            }

            // Transition to first pass if we're idle, otherwise continue where we left off
            if (self.state == .Idle or self.state == .StartSecondPass) {
                try self.transitionStateIfNeeded();
            }

            // Initial document feed for first or second pass - push K documents to each worker
            if (self.state == .FirstPass or self.state == .SecondPass) {
                // Push K documents to each worker initially
                for (0..self.num_workers) |i| {
                    const documents_to_push = self.queue_depth;
                    var pushed: usize = 0;

                    while (pushed < documents_to_push) {
                        const doc = try self.getNextDocument();
                        if (doc == null) {
                            // No more documents to push
                            self.end_of_documents_reached = true;
                            break;
                        }

                        // Add to pending documents
                        const doc_id = self.addToPendingDocuments(doc.?);

                        // Create a message for the worker
                        const pass: u8 = if (self.state == .FirstPass) 1 else 2;
                        const msg = message.createProcessDocumentMessage(i, doc_id, doc.?, pass);

                        const result = self.sendMessageToWorker(i, msg, true); // Expect response
                        pushed += @intFromBool(result);

                        if (self.debug and pushed == 1) {
                            std.debug.print("[Coordinator] Initial document push: sent document to worker {d} (pass {d})\n", .{ i, pass });
                        }
                    }

                    if (self.debug) {
                        std.debug.print("[Coordinator] Initially pushed {d} documents to worker {d}\n", .{ pushed, i });
                    }
                }
            }

            // Main loop with improved worker feeding logic
            const start_time = std.time.nanoTimestamp();
            const max_runtime_ms: u64 = 60 * 60 * 1000; // 1 hour max runtime
            var last_progress_time = start_time;
            var last_progress_count: usize = 0;

            while (self.running) {
                var did_anything = false;

                // Loop through each worker's output queue
                for (0..self.num_workers) |i| {
                    // Process all available messages from this worker
                    if (self.output_queues[i].pop()) |msg| {
                        self.n_outstanding_jobs[i] -= 1;
                        did_anything = true;

                        // Handle the message (documents processed, errors, etc.)
                        try self.handleWorkerMessage(msg);
                    }
                    if ((self.state == .FirstPass or self.state == .SecondPass) and !self.end_of_documents_reached and self.n_outstanding_jobs[i] < self.queue_depth) {
                        const doc = try self.getNextDocument();
                        if (doc) |document| {
                            did_anything = true;
                            // Add to pending documents
                            const doc_id = self.addToPendingDocuments(document);

                            // Create a message for the worker
                            const pass: u8 = if (self.state == .FirstPass) 1 else 2;
                            const new_msg = message.createProcessDocumentMessage(i, doc_id, document, pass);

                            _ = self.sendMessageToWorker(i, new_msg, true); // Expect response

                            if (self.debug and (self.first_pass_count + self.second_pass_count) % 10000 == 0) {
                                std.debug.print("[Coordinator] Fed another document to worker {d} (pass {d})\n", .{ i, pass });
                            }
                        } else {
                            // No more documents, mark that we've reached the end
                            self.end_of_documents_reached = true;

                            if (self.debug) {
                                std.debug.print("[Coordinator] No more documents to process\n", .{});
                            }
                        }
                    }
                }

                // If we're in first or second pass, check if we should transition to the next state
                if ((self.state == .FirstPass or self.state == .SecondPass) and self.end_of_documents_reached) {
                    // Check if all workers are idle (no more documents to process)
                    var all_workers_idle = true;
                    var active_workers: usize = 0;

                    for (0..self.num_workers) |i| {
                        if (self.n_outstanding_jobs[i] > 0) {
                            all_workers_idle = false;
                            active_workers += 1;
                        }
                    }

                    const n_pending = self.pending_documents_free_list.len - self.n_free_pending_documents;
                    if (all_workers_idle and n_pending == 0) {
                        // All documents have been processed and acknowledged, transition to next state
                        if (self.debug) {
                            std.debug.print("[Coordinator] All documents processed, transitioning to next phase\n", .{});
                        }
                        try self.transitionStateIfNeeded();
                    } else if (self.debug and (self.first_pass_count + self.second_pass_count) % 10000 == 0) {
                        std.debug.print("[Coordinator] Waiting for {d} workers to finish, {d} pending documents\n", .{ active_workers, n_pending });
                    }
                }

                // Update progress tracking
                const current_count = if (self.state == .FirstPass) self.first_pass_count else self.second_pass_count;
                if (current_count > last_progress_count) {
                    last_progress_count = current_count;
                    last_progress_time = std.time.nanoTimestamp();
                }

                // Check for timeout or stuck condition
                const elapsed = std.time.nanoTimestamp() - start_time;
                const elapsed_ms = @as(u64, @intCast(@divFloor(elapsed, std.time.ns_per_ms)));

                // Check for stuck condition - no progress for 30 seconds
                // const since_last_progress = std.time.nanoTimestamp() - last_progress_time;
                // const since_last_progress_ms = @as(u64, @intCast(@divFloor(since_last_progress, std.time.ns_per_ms)));

                // if (!did_anything and since_last_progress_ms > 30 * 1000) {
                //     std.debug.print("[Coordinator] No progress for {d} seconds in state {s}, forcing state transition\n", .{ since_last_progress_ms / 1000, @tagName(self.state) });
                //     try self.transitionStateIfNeeded();
                //     last_progress_time = std.time.nanoTimestamp();
                // }

                // Check for timeout
                if (elapsed_ms > max_runtime_ms) {
                    std.debug.print("[Coordinator] Maximum runtime exceeded ({d} ms), forcing completion\n", .{elapsed_ms});
                    self.state = .Complete;
                    self.running = false;
                    break;
                }

                // If we didn't process any messages, sleep a bit to avoid spinning
                if (!did_anything) {
                    std.time.sleep(1);
                }
            }

            // Shut down workers
            // for (0..self.num_workers) |i| {
            //     const msg = message.createShutdownMessage(i);
            //     const result = self.input_queues[i].push(msg);
            //     const n_pushed: usize = if (result) 1 else 0;
            //     self.n_outstanding_jobs[i] += n_pushed;
            // }

            if (self.debug) {
                std.debug.print("[Coordinator] stopped running\n", .{});
            }
        }
    };
}
