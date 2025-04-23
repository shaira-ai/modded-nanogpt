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

/// Multi-threaded document processing coordinator
pub fn Coordinator(
    comptime cms_width: usize,
    comptime cms_depth: usize,
    comptime min_length: usize,
    comptime max_length: usize,
) type {
    // Import worker type
    const Worker = worker_mod.Worker(cms_width, cms_depth, min_length, max_length);
    // Import CMS type
    const CMS = CMS_F(cms_width, cms_depth);

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

        /// Worker threads
        workers: []?*Worker,

        /// Input queues for workers
        input_queues: []?*message_queue.CoordinatorMessageQueue,

        /// Output queues from workers
        output_queues: []?*message_queue.WorkerMessageQueue,

        /// Data loader
        data_loader: *fineweb,

        /// Global merged CMS
        global_cms: ?*CMS = null,

        /// Current state
        state: State = .Idle,

        /// Number of documents processed in first pass
        first_pass_count: usize = 0,

        /// Number of documents processed in second pass
        second_pass_count: usize = 0,

        /// Max documents to process for faster testing (set to 0 for unlimited)
        max_documents: usize = 0, // Process all available documents (no limit)

        /// Top-K strings to track per length
        top_k: usize,

        /// Coordinator thread
        thread: ?Thread = null,

        /// Running flag
        running: bool = false,

        /// Debug flag
        debug: bool,

        /// Error message if any
        error_msg: ?[]const u8 = null,

        /// Queue depth - documents to maintain per worker
        queue_depth: usize = 10,

        /// Documents waiting for acknowledgment from workers
        pending_documents: std.ArrayList([]const u8),

        /// Flag to track if we've reached the end of documents
        end_of_documents_reached: bool = false,

        /// Number of active workers
        active_workers: usize = 0,

        /// Count of CMS merge completions received from workers
        cms_merge_completions: usize = 0,

        /// Initialize a new coordinator
        pub fn init(
            allocator: Allocator,
            num_workers: usize,
            data_loader: *fineweb,
            top_k: usize,
            debug: bool,
        ) !*Self {
            const start_time = time.nanoTimestamp();

            // Create the coordinator
            const self = try allocator.create(Self);
            errdefer allocator.destroy(self);

            // Allocate arrays
            const workers = try allocator.alloc(?*Worker, num_workers);
            errdefer allocator.free(workers);

            const input_queues = try allocator.alloc(?*message_queue.CoordinatorMessageQueue, num_workers);
            errdefer allocator.free(input_queues);

            const output_queues = try allocator.alloc(?*message_queue.WorkerMessageQueue, num_workers);
            errdefer allocator.free(output_queues);

            // Initialize arrays to null
            @memset(workers, null);
            @memset(input_queues, null);
            @memset(output_queues, null);

            // Create pending documents list
            const pending_documents = std.ArrayList([]const u8).init(allocator);

            // Initialize the coordinator
            self.* = .{
                .allocator = allocator,
                .num_workers = num_workers,
                .workers = workers,
                .input_queues = input_queues,
                .output_queues = output_queues,
                .data_loader = data_loader,
                .top_k = top_k,
                .debug = debug,
                .pending_documents = pending_documents,
                .max_documents = 0, // Process all available documents (no limit)
                .end_of_documents_reached = false,
                .active_workers = 0,
                .cms_merge_completions = 0,
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

            // Stop any running thread
            if (self.thread != null) {
                self.running = false;
                self.thread.?.join();
                self.thread = null;
            }

            // Clean up workers
            for (0..self.num_workers) |i| {
                if (self.workers[i]) |worker| {
                    worker.stop();
                    worker.deinit();
                    self.workers[i] = null;
                }

                if (self.input_queues[i]) |queue| {
                    queue.deinit();
                    self.input_queues[i] = null;
                }

                if (self.output_queues[i]) |queue| {
                    queue.deinit();
                    self.output_queues[i] = null;
                }
            }

            // Free the arrays
            self.allocator.free(self.workers);
            self.allocator.free(self.input_queues);
            self.allocator.free(self.output_queues);

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
            for (self.pending_documents.items) |doc| {
                self.allocator.free(doc);
            }
            self.pending_documents.deinit();

            // Free the coordinator itself
            self.allocator.destroy(self);

            if (self.debug) {
                const elapsed = time.nanoTimestamp() - start_time;
                std.debug.print("[Coordinator] deinit: {d:.2}ms\n", .{@as(f64, @floatFromInt(elapsed)) / time.ns_per_ms});
            }
        }

        /// Start the coordinator
        pub fn start(self: *Self) !void {
            const start_time = time.nanoTimestamp();

            // Check if we're already running
            if (self.thread != null) {
                // If we have a thread, wait for it to finish
                self.thread.?.join();
                self.thread = null;
            }

            // Create workers if needed (they might have been shut down)
            if (self.workers[0] == null) {
                try self.initWorkers();
            }

            // Reset end-of-documents flag when starting a new pass
            self.end_of_documents_reached = false;
            self.cms_merge_completions = 0;

            // Start the coordinator thread
            self.running = true;
            self.thread = try Thread.spawn(.{}, Self.run, .{self});

            if (self.debug) {
                const elapsed = time.nanoTimestamp() - start_time;
                std.debug.print("[Coordinator] start: {d:.2}ms\n", .{@as(f64, @floatFromInt(elapsed)) / time.ns_per_ms});
            }
        }

        /// Initialize the workers
        fn initWorkers(self: *Self) !void {
            const start_time = time.nanoTimestamp();

            // Create and start each worker
            for (0..self.num_workers) |i| {
                // Create message queues
                const input_queue = try message_queue.CoordinatorMessageQueue.init(self.allocator);
                const output_queue = try message_queue.WorkerMessageQueue.init(self.allocator);

                self.input_queues[i] = input_queue;
                self.output_queues[i] = output_queue;

                // Create worker
                const worker = try Worker.init(self.allocator, i, input_queue, output_queue, self.top_k, self.debug);

                self.workers[i] = worker;

                // Start worker
                try worker.start();
            }

            if (self.debug) {
                const elapsed = time.nanoTimestamp() - start_time;
                std.debug.print("[Coordinator] initWorkers ({d} workers): {d:.2}ms\n", .{ self.num_workers, @as(f64, @floatFromInt(elapsed)) / time.ns_per_ms });
            }
        }

        /// Stop the coordinator
        pub fn stop(self: *Self) void {
            self.running = false;

            if (self.debug) {
                std.debug.print("[Coordinator] stopping\n", .{});
            }
        }

        /// Wait for the coordinator to complete
        pub fn wait(self: *Self) void {
            if (self.thread) |thread| {
                thread.join();
                self.thread = null;
            }
        }

        /// Monitor worker responses and handle them
        fn handleWorkerResponses(self: *Self) !void {
            var all_workers_idle = true;
            var queued_work = false;
            var active_workers: usize = 0;

            // Check each worker's output queue
            for (0..self.num_workers) |i| {
                if (self.output_queues[i]) |queue| {
                    // Process all available messages
                    while (queue.pop()) |msg| {
                        all_workers_idle = false;

                        // Handle the message
                        try self.handleWorkerMessage(msg);
                    }

                    // Check if this worker has queued work
                    const input_queue = self.input_queues[i].?;
                    if (input_queue.count() > 0) {
                        queued_work = true;
                        all_workers_idle = false;
                        active_workers += 1;
                    }
                }
            }

            self.active_workers = active_workers;

            // Check if we should transition state
            // Either all workers are idle and no pending documents, OR we've hit the document limit
            // OR we've processed a very large number of documents (safeguard)
            const at_document_limit = (self.max_documents > 0) and
                ((self.state == .FirstPass and self.first_pass_count >= self.max_documents) or
                    (self.state == .SecondPass and self.second_pass_count >= self.max_documents));

            const force_transition = (self.state == .FirstPass and self.first_pass_count > 500000) or
                (self.state == .SecondPass and self.second_pass_count > 500000);

            // Additional conditions to detect end of documents:
            // 1. We have reached end of documents AND
            // 2. Either all workers are idle OR pending documents list is empty
            const end_of_phase = self.end_of_documents_reached and
                (all_workers_idle or (active_workers == 0 and self.pending_documents.items.len == 0));

            if (end_of_phase and self.debug) {
                std.debug.print("[Coordinator] Detected end of documents - all {d} documents processed. Transitioning to next phase.\n", .{if (self.state == .FirstPass) self.first_pass_count else self.second_pass_count});
            }

            if ((all_workers_idle and !queued_work and self.pending_documents.items.len == 0) or
                at_document_limit or
                force_transition or
                end_of_phase)
            {
                if (force_transition and self.debug) {
                    std.debug.print("[Coordinator] Forcing state transition after processing {d} documents\n", .{if (self.state == .FirstPass) self.first_pass_count else self.second_pass_count});
                }
                try self.transitionStateIfNeeded();
            }
        }

        /// Handle a message from a worker
        fn handleWorkerMessage(self: *Self, msg: message.WorkerMessage) !void {
            defer message.freeWorkerMessage(self.allocator, &msg);

            switch (msg.msg_type) {
                .DocumentProcessed => {
                    // Remove from pending documents
                    if (msg.document != null and msg.pass != null) {
                        const pass = msg.pass.?;

                        // Count processed documents
                        if (pass == 1) {
                            self.first_pass_count += 1;

                            if (self.debug and self.first_pass_count % 10000 == 0) {
                                std.debug.print("[Coordinator] First pass processed: {d} documents\n", .{self.first_pass_count});
                            }
                        } else if (pass == 2) {
                            self.second_pass_count += 1;

                            if (self.debug and self.second_pass_count % 10000 == 0) {
                                std.debug.print("[Coordinator] Second pass processed: {d} documents\n", .{self.second_pass_count});
                            }
                        }

                        // Attempt to feed another document to the worker
                        const fed_document = try self.feedWorkerIfNeeded(msg.worker_id);

                        // If we couldn't feed a document and this is the first time,
                        // log that we've reached the end of the data stream
                        if (!fed_document and !self.end_of_documents_reached) {
                            self.end_of_documents_reached = true;
                            if (self.debug) {
                                std.debug.print("[Coordinator] End of document stream reached after {d} documents\n", .{if (self.state == .FirstPass) self.first_pass_count else self.second_pass_count});
                            }
                        }
                    }
                },
                .TopKComplete => {
                    // Worker has completed finding top-K strings
                    if (self.debug) {
                        std.debug.print("[Coordinator] Worker {d} completed finding top-K strings\n", .{msg.worker_id});
                    }

                    // Check if all workers are done
                    var all_complete = true;
                    for (0..self.num_workers) |i| {
                        if (self.workers[i]) |_| {
                            // Check if this worker has pending messages
                            if (self.input_queues[i].?.count() > 0) {
                                all_complete = false;
                                break;
                            }
                        }
                    }

                    if (all_complete and self.state == .MergingResults) {
                        // Force a state transition
                        self.state = .Complete;

                        if (self.debug) {
                            std.debug.print("[Coordinator] All workers completed finding top-K strings\n", .{});
                        }

                        // Stop the coordinator
                        self.running = false;
                    }
                },
                .CMSMergeComplete => {
                    // Worker has completed merging its CMS
                    self.cms_merge_completions += 1;

                    if (self.debug) {
                        std.debug.print("[Coordinator] Worker {d} completed CMS merge ({d}/{d} workers done)\n", .{ msg.worker_id, self.cms_merge_completions, self.num_workers });
                    }

                    // If all workers are done with CMS merge, transition to second pass
                    if (self.cms_merge_completions == self.num_workers and self.state == .MergingCMS) {
                        if (self.debug) {
                            std.debug.print("[Coordinator] All workers completed CMS merge, transitioning to Second Pass\n", .{});
                        }

                        try self.startSecondPass();
                    }
                },
                .StateDumped => {
                    // Worker has dumped its state
                    if (self.debug) {
                        std.debug.print("[Coordinator] Worker {d} dumped state\n", .{msg.worker_id});
                    }
                },
                .Error => {
                    // Worker encountered an error
                    if (msg.error_msg != null) {
                        const error_text = try std.fmt.allocPrint(self.allocator, "Worker {d} error: {s}", .{ msg.worker_id, msg.error_msg.? });

                        if (self.error_msg) |old_error| {
                            self.allocator.free(old_error);
                        }

                        self.error_msg = error_text;
                        self.state = .Error;

                        if (self.debug) {
                            std.debug.print("[Coordinator] {s}\n", .{error_text});
                        }
                    }
                },
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

            // Feed initial documents to workers
            for (0..self.num_workers) |i| {
                for (0..self.queue_depth) |_| {
                    _ = try self.feedWorkerIfNeeded(i);
                }
            }
        }

        /// Feed a document to a worker if one is available
        /// Returns true if a document was fed, false if no more documents available
        fn feedWorkerIfNeeded(self: *Self, worker_id: usize) !bool {
            // Only feed in first or second pass
            if (self.state != .FirstPass and self.state != .SecondPass) {
                return false;
            }

            const input_queue = self.input_queues[worker_id].?;

            // Check if the worker needs documents
            if (input_queue.count() >= self.queue_depth) {
                return false;
            }

            // Get the next document
            const doc = try self.getNextDocument();
            if (doc == null) {
                // No more documents
                return false;
            }

            // Add to pending documents
            try self.pending_documents.append(doc.?);

            // Create a message for the worker
            const pass: u8 = if (self.state == .FirstPass) 1 else 2;
            const msg = message.createProcessDocumentMessage(worker_id, doc.?, pass);

            // Send the message
            _ = input_queue.push(msg);

            if (self.debug) {
                std.debug.print("[Coordinator] Sent document to worker {d} (pass {d}, {d} bytes)\n", .{ worker_id, pass, doc.?.len });
            }

            return true;
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
                    self.state = .MergingCMS;

                    if (self.debug) {
                        std.debug.print("[Coordinator] First pass complete ({d} documents), merging CMS\n", .{self.first_pass_count});
                    }

                    // Create the global CMS
                    self.global_cms = try CMS.init(self.allocator);

                    // Send merge CMS messages to all workers
                    for (0..self.num_workers) |i| {
                        const input_queue = self.input_queues[i].?;
                        const msg = message.createMergeCMSMessage(i, @as(*anyopaque, @ptrCast(self.global_cms.?)));
                        _ = input_queue.push(msg);
                    }

                    // Reset the CMS merge completions counter
                    self.cms_merge_completions = 0;

                    // Set a reminder to check for all workers completing
                    if (self.debug) {
                        std.debug.print("[Coordinator] Sent CMS merge messages to all workers\n", .{});
                    }
                },
                .MergingCMS => {
                    // In MergingCMS state we don't transition automatically - we wait for all
                    // workers to complete their merge operations which is handled via the
                    // CMSMergeComplete message handler
                    if (self.debug) {
                        std.debug.print("[Coordinator] CMS merging in progress ({d}/{d} workers completed)\n", .{ self.cms_merge_completions, self.num_workers });
                    }
                },
                .SecondPass => {
                    // All documents processed in second pass, transition to merging results
                    self.state = .MergingResults;

                    if (self.debug) {
                        std.debug.print("[Coordinator] Second pass complete ({d} documents), merging results\n", .{self.second_pass_count});
                    }

                    // Send find top-K messages to all workers
                    for (0..self.num_workers) |i| {
                        const input_queue = self.input_queues[i].?;
                        const msg = message.createFindTopKMessage(i);
                        _ = input_queue.push(msg);
                    }
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
                self.cms_merge_completions = 0;
            }

            // Transition to first pass if we're idle, otherwise continue where we left off
            if (self.state == .Idle) {
                try self.transitionStateIfNeeded();
            }

            // Initial document feed for first or second pass
            if (self.state == .FirstPass or self.state == .SecondPass) {
                for (0..self.num_workers) |i| {
                    for (0..self.queue_depth) |_| {
                        _ = try self.feedWorkerIfNeeded(i);
                    }
                }
            }

            // Main loop with timeout and stuck-detection
            const start_time = std.time.nanoTimestamp();
            const max_runtime_ms: u64 = 60 * 60 * 1000; // 1 hour max runtime
            var last_progress_time = start_time;
            var last_progress_count: usize = 0;
            var consecutive_no_message_count: usize = 0;
            const max_no_message_iterations: usize = 1000; // Force transition after 1000 iterations with no progress

            while (self.running) {
                var had_messages = false;

                // Process messages from all output queues
                for (0..self.num_workers) |i| {
                    if (self.output_queues[i]) |queue| {
                        if (queue.count() > 0) {
                            had_messages = true;
                            break;
                        }
                    }
                }

                // Handle worker responses
                try self.handleWorkerResponses();

                // If we had no messages, increment counter
                if (!had_messages) {
                    consecutive_no_message_count += 1;
                } else {
                    consecutive_no_message_count = 0;
                }

                // If we've gone many iterations with no messages, force state transition
                if (consecutive_no_message_count > max_no_message_iterations) {
                    if (self.debug) {
                        std.debug.print("[Coordinator] No activity for {d} iterations, forcing state transition from {s}\n", .{ consecutive_no_message_count, @tagName(self.state) });
                    }
                    try self.transitionStateIfNeeded();
                    consecutive_no_message_count = 0;
                }

                // Check for timeout
                const elapsed = std.time.nanoTimestamp() - start_time;
                const elapsed_ms = @as(u64, @intCast(@divFloor(elapsed, std.time.ns_per_ms)));

                // Detect if we're stuck (no progress for 30 seconds)
                const current_count = if (self.state == .FirstPass) self.first_pass_count else self.second_pass_count;
                const since_last_progress = std.time.nanoTimestamp() - last_progress_time;
                const since_last_progress_ms = @as(u64, @intCast(@divFloor(since_last_progress, std.time.ns_per_ms)));

                if (current_count > last_progress_count) {
                    last_progress_count = current_count;
                    last_progress_time = std.time.nanoTimestamp();
                } else if (since_last_progress_ms > 30 * 1000) { // 30 seconds with no progress
                    std.debug.print("[Coordinator] No progress for {d} seconds in state {s}, forcing state transition\n", .{ since_last_progress_ms / 1000, @tagName(self.state) });

                    // If we've reached the end of documents, this is likely the cause
                    if (self.end_of_documents_reached) {
                        std.debug.print("[Coordinator] End of documents already reached - this is likely why no progress is being made\n", .{});
                    }

                    try self.transitionStateIfNeeded();
                    last_progress_time = std.time.nanoTimestamp(); // Reset timer
                }

                // Check if we've been running too long
                if (elapsed_ms > max_runtime_ms) {
                    std.debug.print("[Coordinator] Maximum runtime exceeded ({d} ms), forcing completion\n", .{elapsed_ms});
                    self.state = .Complete;
                    self.running = false;
                    break;
                }

                // Sleep a bit to avoid spinning
                std.time.sleep(1 * std.time.ns_per_ms);
            }

            // Shut down workers
            for (0..self.num_workers) |i| {
                if (self.input_queues[i]) |queue| {
                    const msg = message.createShutdownMessage(i);
                    _ = queue.push(msg);
                }
            }

            if (self.debug) {
                std.debug.print("[Coordinator] stopped running\n", .{});
            }
        }
    };
}
