const std = @import("std");
const Allocator = std.mem.Allocator;
const Thread = std.Thread;
const message = @import("message.zig");
const message_queue = @import("message_queue.zig");
const time = std.time;

// Import Count-Min Sketch
const CMS_F = @import("count_min_sketch.zig").CountMinSketch;
const SFM = @import("string_frequency_manager.zig").StringFrequencyManager;

/// Worker thread that processes documents
pub fn Worker(
    comptime cms_width: usize,
    comptime cms_depth: usize,
    comptime MY_LEN: comptime_int,
    comptime top_k: usize,
    comptime debug: bool,
) type {
    // Get the CMS type
    const CMS = CMS_F(cms_width, cms_depth, MY_LEN);
    const SFMType = SFM(cms_width, cms_depth, MY_LEN, top_k);

    return struct {
        const Self = @This();

        /// Worker ID
        id: usize,

        /// Allocator
        allocator: Allocator,

        /// String Frequency Manager for processing documents
        sfm: *SFMType,

        /// Thread handle
        thread: ?Thread = null,

        /// Input queue for receiving messages from coordinator
        input_queue: *message_queue.CoordinatorMessageQueue,

        /// Output queue for sending messages to coordinator
        output_queue: *message_queue.WorkerMessageQueue,

        /// Worker state
        running: bool = false,

        // is my sfm ready to be used?
        sfm_ready: bool = false,

        /// Initialize a new worker
        pub fn init(
            allocator: Allocator,
            id: usize,
            input_queue: *message_queue.CoordinatorMessageQueue, // Keep as pointer
            output_queue: *message_queue.WorkerMessageQueue, // Keep as pointer
        ) !*Self {
            const start_time = time.nanoTimestamp();

            // Create the worker
            const self = try allocator.create(Self);
            errdefer allocator.destroy(self);

            // Initialize the worker with pointers to the queues
            self.* = .{
                .id = id,
                .allocator = allocator,
                .sfm = undefined,
                .input_queue = input_queue, // Store the pointer
                .output_queue = output_queue, // Store the pointer
            };

            if (debug) {
                const elapsed = time.nanoTimestamp() - start_time;
                std.debug.print("[Worker {d}] init: {d:.2}ms\n", .{ id, @as(f64, @floatFromInt(elapsed)) / time.ns_per_ms });
            }

            return self;
        }

        /// Clean up resources
        pub fn deinit(self: *Self) void {
            const start_time = time.nanoTimestamp();

            if (self.thread) |thread| {
                thread.join();
                self.thread = null;
            }

            // Free the SFM (which will also free its CMS)
            self.sfm.deinit();

            // Free the worker itself
            self.allocator.destroy(self);

            if (debug) {
                const elapsed = time.nanoTimestamp() - start_time;
                std.debug.print("[Worker {d}] deinit: {d:.2}ms\n", .{ self.id, @as(f64, @floatFromInt(elapsed)) / time.ns_per_ms });
            }
        }

        /// Start the worker thread
        pub fn start(self: *Self) !void {
            if (self.thread != null) return;

            self.running = true;
            self.thread = try Thread.spawn(.{}, Self.run, .{self});

            if (debug) {
                std.debug.print("[Worker {d}] started\n", .{self.id});
            }
        }

        /// Stop the worker thread
        pub fn stop(self: *Self) void {
            self.running = false;

            if (debug) {
                std.debug.print("[Worker {d}] stopping\n", .{self.id});
            }
        }

        /// Process a document (first pass)
        fn processDocumentFirstPass(self: *Self, document: []const u8) !void {
            const start_time = time.nanoTimestamp();

            // Build CMS for this document
            try self.sfm.buildCMS(document);

            if (debug) {
                const elapsed = time.nanoTimestamp() - start_time;
                std.debug.print("[Worker {d}] processDocumentFirstPass ({d} bytes): {d:.2}ms\n", .{ self.id, document.len, @as(f64, @floatFromInt(elapsed)) / time.ns_per_ms });
            }
        }

        /// Process a document (second pass)
        fn processDocumentSecondPass(self: *Self, document: []const u8) !void {
            const start_time = time.nanoTimestamp();

            // Process document with shared CMS
            try self.sfm.processDocumentSecondPass(document);

            if (debug) {
                const elapsed = time.nanoTimestamp() - start_time;
                std.debug.print("[Worker {d}] processDocumentSecondPass ({d} bytes): {d:.2}ms\n", .{ self.id, document.len, @as(f64, @floatFromInt(elapsed)) / time.ns_per_ms });
            }
        }

        /// Dump state to a file
        fn dumpState(self: *Self, path: []const u8) !void {
            const start_time = time.nanoTimestamp();

            // Create the path by appending worker ID
            var buf: [1024]u8 = undefined;
            const worker_path = try std.fmt.bufPrint(&buf, "{s}.{d}", .{ path, self.id });

            // Save first pass data to disk
            try self.sfm.saveFirstPassToDisk(worker_path);

            if (debug) {
                const elapsed = time.nanoTimestamp() - start_time;
                std.debug.print("[Worker {d}] dumpState to {s}: {d:.2}ms\n", .{ self.id, worker_path, @as(f64, @floatFromInt(elapsed)) / time.ns_per_ms });
            }
        }

        /// Handle a coordinator message
        fn handleCoordinatorMessage(self: *Self, msg: message.CoordinatorMessage) !void {
            const start_time = time.nanoTimestamp();

            if (debug) {
                std.debug.print("[Worker {d}] [DEBUG] Received {s} message from coordinator\n", .{ self.id, @tagName(msg) });
            }

            switch (msg) {
                .ProcessDocument => |process_data| {
                    const document = process_data.document;
                    const pass = process_data.pass;

                    if (pass == 1) {
                        try self.processDocumentFirstPass(document);
                    } else if (pass == 2) {
                        try self.processDocumentSecondPass(document);
                    } else {
                        if (debug) {
                            std.debug.print("[Worker {d}] Warning: Received ProcessDocument message with invalid pass {d}\n", .{ self.id, pass });
                        }
                        return;
                    }

                    // Send the document processed message
                    const response = message.createDocumentProcessedMessage(self.id, process_data.document_id, document, pass);
                    _ = self.output_queue.push(response);
                },
                // .FindTopK => {
                //     // TODO: Implement finding top K strings
                //     // For now, just send a response
                //     const response = message.createTopKCompleteMessage(self.id);
                //     _ = self.output_queue.push(response);
                // },
                .MergeCMS => |cms_data| {
                    const other_cms = @as(*CMS, @ptrCast(@alignCast(cms_data.cms)));
                    if (MY_LEN >= 4) {
                        try self.sfm.cms.merge(other_cms);
                    }

                    // Create a response that includes a pointer to our local CMS
                    const response = message.createMergedCMSMessage(self.id);
                    const push_result = self.output_queue.push(response);

                    if (debug) {
                        if (push_result) {
                            std.debug.print("[Worker {d}] Sent completion after merging CMS data to coordinator\n", .{self.id});
                        } else {
                            std.debug.print("[Worker {d}] [ERROR] Failed to send CMS data to coordinator! Queue full?\n", .{self.id});
                        }
                    }
                },
                .CopyCMS => |cms_data| {
                    const other_cms = @as(*CMS, @ptrCast(@alignCast(cms_data.cms)));
                    try self.sfm.cms.copyFrom(other_cms);

                    // Create a response that includes a pointer to our local CMS
                    const response = message.createCopiedCMSMessage(self.id);
                    const push_result = self.output_queue.push(response);

                    if (debug) {
                        if (push_result) {
                            std.debug.print("[Worker {d}] Sent completion after copying CMS data to coordinator\n", .{self.id});
                        } else {
                            std.debug.print("[Worker {d}] [ERROR] Failed to send CMS data to coordinator! Queue full?\n", .{self.id});
                        }
                    }
                },
                .MergeCounts => |sfm_data| {
                    const other_sfm = @as(*SFMType, @ptrCast(@alignCast(sfm_data.sfm)));
                    try self.sfm.mergeCounts(other_sfm);

                    // Create a response that includes a pointer to our local CMS
                    const response = message.createMergedCountsMessage(self.id);
                    const push_result = self.output_queue.push(response);

                    if (debug) {
                        if (push_result) {
                            std.debug.print("[Worker {d}] Sent completion after merging Counts data to coordinator\n", .{self.id});
                        } else {
                            std.debug.print("[Worker {d}] [ERROR] Failed to send Counts data to coordinator! Queue full?\n", .{self.id});
                        }
                    }
                },
                .DumpState => |dump_data| {
                    try self.dumpState(dump_data.dump_path);

                    // Send the state dumped message
                    const response = message.createStateDumpedMessage(self.id);
                    _ = self.output_queue.push(response);
                },
                .Shutdown => {
                    if (debug) {
                        std.debug.print("[Worker {d}] Received shutdown message\n", .{self.id});
                    }
                    self.running = false;

                    // No response needed
                },
            }

            const elapsed = time.nanoTimestamp() - start_time;
            if (debug and elapsed > 10 * time.ns_per_ms) { // Log if handling took more than 10ms
                std.debug.print("[Worker {d}] [DEBUG] handleCoordinatorMessage for {s} took {d:.2}ms\n", .{ self.id, @tagName(msg), @as(f64, @floatFromInt(elapsed)) / time.ns_per_ms });
            }
        }

        /// Main worker loop
        fn run(self: *Self) !void {
            if (!self.sfm_ready) {
                // Create the SFM (which will create its own CMS)
                self.sfm = try SFMType.init(self.allocator);
                self.sfm_ready = true;
            }

            if (debug) {
                std.debug.print("[Worker {d}] started running\n", .{self.id});
            }

            while (self.running) {
                // Check for messages from the coordinator
                if (self.input_queue.pop()) |msg| {
                    // Handle the message
                    self.handleCoordinatorMessage(msg) catch |err| {
                        if (debug) {
                            std.debug.print("[Worker {d}] Error handling message: {any}\n", .{ self.id, err });
                        }

                        // Create an error message
                        var buf: [256]u8 = undefined;
                        const error_msg = try std.fmt.bufPrintZ(&buf, "Error handling message: {any}", .{err});
                        const error_response = message.createErrorMessage(self.id, error_msg);
                        _ = self.output_queue.push(error_response);
                    };
                } else {
                    // No messages, sleep for a bit
                    //std.time.sleep(1 * std.time.ns_per_ms); // 300ms
                }
            }

            if (debug) {
                std.debug.print("[Worker {d}] stopped running\n", .{self.id});
            }
        }
    };
}
