const std = @import("std");
const Allocator = std.mem.Allocator;
const fs = std.fs;
const time = std.time;

const fineweb = @import("data_loader.zig").FinewebDataLoader;
const SFM = @import("string_frequency_manager.zig").StringFrequencyManager;
const CMS_F = @import("count_min_sketch.zig").CountMinSketch;
const coordinator_mod = @import("coordinator.zig");
const spsc = @import("spsc.zig");
const CandidateString = @import("string_frequency_manager.zig").CandidateString;
const MY_LEN = @import("count_min_sketch.zig").MY_LEN;

/// Parallel string frequency analysis framework
pub fn ParallelAnalyzer(
    comptime cms_width: usize,
    comptime cms_depth: usize,
    comptime min_length: usize,
    comptime max_length: usize,
    comptime top_k: usize,
) type {
    // Define types
    const SFMType = SFM(cms_width, cms_depth, min_length, max_length, top_k);
    const Coordinator = coordinator_mod.Coordinator(cms_width, cms_depth, min_length, max_length, top_k);

    return struct {
        const Self = @This();

        allocator: Allocator,
        coordinator: *Coordinator,
        manager: ?*SFMType,
        data_loader: *fineweb,
        debug: bool,
        num_threads: usize,
        saved_data_path: []const u8,

        pub fn init(
            allocator: Allocator,
            num_threads: usize,
            data_files: []const []const u8,
            vocab_file: []const u8,
            saved_data_path: []const u8,
            debug: bool,
        ) !*Self {
            const start_time = time.nanoTimestamp();

            // Validate inputs
            if (data_files.len == 0) {
                std.debug.print("[ERROR] No files provided to ParallelAnalyzer.init\n", .{});
                return error.NoFilesProvided;
            }

            // Create the analyzer
            const self = try allocator.create(Self);
            errdefer allocator.destroy(self);

            // Initialize data loader with the files
            var loader = try fineweb.init(allocator, data_files);
            errdefer loader.deinit();
            try loader.loadVocabulary(vocab_file);

            // Initialize coordinator
            var coordinator = try Coordinator.init(allocator, num_threads, loader, debug);
            errdefer coordinator.deinit();

            // Save the path
            const saved_path = try allocator.dupe(u8, saved_data_path);
            errdefer allocator.free(saved_path);

            // Initialize the analyzer
            self.* = .{
                .allocator = allocator,
                .coordinator = coordinator,
                .manager = null,
                .data_loader = loader,
                .debug = debug,
                .num_threads = num_threads,
                .saved_data_path = saved_path,
            };

            if (debug) {
                const elapsed = time.nanoTimestamp() - start_time;
                std.debug.print("[ParallelAnalyzer] init: {d:.2}ms\n", .{@as(f64, @floatFromInt(elapsed)) / time.ns_per_ms});

                // Print file information
                const status = loader.getFileStatus();
                std.debug.print("[ParallelAnalyzer] Using {d} files for processing\n", .{status.total_files});

                // Log the first few files
                const max_to_show = @min(5, data_files.len);
                for (data_files[0..max_to_show], 0..) |file, i| {
                    std.debug.print("[ParallelAnalyzer] File {d}: {s}\n", .{ i + 1, file });
                }
                if (data_files.len > max_to_show) {
                    std.debug.print("[ParallelAnalyzer] ... and {d} more files\n", .{data_files.len - max_to_show});
                }
            }

            return self;
        }

        /// Clean up resources
        pub fn deinit(self: *Self) void {
            const start_time = time.nanoTimestamp();

            // Free the saved_data_path first
            self.allocator.free(self.saved_data_path);

            // Before deinitiing the coordinator, clean up any manager
            if (self.manager) |manager| {
                manager.deinit();
                self.manager = null;
            }

            // The data_loader is owned by the coordinator, so don't deinit it separately

            // Deinit coordinator (which will clean up its workers)
            self.coordinator.deinit();
            self.coordinator = undefined; // Don't access it after deinit

            // Free self
            self.allocator.destroy(self);

            if (self.debug) {
                const elapsed = time.nanoTimestamp() - start_time;
                std.debug.print("[ParallelAnalyzer] deinit: {d:.2}ms\n", .{@as(f64, @floatFromInt(elapsed)) / time.ns_per_ms});
            }
        }

        /// Check if saved first pass data exists
        pub fn hasSavedData(self: *Self) !bool {
            const file = fs.cwd().openFile(self.saved_data_path, .{}) catch |err| {
                if (err == error.FileNotFound) {
                    return false;
                }
                return err;
            };
            file.close();
            return true;
        }

        /// Run first pass using parallel processing
        pub fn runFirstPass(self: *Self) !void {
            if (self.debug) {
                std.debug.print("[ParallelAnalyzer] Starting first pass with {d} threads\n", .{self.num_threads});

                // Print data source information
                const status = self.data_loader.getFileStatus();
                std.debug.print("[ParallelAnalyzer] Processing data from {d} files\n", .{status.total_files});
            }

            // Start the coordinator
            try self.coordinator.start();

            // Wait for completion
            self.coordinator.wait();

            if (self.debug) {
                std.debug.print("[ParallelAnalyzer] First pass completed\n", .{});
            }
        }

        /// Prepare data for second pass directly in memory (skipping disk I/O)
        pub fn prepareSecondPassInMemory(self: *Self) !void {
            if (self.debug) {
                std.debug.print("[ParallelAnalyzer] Preparing data for second pass in memory\n", .{});
            }

            if (MY_LEN < 4) {
                const start_time = time.nanoTimestamp();
                var new_manager = try SFMType.init(self.allocator);
                // Sum counters from all workers
                for (self.coordinator.workers) |worker| {
                    // Add each worker's length2 counters
                    if (MY_LEN == 2) {
                        for (0..new_manager.length2_counters.len) |i| {
                            new_manager.length2_counters[i] += worker.sfm.length2_counters[i];
                        }
                    }

                    // Add each worker's length3 counters
                    if (MY_LEN == 3) {
                        for (0..new_manager.length3_counters.len) |i| {
                            new_manager.length3_counters[i] += worker.sfm.length3_counters[i];
                        }
                    }

                    if (self.debug) {
                        std.debug.print("[ParallelAnalyzer] Merged counters from worker {d}\n", .{worker.id});
                    }
                }
                if (self.manager) |old_manager| {
                    old_manager.deinit();
                }
                self.manager = new_manager;
                const elapsed = time.nanoTimestamp() - start_time;
                if (self.debug) {
                    std.debug.print("[ParallelAnalyzer] Data prepared for second pass in {d:.2}ms\n", .{@as(f64, @floatFromInt(elapsed)) / time.ns_per_ms});
                }
            }
        }

        /// Save first pass data to disk
        pub fn saveFirstPassData(self: *Self) !void {
            if (self.debug) {
                std.debug.print("[ParallelAnalyzer] Saving first pass data to {s}\n", .{self.saved_data_path});
            }

            // Get access to the global CMS from coordinator
            if (self.coordinator.global_cms) |global_cms| {
                // Create a temporary SFM to save the data
                var temp_sfm = try SFMType.init(self.allocator);
                defer temp_sfm.deinit();

                // Copy the global CMS data to the SFM
                try temp_sfm.cms.merge(global_cms);

                // Initialize temp counter arrays to zero
                @memset(temp_sfm.length2_counters, 0);
                @memset(temp_sfm.length3_counters, 0);

                // Sum counters from all workers instead of just copying from worker 0
                for (self.coordinator.workers) |worker| {
                    // Add each worker's length2 counters
                    for (0..temp_sfm.length2_counters.len) |i| {
                        temp_sfm.length2_counters[i] += worker.sfm.length2_counters[i];
                    }

                    // Add each worker's length3 counters
                    for (0..temp_sfm.length3_counters.len) |i| {
                        temp_sfm.length3_counters[i] += worker.sfm.length3_counters[i];
                    }

                    if (self.debug) {
                        std.debug.print("[ParallelAnalyzer] Merged counters from worker {d}\n", .{worker.id});
                    }
                }

                // Save to disk
                try temp_sfm.saveFirstPassToDisk(self.saved_data_path);

                if (self.debug) {
                    std.debug.print("[ParallelAnalyzer] First pass data saved successfully\n", .{});
                }
            } else {
                std.debug.print("[ParallelAnalyzer] ERROR: No global CMS available for saving\n", .{});
                return error.NoGlobalCMS;
            }
        }

        /// Load first pass data from disk
        pub fn loadFirstPassData(self: *Self) !void {
            if (self.debug) {
                std.debug.print("[ParallelAnalyzer] Loading first pass data from {s}\n", .{self.saved_data_path});
            }

            // Load the data from disk
            const manager = try SFMType.loadFirstPassFromDisk(self.allocator, self.saved_data_path);
            self.manager = manager;

            if (self.debug) {
                std.debug.print("[ParallelAnalyzer] First pass data loaded\n", .{});
            }
        }

        /// Run second pass using parallel processing
        pub fn runSecondPass(self: *Self) !void {
            const overall_start_time = time.nanoTimestamp();

            if (self.debug) {
                std.debug.print("[ParallelAnalyzer] Starting second pass with {d} threads\n", .{self.num_threads});
            }

            // OPTIMIZATION: Reuse the existing data loader instead of creating a new one
            const data_loader_start = time.nanoTimestamp();

            // Rewind the data loader to start from the beginning
            try self.data_loader.rewind();

            const data_loader_time = time.nanoTimestamp() - data_loader_start;

            if (self.debug) {
                std.debug.print("[ParallelAnalyzer] Data loader rewound in {d:.2}ms\n", .{@as(f64, @floatFromInt(data_loader_time)) / time.ns_per_ms});
            }

            // Share the global CMS with all workers

            if (self.debug) {
                std.debug.print("[ParallelAnalyzer] Setting up workers to share global CMS\n", .{});
            }

            try self.coordinator.runSecondPass();

            const total_time = time.nanoTimestamp() - overall_start_time;
            if (self.debug) {
                std.debug.print("[ParallelAnalyzer] Second pass completed in {d:.2}ms\n", .{@as(f64, @floatFromInt(total_time)) / time.ns_per_ms});
            }
        }

        // This function runs the coordinator for the second pass
        fn runSecondPassCoordinator(params: anytype) !void {
            const start_time = std.time.nanoTimestamp();

            // Create worker threads
            const workers = try params.allocator.alloc(SecondPassWorkerThread, params.numThreads);
            defer params.allocator.free(workers);

            const init_start = std.time.nanoTimestamp();
            if (params.debug) {
                std.debug.print("[SecondPassCoordinator] Creating {d} worker threads\n", .{params.numThreads});
            }

            // Initialize workers
            for (workers, 0..) |*worker, i| {
                // Create a new manager for this worker but share the CMS from the main manager
                const manager_start = std.time.nanoTimestamp();

                // Create a new manager with empty CMS
                worker.manager = try SFMType.init(params.allocator, params.topK);

                // First, deinit the worker's new empty CMS
                worker.manager.cms.deinit();

                // Then point to the main CMS instead of copying it
                worker.manager.cms = params.mainManager.cms;

                // Mark that this worker doesn't own the CMS (to avoid double-free)
                worker.manager.cms_is_owned = false;

                const manager_time = std.time.nanoTimestamp() - manager_start;

                // Initialize SPSC queues - using larger capacity for better performance
                worker.input_queue = try params.allocator.create(spsc.BoundedQueue([]const u8, 32));
                worker.input_queue.* = try spsc.BoundedQueue([]const u8, 32).init(params.allocator);

                worker.completion_queue = try params.allocator.create(spsc.BoundedQueue(usize, 32));
                worker.completion_queue.* = try spsc.BoundedQueue(usize, 32).init(params.allocator);

                worker.id = i;
                worker.running = true;
                worker.doc_count = 0;
                worker.allocator = params.allocator;
                worker.debug = params.debug;

                // Start the worker thread
                worker.thread = try std.Thread.spawn(.{}, runSecondPassWorker, .{worker});

                if (params.debug) {
                    std.debug.print("[SecondPassCoordinator] Started worker {d} (CMS shared in {d:.2}ms)\n", .{ i, @as(f64, @floatFromInt(manager_time)) / time.ns_per_ms });
                }
            }

            const init_time = std.time.nanoTimestamp() - init_start;
            if (params.debug) {
                std.debug.print("[SecondPassCoordinator] All workers initialized in {d:.2}ms\n", .{@as(f64, @floatFromInt(init_time)) / time.ns_per_ms});
            }

            // Process documents
            var total_docs: usize = 0;
            var last_log_time = start_time;
            var doc_count_at_last_log: usize = 0;
            var docs_in_flight: usize = 0;

            // Pre-fill worker queues first
            const prefill_start = std.time.nanoTimestamp();
            for (workers) |*worker| {
                // Fill each worker's queue initially
                var docs_queued: usize = 0;
                while (docs_queued < 20 and worker.running) { // Fill with 20 docs initially
                    const doc_opt = try params.dataLoader.nextDocumentString();
                    if (doc_opt == null) break;

                    const doc = doc_opt.?;
                    if (worker.input_queue.push(doc)) {
                        docs_queued += 1;
                        docs_in_flight += 1;
                    } else {
                        // Queue is full, stop filling this worker
                        break;
                    }
                }

                if (params.debug) {
                    std.debug.print("[SecondPassCoordinator] Pre-filled worker {d} with {d} documents\n", .{ worker.id, docs_queued });
                }
            }

            const prefill_time = std.time.nanoTimestamp() - prefill_start;
            if (params.debug) {
                std.debug.print("[SecondPassCoordinator] Pre-filled all worker queues in {d:.2}ms\n", .{@as(f64, @floatFromInt(prefill_time)) / time.ns_per_ms});
            }

            // Main processing loop
            const process_start = std.time.nanoTimestamp();
            var end_of_documents = false;

            while (true) {
                // Check if all workers need more documents
                var all_workers_idle = true;
                var all_workers_done = true;
                var active_workers: usize = 0;

                for (workers) |*worker| {
                    if (worker.running) {
                        all_workers_done = false;
                        active_workers += 1;
                    }

                    // Check for completions first
                    var completions: usize = 0;
                    while (worker.completion_queue.pop()) |_| {
                        completions += 1;
                        total_docs += 1;
                        docs_in_flight -= 1;
                    }

                    if (completions > 0) {
                        worker.doc_count += completions;
                        all_workers_idle = false;
                    }

                    // If this worker needs more documents (queue has space)
                    if (!end_of_documents and worker.input_queue.count() < 10 and worker.running) {
                        all_workers_idle = false;

                        // Try to get a new document
                        const doc_opt = try params.dataLoader.nextDocumentString();
                        if (doc_opt == null) {
                            // No more documents, mark that we've reached the end
                            end_of_documents = true;

                            if (params.debug) {
                                std.debug.print("[SecondPassCoordinator] Reached end of documents after {d} documents\n", .{total_docs + docs_in_flight});
                            }
                        } else {
                            // Unwrap the optional to get the actual string slice
                            const doc = doc_opt.?;

                            // Add document to worker's queue (retry if full)
                            if (worker.input_queue.push(doc)) {
                                docs_in_flight += 1;
                            }
                        }
                    }
                }

                // Check for progress logging
                const current_time = time.nanoTimestamp();
                const elapsed_since_log = current_time - last_log_time;
                if (elapsed_since_log > 2 * std.time.ns_per_s) { // Log every 2 seconds
                    const docs_since_last_log = total_docs - doc_count_at_last_log;
                    const elapsed_sec = @as(f64, @floatFromInt(elapsed_since_log)) / std.time.ns_per_s;
                    const docs_per_sec = @as(f64, @floatFromInt(docs_since_last_log)) / elapsed_sec;

                    std.debug.print("[SecondPassCoordinator] Processed {d} documents ({d:.2} docs/sec), {d} in flight, {d} active workers\n", .{ total_docs, docs_per_sec, docs_in_flight, active_workers });

                    // Show per-worker stats
                    for (workers, 0..) |worker, i| {
                        if (worker.running) {
                            std.debug.print("[SecondPassCoordinator] Worker {d}: {d} documents processed, queue depth: {d}\n", .{ i, worker.doc_count, worker.input_queue.count() });
                        }
                    }

                    // Show data loader file status
                    const status = params.dataLoader.getFileStatus();
                    if (!status.reached_end) {
                        std.debug.print("[SecondPassCoordinator] Processing file {d} of {d}: {s}\n", .{ status.current_file_index + 1, status.total_files, status.current_file_path });
                    }

                    last_log_time = current_time;
                    doc_count_at_last_log = total_docs;
                }

                // If we've reached the end of documents and all workers are done processing
                if (end_of_documents and docs_in_flight == 0) {
                    // Send shutdown signals to all workers
                    for (workers) |*worker| {
                        if (worker.running) {
                            // Create an empty string as a shutdown signal
                            const empty_string = try params.allocator.dupe(u8, "");

                            // Keep trying until the queue has space
                            while (!worker.input_queue.push(empty_string)) {
                                std.time.sleep(1 * std.time.ns_per_ms);
                            }

                            if (params.debug) {
                                std.debug.print("[SecondPassCoordinator] Signaled worker {d} to stop\n", .{worker.id});
                            }
                        }
                    }

                    if (params.debug) {
                        std.debug.print("[SecondPassCoordinator] All documents processed, waiting for workers to complete\n", .{});
                    }
                    break;
                }

                // If all workers are done, break out
                if (all_workers_done) {
                    if (params.debug) {
                        std.debug.print("[SecondPassCoordinator] All workers completed, waiting for threads to join\n", .{});
                    }
                    break;
                }

                // Sleep a bit to avoid spinning
                std.time.sleep(1 * std.time.ns_per_ms);
            }

            const process_time = time.nanoTimestamp() - process_start;
            if (params.debug) {
                std.debug.print("[SecondPassCoordinator] Processing phase completed in {d:.2}ms\n", .{@as(f64, @floatFromInt(process_time)) / time.ns_per_ms});
            }

            // Wait for all worker threads to finish
            const merge_start = time.nanoTimestamp();
            for (workers) |*worker| {
                worker.thread.join();

                if (params.debug) {
                    std.debug.print("[SecondPassCoordinator] Worker {d} processed {d} documents\n", .{ worker.id, worker.doc_count });
                }
            }

            // Merge results from all worker managers into the main manager
            if (params.debug) {
                std.debug.print("[SecondPassCoordinator] All workers finished, merging results...\n", .{});
            }

            for (workers, 0..) |*worker, i| {
                const worker_merge_start = time.nanoTimestamp();

                // Merge the actual counts into main manager
                var worker_heap = &worker.manager.heap;
                var worker_counts = &worker.manager.actual_counts;
                var main_counts = &params.mainManager.actual_counts;

                // Process the heap items (top-K candidates)
                while (worker_heap.count() > 0) {
                    const item = worker_heap.remove();
                    const count = worker_counts.get(item.string) orelse 0;

                    // Add to main manager if not already there, or increase count if it is
                    if (main_counts.contains(item.string)) {
                        // Already exists, just add the count
                        main_counts.getPtr(item.string).?.* += count;
                    } else {
                        // New string, create a copy and add it
                        const str_copy = try params.allocator.dupe(u8, item.string);
                        try main_counts.put(str_copy, count);

                        // Add to main manager's heap if needed
                        var main_heap = &params.mainManager.heap;
                        if (main_heap.count() < params.topK) {
                            try main_heap.add(CandidateString.init(str_copy, item.guess_count));
                        } else if (main_heap.peek().?.guess_count < item.guess_count) {
                            const evicted = main_heap.remove();
                            params.allocator.free(evicted.string);
                            try main_heap.add(CandidateString.init(str_copy, item.guess_count));
                        }
                    }

                    // Free the worker's string
                    params.allocator.free(item.string);
                }

                // Clean up the worker's manager and queues
                // OPTIMIZATION: Don't deinit the CMS itself since it's shared and not owned by worker
                worker.manager.cms_is_owned = false;
                worker.manager.deinit();
                worker.input_queue.deinit(params.allocator);
                params.allocator.destroy(worker.input_queue);
                worker.completion_queue.deinit(params.allocator);
                params.allocator.destroy(worker.completion_queue);

                const worker_merge_time = time.nanoTimestamp() - worker_merge_start;
                if (params.debug) {
                    std.debug.print("[SecondPassCoordinator] Merged results from worker {d} in {d:.2}ms\n", .{ i, @as(f64, @floatFromInt(worker_merge_time)) / time.ns_per_ms });
                }
            }

            const merge_time = time.nanoTimestamp() - merge_start;
            if (params.debug) {
                std.debug.print("[SecondPassCoordinator] Results merge phase completed in {d:.2}ms\n", .{@as(f64, @floatFromInt(merge_time)) / time.ns_per_ms});
            }

            // Signal completion
            params.completionFlag.* = true;

            const elapsed_ms = @as(f64, @floatFromInt(std.time.nanoTimestamp() - start_time)) / std.time.ns_per_ms;
            if (params.debug) {
                std.debug.print("[SecondPassCoordinator] Second pass completed in {d:.2}ms, processed {d} documents ({d:.2} docs/sec)\n", .{ elapsed_ms, total_docs, @as(f64, @floatFromInt(total_docs)) / (elapsed_ms / 1000.0) });
            }
        }

        // Define the worker thread type for the second pass
        const SecondPassWorkerThread = struct {
            id: usize,
            thread: std.Thread,
            input_queue: *spsc.BoundedQueue([]const u8, 32),
            completion_queue: *spsc.BoundedQueue(usize, 32),
            manager: *SFMType,
            allocator: Allocator,
            running: bool,
            doc_count: usize,
            debug: bool,
        };

        // This function runs a worker for the second pass
        fn runSecondPassWorker(worker: *SecondPassWorkerThread) !void {
            const start_time = time.nanoTimestamp();

            if (worker.debug) {
                std.debug.print("[SecondPassWorker {d}] Started\n", .{worker.id});
            }

            var doc_count: usize = 0;
            var last_log_time = start_time;
            var last_log_count: usize = 0;

            while (worker.running) {
                // Get a document from the input queue
                const doc_opt = worker.input_queue.pop();
                if (doc_opt == null) {
                    // Empty queue, wait a bit
                    std.time.sleep(1 * std.time.ns_per_ms);
                    continue;
                }

                // Process the document
                const doc = doc_opt.?;

                if (doc.len == 0) {
                    // Empty string signals shutdown
                    if (worker.debug) {
                        std.debug.print("[SecondPassWorker {d}] Received shutdown signal\n", .{worker.id});
                    }
                    worker.running = false;
                    worker.allocator.free(doc); // Free the shutdown signal string
                    break;
                }

                const process_start = time.nanoTimestamp();
                try worker.manager.processDocumentSecondPass(doc);
                const process_time = time.nanoTimestamp() - process_start;

                worker.allocator.free(doc);

                // Signal completion
                _ = worker.completion_queue.push(1);

                doc_count += 1;

                if (doc_count % 5000 == 0 or doc_count == 1) {
                    const current_time = time.nanoTimestamp();
                    const elapsed = @as(f64, @floatFromInt(current_time - start_time)) / time.ns_per_ms;
                    const docs_per_sec = @as(f64, @floatFromInt(doc_count)) / (elapsed / 1000.0);

                    if (worker.debug) {
                        std.debug.print("[SecondPassWorker {d}] Processed {d} documents ({d:.2} docs/sec, last doc: {d:.2}ms)\n", .{ worker.id, doc_count, docs_per_sec, @as(f64, @floatFromInt(process_time)) / time.ns_per_ms });
                    }

                    last_log_time = current_time;
                    last_log_count = doc_count;
                }
            }

            const elapsed_ms = @as(f64, @floatFromInt(time.nanoTimestamp() - start_time)) / time.ns_per_ms;
            if (worker.debug) {
                std.debug.print("[SecondPassWorker {d}] Finished, processed {d} documents in {d:.2}ms ({d:.2} docs/sec)\n", .{ worker.id, doc_count, elapsed_ms, @as(f64, @floatFromInt(doc_count)) / (elapsed_ms / 1000.0) });
            }
        }

        pub fn getResults(self: *Self) !void {
            if (self.debug) {
                std.debug.print("[ParallelAnalyzer] Getting results\n", .{});
            }

            if (self.manager) |manager| {
                try manager.getResults();
            } else {
                try self.coordinator.workers[0].sfm.getResults();
            }
            if (self.debug) {
                std.debug.print("[ParallelAnalyzer] Results displayed\n", .{});
            }
        }
    };
}
