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

/// Parallel string frequency analysis framework
pub fn ParallelAnalyzer(
    comptime cms_width: usize,
    comptime cms_depth: usize,
    comptime MY_LEN: comptime_int,
    comptime top_k: usize,
) type {
    // Define types
    const SFMType = SFM(cms_width, cms_depth, MY_LEN, top_k);
    const Coordinator = coordinator_mod.Coordinator(cms_width, cms_depth, MY_LEN, top_k);

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
            if (MY_LEN < 4) {
                return;
            }
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

            // After second pass, merge results from all workers into worker 0
            if (self.debug) {
                std.debug.print("[ParallelAnalyzer] Merging results from all workers\n", .{});
            }

            // Merge all worker results into worker 0's SFM
            for (1..self.coordinator.workers.len) |i| {
                try self.coordinator.workers[0].sfm.mergeCounts(self.coordinator.workers[i].sfm);
                if (self.debug) {
                    std.debug.print("[ParallelAnalyzer] Merged results from worker {d}\n", .{i});
                }
            }

            const total_time = time.nanoTimestamp() - overall_start_time;
            if (self.debug) {
                std.debug.print("[ParallelAnalyzer] Second pass completed in {d:.2}ms\n", .{@as(f64, @floatFromInt(total_time)) / time.ns_per_ms});
            }
        }

        pub fn addSmallStringsToHeap(self: *Self) !void {
            const manager = if (self.manager) |m| m else self.coordinator.workers[0].sfm;
            try manager.addSmallStringsToHeap();
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

        /// Save token set to binary format
        pub fn saveTokensToBinaryFormat(self: *Self, output_path: []const u8) !void {
            const start_time = time.nanoTimestamp();

            if (self.debug) {
                std.debug.print("[ParallelAnalyzer] Saving token set to binary format: {s}\n", .{output_path});
            }

            const manager = if (self.manager) |m| m else self.coordinator.workers[0].sfm;

            // Call the new method that saves tokens with non-overlapping counts
            try manager.saveTokensToBinaryFormat(output_path);

            const elapsed = time.nanoTimestamp() - start_time;
            if (self.debug) {
                std.debug.print("[ParallelAnalyzer] Tokens with non-overlapping counts saved in {d:.2}ms\n", .{@as(f64, @floatFromInt(elapsed)) / time.ns_per_ms});
            }
        }
    };
}
