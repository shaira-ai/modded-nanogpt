const std = @import("std");
const Allocator = std.mem.Allocator;
const fs = std.fs;
const time = std.time;

const parallel = @import("parallel.zig");
const fineweb = @import("data_loader.zig");

const input_files_ = [_][]const u8{
    "fineweb_train_000001.bin",
    "fineweb_train_000002.bin",
    "fineweb_train_000003.bin",
    "fineweb_train_000004.bin",
    "fineweb_train_000005.bin",
    "fineweb_train_000006.bin",
    "fineweb_train_000007.bin",
    "fineweb_train_000008.bin",
    "fineweb_train_000009.bin",
    "fineweb_train_000010.bin",
    "fineweb_train_000011.bin",
    "fineweb_train_000012.bin",
    "fineweb_train_000013.bin",
    "fineweb_train_000014.bin",
    "fineweb_train_000015.bin",
    "fineweb_train_000016.bin",
    "fineweb_train_000017.bin",
    "fineweb_train_000018.bin",
    "fineweb_train_000019.bin",
    "fineweb_train_000020.bin",
    "fineweb_train_000021.bin",
    "fineweb_train_000022.bin",
    "fineweb_train_000023.bin",
    "fineweb_train_000024.bin",
    "fineweb_train_000025.bin",
    "fineweb_train_000026.bin",
    "fineweb_train_000027.bin",
    "fineweb_train_000028.bin",
    "fineweb_train_000029.bin",
    "fineweb_train_000030.bin",
    "fineweb_train_000031.bin",
    "fineweb_train_000032.bin",
    "fineweb_train_000033.bin",
    "fineweb_train_000034.bin",
    "fineweb_train_000035.bin",
    "fineweb_train_000036.bin",
    "fineweb_train_000037.bin",
    "fineweb_train_000038.bin",
    "fineweb_train_000039.bin",
    "fineweb_train_000040.bin",
    "fineweb_train_000041.bin",
    "fineweb_train_000042.bin",
    "fineweb_train_000043.bin",
    "fineweb_train_000044.bin",
    "fineweb_train_000045.bin",
    "fineweb_train_000046.bin",
    "fineweb_train_000047.bin",
    "fineweb_train_000048.bin",
    "fineweb_train_000049.bin",
    "fineweb_train_000050.bin",
    "fineweb_train_000051.bin",
    "fineweb_train_000052.bin",
    "fineweb_train_000053.bin",
    "fineweb_train_000054.bin",
    "fineweb_train_000055.bin",
    "fineweb_train_000056.bin",
    "fineweb_train_000057.bin",
    "fineweb_train_000058.bin",
    "fineweb_train_000059.bin",
    "fineweb_train_000060.bin",
    "fineweb_train_000061.bin",
    "fineweb_train_000062.bin",
    "fineweb_train_000063.bin",
    "fineweb_train_000064.bin",
    "fineweb_train_000065.bin",
    "fineweb_train_000066.bin",
    "fineweb_train_000067.bin",
    "fineweb_train_000068.bin",
    "fineweb_train_000069.bin",
    "fineweb_train_000070.bin",
    "fineweb_train_000071.bin",
    "fineweb_train_000072.bin",
    "fineweb_train_000073.bin",
    "fineweb_train_000074.bin",
    "fineweb_train_000075.bin",
    "fineweb_train_000076.bin",
    "fineweb_train_000077.bin",
    "fineweb_train_000078.bin",
    "fineweb_train_000079.bin",
    "fineweb_train_000080.bin",
    "fineweb_train_000081.bin",
    "fineweb_train_000082.bin",
    "fineweb_train_000083.bin",
    "fineweb_train_000084.bin",
    "fineweb_train_000085.bin",
    "fineweb_train_000086.bin",
    "fineweb_train_000087.bin",
    "fineweb_train_000088.bin",
    "fineweb_train_000089.bin",
    "fineweb_train_000090.bin",
    "fineweb_train_000091.bin",
    "fineweb_train_000092.bin",
    "fineweb_train_000093.bin",
    "fineweb_train_000094.bin",
    "fineweb_train_000095.bin",
    "fineweb_train_000096.bin",
    "fineweb_train_000097.bin",
    "fineweb_train_000098.bin",
    "fineweb_train_000099.bin",
    "fineweb_train_000100.bin",
    "fineweb_train_000101.bin",
    "fineweb_train_000102.bin",
    "fineweb_train_000103.bin",
};

// Function to build full paths with data directory
fn buildDataPaths(allocator: Allocator, data_dir: []const u8, n_files: usize) ![][]const u8 {
    const paths = try allocator.alloc([]const u8, n_files);
    for (0..n_files) |i| {
        paths[i] = try std.fs.path.join(allocator, &[_][]const u8{ data_dir, input_files_[i] });
    }
    return paths;
}

fn freeDataPaths(allocator: Allocator, paths: [][]const u8) void {
    for (paths) |path| {
        allocator.free(path);
    }
    allocator.free(paths);
}

pub fn main() !void {
    // Use this instead if you want fast compile times
    // try MainHaver(10).main(2, 3, "data");
    try anyMain();
}

pub fn anyMain() !void {
    var mains: [257]*const fn (usize, usize, []const u8) anyerror!void = undefined;
    inline for (2..257) |i| {
        mains[i] = MainHaver(i).main;
    }

    var args_iterator = std.process.args();
    _ = args_iterator.skip();
    const first_arg = args_iterator.next() orelse {
        std.debug.print("Error: Expected at least three arguments\n", .{});
        std.debug.print("Usage: program <which> <n_threads> <n_files> [data_dir]\n", .{});
        std.process.exit(1);
    };
    const which = try std.fmt.parseUnsigned(usize, first_arg, 10);
    const second_arg = args_iterator.next() orelse {
        std.debug.print("Error: Expected at least three arguments\n", .{});
        std.debug.print("Usage: program <which> <n_threads> <n_files> [data_dir]\n", .{});
        std.process.exit(1);
    };
    const n_threads = try std.fmt.parseUnsigned(usize, second_arg, 10);
    const third_arg = args_iterator.next() orelse {
        std.debug.print("Error: Expected at least three arguments\n", .{});
        std.debug.print("Usage: program <which> <n_threads> <n_files> [data_dir]\n", .{});
        std.process.exit(1);
    };
    const n_files = try std.fmt.parseUnsigned(usize, third_arg, 10);

    const data_dir = args_iterator.next() orelse "data";

    if (which >= 2 and which < 257) {
        try mains[which](n_files, n_threads, data_dir);
    }
}

pub fn MainHaver(MY_LEN: comptime_int) type {
    return struct {
        pub fn main(n_files: usize, num_threads: usize, data_dir: []const u8) anyerror!void {
            const allocator = std.heap.c_allocator;
            var timer = try std.time.Timer.start();

            // Build the input file paths
            const input_files = try buildDataPaths(allocator, data_dir, n_files);
            defer freeDataPaths(allocator, input_files);

            // Configure parameters
            const top_k = 1000000; // Track top 100000 strings per length
            const cms_width = 1 << 24; // ~16 million counters per hash function
            const cms_depth = 10;

            const available_cores = try std.Thread.getCpuCount();

            const debug = true;

            // Flag to skip disk I/O and keep data in memory between passes
            const skip_disk_io = true;

            // Build paths for vocab file and saved data in the data directory
            const vocab_file = try std.fs.path.join(allocator, &[_][]const u8{ data_dir, "vocab.json" });
            defer allocator.free(vocab_file);

            const saved_data_path = try std.fs.path.join(allocator, &[_][]const u8{ data_dir, "fineweb_first_pass_parallel.bin" });
            defer allocator.free(saved_data_path);

            std.debug.print("=== Parallel String Frequency Analysis ===\n", .{});
            std.debug.print("Parameters:\n", .{});
            std.debug.print("  - Data directory: {s}\n", .{data_dir});
            std.debug.print("  - CMS width: {d} counters\n", .{cms_width});
            std.debug.print("  - CMS depth: {d} hash functions\n", .{cms_depth});
            std.debug.print("  - length: {d}\n", .{MY_LEN});
            std.debug.print("  - Top K: {d} strings per length\n", .{top_k});
            std.debug.print("  - Worker threads: {d} (of {d} cores)\n", .{ num_threads, available_cores });
            std.debug.print("  - Skip disk I/O: {}\n", .{skip_disk_io});
            std.debug.print("  - Using {d} input files\n", .{n_files});

            // Print file list
            for (input_files, 0..) |file, i| {
                std.debug.print("    {d}: {s}\n", .{ i + 1, file });
            }

            // Create the parallel analyzer
            const ParallelAnalyzer = parallel.ParallelAnalyzer(cms_width, cms_depth, MY_LEN, top_k);

            // Create analyzer with specific files
            var analyzer = try ParallelAnalyzer.init(allocator, num_threads, input_files, vocab_file, saved_data_path, debug);
            defer analyzer.deinit();

            // Check if saved data exists
            const saved_data_exists = try analyzer.hasSavedData();

            if (saved_data_exists) {
                // Load the first pass data from disk
                std.debug.print("\n=== LOADING FIRST PASS DATA FROM DISK ===\n", .{});
                _ = timer.lap(); // Start loading timer

                try analyzer.loadFirstPassData();

                const load_time = timer.lap();
                const load_ms = @as(f64, @floatFromInt(load_time)) / time.ns_per_ms;
                std.debug.print("Data loaded in {d:.2}ms\n", .{load_ms});

                // Run second pass
                std.debug.print("\n=== PASS 2: Finding Top Strings (Parallel) ===\n", .{});
                _ = timer.lap();

                try analyzer.runSecondPass();

                const second_pass_time = timer.lap();
                const second_pass_ms = @as(f64, @floatFromInt(second_pass_time)) / time.ns_per_ms;
                std.debug.print("Pass 2 completed in {d:.2}ms\n", .{second_pass_ms});
            } else {
                // Run first pass
                std.debug.print("\n=== PASS 1: Building Count-Min Sketch (Parallel) ===\n", .{});
                _ = timer.lap();

                try analyzer.runFirstPass();

                const first_pass_time = timer.lap();
                const first_pass_ms = @as(f64, @floatFromInt(first_pass_time)) / time.ns_per_ms;
                std.debug.print("Pass 1 completed in {d:.2}ms\n", .{first_pass_ms});

                if (skip_disk_io) {
                    // Keep data in memory for second pass (skip disk I/O)
                    std.debug.print("\nPreparing data for second pass in memory...\n", .{});
                    _ = timer.lap();

                    try analyzer.prepareSecondPassInMemory();

                    const prep_time = timer.lap();
                    const prep_ms = @as(f64, @floatFromInt(prep_time)) / time.ns_per_ms;
                    std.debug.print("Data prepared in {d:.2}ms\n", .{prep_ms});
                } else {
                    // Save first pass data to disk
                    std.debug.print("\nSaving first pass data to disk...\n", .{});
                    _ = timer.lap();

                    try analyzer.saveFirstPassData();

                    const save_time = timer.lap();
                    const save_ms = @as(f64, @floatFromInt(save_time)) / time.ns_per_ms;
                    std.debug.print("Data saved in {d:.2}ms\n", .{save_ms});

                    // Load saved data for second pass
                    std.debug.print("\nLoading saved data for second pass...\n", .{});
                    _ = timer.lap();

                    try analyzer.loadFirstPassData();

                    const load_time = timer.lap();
                    const load_ms = @as(f64, @floatFromInt(load_time)) / time.ns_per_ms;
                    std.debug.print("Data loaded in {d:.2}ms\n", .{load_ms});
                }

                // Run second pass
                std.debug.print("\n=== PASS 2: Finding Top Strings (Parallel) ===\n", .{});
                _ = timer.lap();

                if (MY_LEN < 4) {
                    std.debug.print("\n=== ADDING SMALL STRINGS TO HEAP ===\n", .{});
                    try analyzer.addSmallStringsToHeap();
                    std.debug.print("Small strings added to heap\n", .{});

                    std.debug.print("\n=== SECOND PASS: Non-overlapping counting ===\n", .{});
                    _ = timer.lap();
                    try analyzer.runSecondPassSmallStrings();
                    const second_pass_time = timer.lap();
                    const second_pass_ms = @as(f64, @floatFromInt(second_pass_time)) / time.ns_per_ms;
                    std.debug.print("Non-overlapping counting completed in {d:.2}ms\n", .{second_pass_ms});
                } else {
                    try analyzer.runSecondPass();
                }

                const second_pass_time = timer.lap();
                const second_pass_ms = @as(f64, @floatFromInt(second_pass_time)) / time.ns_per_ms;
                std.debug.print("Pass 2 completed in {d:.2}ms\n", .{second_pass_ms});
            }

            // Get and display results
            std.debug.print("\n=== RESULTS ===\n", .{});
            try analyzer.getResults();

            // Save tokens to binary format
            std.debug.print("\n=== SAVING TOKEN SET TO BINARY FORMAT ===\n", .{});
            var buf: [1024]u8 = undefined;
            const token_filename = try std.fmt.bufPrint(&buf, "tokenset_{d}.bin", .{MY_LEN});
            const token_set_path = try std.fs.path.join(allocator, &[_][]const u8{ data_dir, token_filename });
            defer allocator.free(token_set_path);

            _ = timer.lap();
            try analyzer.saveTokensToBinaryFormat(token_set_path);
            const save_token_time = timer.lap();
            const save_token_ms = @as(f64, @floatFromInt(save_token_time)) / time.ns_per_ms;
            std.debug.print("Token set saved to {s} in {d:.2}ms\n", .{ token_set_path, save_token_ms });

            // Display overall statistics
            const total_time = timer.read();
            const total_ms = @as(f64, @floatFromInt(total_time)) / time.ns_per_ms;
            std.debug.print("\n=== Processing Complete ===\n", .{});
            std.debug.print("Total execution time: {d:.2}ms\n", .{total_ms});
        }
    };
}
