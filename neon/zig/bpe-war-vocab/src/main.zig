const std = @import("std");
const Allocator = std.mem.Allocator;
const fs = std.fs;
const time = std.time;

const parallel = @import("parallel.zig");
const fineweb = @import("data_loader.zig");

pub fn main() !void {
    const allocator = std.heap.c_allocator;
    var timer = try std.time.Timer.start();

    // Configure parameters
    const min_length = 2;
    const max_length = 256;
    const top_k = 10000; // Track top 10000 strings per length
    const cms_width = 1 << 24; // ~16 million counters per hash function
    const cms_depth = 10;

    const available_cores = try std.Thread.getCpuCount();
    const num_threads = 10;

    const debug = true;

    // Flag to skip disk I/O and keep data in memory between passes
    const skip_disk_io = true;

    const input_files = [_][]const u8{
        "fineweb_train_000001.bin",
        "fineweb_train_000002.bin",
    };

    const vocab_file = "vocab.json";
    const saved_data_path = "fineweb_first_pass_parallel.bin";

    std.debug.print("=== Parallel String Frequency Analysis ===\n", .{});
    std.debug.print("Parameters:\n", .{});
    std.debug.print("  - CMS width: {d} counters\n", .{cms_width});
    std.debug.print("  - CMS depth: {d} hash functions\n", .{cms_depth});
    std.debug.print("  - Min length: {d}\n", .{min_length});
    std.debug.print("  - Max length: {d}\n", .{max_length});
    std.debug.print("  - Top K: {d} strings per length\n", .{top_k});
    std.debug.print("  - Worker threads: {d} (of {d} cores)\n", .{ num_threads, available_cores });
    std.debug.print("  - Skip disk I/O: {}\n", .{skip_disk_io});
    std.debug.print("  - Using {d} input files\n", .{input_files.len});

    // Print file list
    for (input_files, 0..) |file, i| {
        std.debug.print("    {d}: {s}\n", .{ i + 1, file });
    }

    // Create the parallel analyzer
    const ParallelAnalyzer = parallel.ParallelAnalyzer(cms_width, cms_depth, min_length, max_length);

    // Create analyzer with specific files
    var analyzer = try ParallelAnalyzer.init(allocator, num_threads, &input_files, vocab_file, saved_data_path, top_k, debug);
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

        try analyzer.runSecondPass();

        const second_pass_time = timer.lap();
        const second_pass_ms = @as(f64, @floatFromInt(second_pass_time)) / time.ns_per_ms;
        std.debug.print("Pass 2 completed in {d:.2}ms\n", .{second_pass_ms});
    }

    // Get and display results
    std.debug.print("\n=== RESULTS ===\n", .{});
    try analyzer.getResults();

    // Display overall statistics
    const total_time = timer.read();
    const total_ms = @as(f64, @floatFromInt(total_time)) / time.ns_per_ms;
    std.debug.print("\n=== Processing Complete ===\n", .{});
    std.debug.print("Total execution time: {d:.2}ms\n", .{total_ms});
}
