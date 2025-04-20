const std = @import("std");
const fineweb = @import("data_loader.zig").FinewebDataLoader;
const SFM = @import("string_frequency_manager.zig").StringFrequencyManager;
const fs = std.fs;

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    // Configure parameters
    const min_length = 2;
    const max_length = 256;
    const top_k = 100_000; // Track top 100k strings per length
    const cms_width = 1 << 24; // ~16 million counters per hash function
    const cms_depth = 5; // 5 hash functions

    const saved_data_path = "fineweb_first_pass.bin";
    var manager: *SFM(cms_width, cms_depth) = undefined;

    // Check if saved first pass data exists
    const saved_data_exists = blk: {
        const file = fs.cwd().openFile(saved_data_path, .{}) catch |err| {
            if (err == error.FileNotFound) {
                break :blk false;
            }
            return err;
        };
        file.close();
        break :blk true;
    };

    if (saved_data_exists) {
        // Load the first pass data from disk
        std.debug.print("=== LOADING FIRST PASS DATA FROM DISK ===\n", .{});
        manager = try SFM(cms_width, cms_depth).loadFirstPassFromDisk(allocator, saved_data_path);
        std.debug.print("First pass data loaded successfully.\n", .{});
    } else {
        // Create the manager
        manager = try SFM(cms_width, cms_depth).init(allocator, min_length, max_length, top_k);

        // Load documents and process them
        var loader = try fineweb.init(allocator, "fineweb_train_000001.bin");
        defer loader.deinit();
        try loader.loadVocabulary("vocab.json");

        // Pass 1: Build CMS and direct counters for length 2 and 3
        std.debug.print("=== PASS 1: Building Count-Min Sketch and Direct Counters ===\n", .{});
        var doc_count: usize = 0;
        const start_time = std.time.nanoTimestamp();

        while (true) {
            const doc = try loader.nextDocumentString();
            if (doc == null) break;
            defer allocator.free(doc.?);

            try manager.buildCMS(doc.?);
            doc_count += 1;

            if (doc_count % 20 == 0) {
                std.debug.print("Processed {d} documents in pass 1\n", .{doc_count});
            }
        }

        const elapsed = std.time.nanoTimestamp() - start_time;
        std.debug.print("Pass 1 completed in {d:.2}ms\n", .{@as(f64, @floatFromInt(elapsed)) / std.time.ns_per_ms});

        // Save first pass data for future runs
        std.debug.print("Saving first pass data to disk...\n", .{});
        try manager.saveFirstPassToDisk(saved_data_path);
        std.debug.print("First pass data saved successfully.\n", .{});
    }

    defer manager.deinit();

    var loader = try fineweb.init(allocator, "fineweb_train_000001.bin");
    defer loader.deinit();
    try loader.loadVocabulary("vocab.json");

    // Pass 2: Find top strings and track counts
    std.debug.print("\n=== PASS 2: Finding Top Strings ===\n", .{});
    var doc_count: usize = 0;
    const start_time = std.time.nanoTimestamp();

    while (true) {
        const doc = try loader.nextDocumentString();
        if (doc == null) break;
        defer allocator.free(doc.?);

        try manager.processDocumentSecondPass(doc.?);
        doc_count += 1;

        if (doc_count % 20 == 0) {
            std.debug.print("Processed {d} documents in pass 2\n", .{doc_count});
        }
    }

    const elapsed = std.time.nanoTimestamp() - start_time;
    std.debug.print("Pass 2 completed in {d:.2}ms\n", .{@as(f64, @floatFromInt(elapsed)) / std.time.ns_per_ms});

    try manager.getResults();
}
