// Main file example
const std = @import("std");
const fineweb = @import("data_loader.zig").FinewebDataLoader;
const SFM = @import("string_frequency_manager.zig").StringFrequencyManager;

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    // Configure parameters
    const min_length = 2;
    const max_length = 256;
    const top_k = 100_000; // Track top 100k strings per length
    const cms_width = 1 << 24; // ~16 million counters per hash function
    const cms_depth = 5; // 5 hash functions

    // Create the manager
    const manager = try SFM(cms_width, cms_depth).init(allocator, min_length, max_length, top_k);
    defer manager.deinit();

    // Load documents and process them
    var loader = try fineweb.init(allocator, "fineweb_train_000001.bin");
    defer loader.deinit();
    try loader.loadVocabulary("vocab.json");

    // Pass 1: Build CMS
    std.debug.print("=== PASS 1: Building Count-Min Sketch ===\n", .{});
    var doc_count: usize = 0;
    while (true) {
        const doc = try loader.nextDocumentString();
        if (doc == null) break;
        defer allocator.free(doc.?);

        try manager.buildCMS(doc.?);
        doc_count += 1;

        if (doc_count % 100 == 0) {
            std.debug.print("Processed {d} documents in pass 1\n", .{doc_count});
        }
    }

    // Reset loader for second pass
    loader.deinit();
    loader = try fineweb.init(allocator, "fineweb_train_000001.bin");
    try loader.loadVocabulary("vocab.json");

    // Pass 2: Find top strings and track counts
    std.debug.print("\n=== PASS 2: Finding Top Strings ===\n", .{});
    doc_count = 0;
    while (true) {
        const doc = try loader.nextDocumentString();
        if (doc == null) break;
        defer allocator.free(doc.?);

        try manager.processDocumentSecondPass(doc.?);
        doc_count += 1;

        if (doc_count % 100 == 0) {
            std.debug.print("Processed {d} documents in pass 2\n", .{doc_count});
        }
    }

    // Show results
    try manager.getResults();
}
