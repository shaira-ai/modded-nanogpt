const std = @import("std");
const types = @import("types.zig");
const input_parser = @import("input_parser.zig");

// token count information for the inverted index

pub const BPE = struct {
    allocator: std.mem.Allocator,
    // single source of truth for tokens > _ <
    id_to_token: std.AutoHashMap(usize, types.Token),
    string_info: std.AutoHashMap(usize, types.StringInfo),
    strings: *input_parser.TopStringsByLength,
    next_token_id: usize,
    // enhanced inverted index with token counts
    inverted_index: std.AutoHashMap(usize, std.ArrayList(types.TokenCount)),
    war_scores: std.AutoHashMap(usize, f64),

    /// Initialize the BPE algorithm with basic byte tokens
    pub fn init(allocator: std.mem.Allocator, strings: *input_parser.TopStringsByLength) !BPE {
        var bpe = BPE{
            .allocator = allocator,
            .id_to_token = std.AutoHashMap(usize, types.Token).init(allocator),
            .string_info = std.AutoHashMap(usize, types.StringInfo).init(allocator),
            .strings = strings,
            .next_token_id = 256,
            .inverted_index = std.AutoHashMap(usize, std.ArrayList(types.TokenCount)).init(allocator),
            .war_scores = std.AutoHashMap(usize, f64).init(allocator),
        };

        try bpe.initializeByteTokens();
        try bpe.initializeSegmentations();

        return bpe;
    }

    pub fn deinit(self: *BPE) void {
        var token_it = self.id_to_token.valueIterator();
        while (token_it.next()) |token| {
            self.allocator.free(token.bytes);
        }
        self.id_to_token.deinit();

        var info_it = self.string_info.valueIterator();
        while (info_it.next()) |info| {
            self.allocator.free(info.segmentation);
        }
        self.string_info.deinit();

        var idx_it = self.inverted_index.valueIterator();
        while (idx_it.next()) |list| {
            list.deinit();
        }
        self.inverted_index.deinit();
        self.war_scores.deinit();
    }

    /// create the initial vocabulary with single-byte tokens.
    fn initializeByteTokens(self: *BPE) !void {
        var i: usize = 0;
        while (i < 256) : (i += 1) {
            var bytes = try self.allocator.alloc(u8, 1);
            bytes[0] = @intCast(i);

            const token = types.Token{
                .bytes = bytes,
                .id = i,
            };

            try self.id_to_token.put(i, token);
        }
    }

    /// create initial byte-level tokenization for all strings
    fn initializeSegmentations(self: *BPE) !void {
        var string_index: usize = 0;

        var length_it = self.strings.strings_by_length.iterator();
        while (length_it.next()) |entry| {
            const strings = entry.value_ptr.*.items;

            for (strings) |*string| {
                // Each byte becomes a token
                var segmentation = try self.allocator.alloc(usize, string.content.len);

                for (string.content, 0..) |byte, i| {
                    segmentation[i] = byte; // Token ID = byte value initially
                }

                // Store both segmentation and string pointer together
                try self.string_info.put(string_index, .{
                    .segmentation = segmentation,
                    .freq_string = string,
                });

                string_index += 1;
            }
        }
    }

    /// Count frequencies of adjacent token pairs across all strings
    fn countPairs(self: *BPE) !std.ArrayList(types.TokenPair) {
        var pair_counts = std.AutoHashMap(u64, usize).init(self.allocator);
        defer pair_counts.deinit();

        var string_index: usize = 0;

        var length_it = self.strings.strings_by_length.iterator();
        while (length_it.next()) |length_entry| {
            const strings = length_entry.value_ptr.*.items;

            for (strings) |_| {
                const string_data = self.string_info.get(string_index).?;
                const segmentation = string_data.segmentation;
                const frequency = string_data.freq_string.frequency;

                // Skip strings with fewer than 2 tokens
                if (segmentation.len < 2) {
                    string_index += 1;
                    continue;
                }

                // Count adjacent pairs, weighted by string frequency
                var i: usize = 0;
                while (i < segmentation.len - 1) : (i += 1) {
                    const first = segmentation[i];
                    const second = segmentation[i + 1];

                    // Create a unique key for the pair
                    const pair_key = (@as(u64, first) << 32) | @as(u64, second);

                    // Update count
                    if (pair_counts.contains(pair_key)) {
                        // Update existing count
                        pair_counts.getPtr(pair_key).?.* += frequency;
                    } else {
                        // Insert new count
                        try pair_counts.put(pair_key, frequency);
                    }
                }

                string_index += 1;
            }
        }

        // Convert hashmap to sorted list
        var pairs = std.ArrayList(types.TokenPair).init(self.allocator);

        var it = pair_counts.iterator();
        while (it.next()) |entry| {
            const pair_key = entry.key_ptr.*;
            const first: usize = @truncate(pair_key >> 32);
            const second: usize = @truncate(pair_key & 0xFFFFFFFF);

            try pairs.append(.{
                .first = first,
                .second = second,
                .frequency = entry.value_ptr.*,
            });
        }

        // Sort by frequency (descending)
        std.sort.insertion(types.TokenPair, pairs.items, {}, types.TokenPair.lessThan);

        return pairs;
    }

    /// Merge a pair of tokens into a new token
    fn mergePair(self: *BPE, pair: types.TokenPair) !void {
        const first_token = self.id_to_token.get(pair.first).?;
        const second_token = self.id_to_token.get(pair.second).?;

        // Create new token by concatenating the pair
        const new_bytes = try self.allocator.alloc(u8, first_token.bytes.len + second_token.bytes.len);

        std.mem.copyForwards(u8, new_bytes, first_token.bytes);
        std.mem.copyForwards(u8, new_bytes[first_token.bytes.len..], second_token.bytes);

        const new_token = types.Token{
            .bytes = new_bytes,
            .id = self.next_token_id,
        };

        const new_id = self.next_token_id;
        self.next_token_id += 1;

        // Add to vocabulary
        try self.id_to_token.put(new_id, new_token);

        // Update all segmentations
        var string_index: usize = 0;

        var length_it = self.strings.strings_by_length.iterator();
        while (length_it.next()) |length_entry| {
            const strings = length_entry.value_ptr.*.items;

            for (strings) |_| {
                const string_data = self.string_info.getPtr(string_index).?;
                const old_seg = string_data.segmentation;

                var new_seg = std.ArrayList(usize).init(self.allocator);
                defer new_seg.deinit();

                var i: usize = 0;
                while (i < old_seg.len) {
                    if (i < old_seg.len - 1 and
                        old_seg[i] == pair.first and
                        old_seg[i + 1] == pair.second)
                    {
                        try new_seg.append(new_id);
                        i += 2;
                    } else {
                        try new_seg.append(old_seg[i]);
                        i += 1;
                    }
                }

                // Update segmentation while keeping the FreqString pointer
                self.allocator.free(old_seg);
                const new_seg_slice = try self.allocator.alloc(usize, new_seg.items.len);
                std.mem.copyForwards(usize, new_seg_slice, new_seg.items);
                string_data.segmentation = new_seg_slice;

                string_index += 1;
            }
        }
    }

    fn findBestPair(self: *BPE) !?types.TokenPair {
        var pairs = try self.countPairs();
        defer pairs.deinit();

        if (pairs.items.len == 0) {
            return null;
        }

        // Return the pair with highest frequency
        return pairs.items[0];
    }

    /// Build the enhanced inverted index with token counts
    pub fn buildInvertedIndex(self: *BPE) !void {
        std.debug.print("Building inverted index...\n", .{});

        // Initialize inverted index with token counts
        self.inverted_index = std.AutoHashMap(usize, std.ArrayList(types.TokenCount)).init(self.allocator);

        // For each token, create an empty list
        var token_it = self.id_to_token.valueIterator();
        while (token_it.next()) |token| {
            try self.inverted_index.put(token.id, std.ArrayList(types.TokenCount).init(self.allocator));
        }

        // Scan all strings and record which tokens appear in each
        var string_index: usize = 0;
        var length_it = self.strings.strings_by_length.iterator();
        while (length_it.next()) |length_entry| {
            const strings = length_entry.value_ptr.*.items;

            for (strings) |_| {
                const string_data = self.string_info.get(string_index).?;
                const segmentation = string_data.segmentation;

                // Count occurrences of each token in this string
                var token_counts = std.AutoHashMap(usize, usize).init(self.allocator);
                defer token_counts.deinit();

                // Count each token occurrence
                for (segmentation) |token_id| {
                    const entry = try token_counts.getOrPut(token_id);
                    if (!entry.found_existing) {
                        entry.value_ptr.* = 0;
                    }
                    entry.value_ptr.* += 1;
                }

                // Add counts to the inverted index
                var count_it = token_counts.iterator();
                while (count_it.next()) |entry| {
                    const token_id = entry.key_ptr.*;
                    const count = entry.value_ptr.*;

                    try self.inverted_index.getPtr(token_id).?.append(.{
                        .string_index = string_index,
                        .count = count,
                    });
                }

                string_index += 1;
            }
        }

        // Print some stats about the inverted index
        var total_mappings: usize = 0;
        var token = self.inverted_index.iterator();
        while (token.next()) |entry| {
            total_mappings += entry.value_ptr.items.len;
        }

        std.debug.print("Inverted index built: {} tokens, {} total mappings\n", .{ self.inverted_index.count(), total_mappings });
    }

    /// Calculate the WAR score for a specific token with optimized string lookup
    fn calculateTokenWAR(self: *BPE, token_id: usize) !f64 {
        // Skip basic byte tokens (0-255)
        if (token_id < 256) {
            return std.math.inf(f64);
        }

        // Get the token counts from inverted index
        const token_counts = self.inverted_index.get(token_id) orelse return 0.0;
        if (token_counts.items.len == 0) {
            return 0.0;
        }

        var total_savings: f64 = 0.0;

        // For each string where this token appears
        for (token_counts.items) |token_count| {
            const string_data = self.string_info.get(token_count.string_index) orelse continue;

            // Each occurrence adds one token when removed (our +1 model)
            const saving = @as(f64, @floatFromInt(token_count.count)) *
                @as(f64, @floatFromInt(string_data.freq_string.frequency));
            total_savings += saving;
        }

        return total_savings;
    }

    /// Calculate WAR scores for all tokens in the vocabulary
    pub fn calculateWAR(self: *BPE) !void {
        std.debug.print("Calculating WAR scores...\n", .{});

        // Initialize WAR scores map
        self.war_scores = std.AutoHashMap(usize, f64).init(self.allocator);

        // Calculate WAR for each token
        var token_it = self.id_to_token.valueIterator();
        while (token_it.next()) |token| {
            const war = try self.calculateTokenWAR(token.id);
            try self.war_scores.put(token.id, war);
        }

        std.debug.print("WAR calculation complete for {} tokens\n", .{self.war_scores.count()});
    }

    /// Find the token with the lowest WAR score
    fn findLowestWARToken(self: *BPE) ?usize {
        var lowest_id: ?usize = null;
        var lowest_war: f64 = std.math.inf(f64);

        var it = self.war_scores.iterator();
        while (it.next()) |entry| {
            const token_id = entry.key_ptr.*;
            const war = entry.value_ptr.*;

            // Skip byte tokens (0-255) - we always keep these
            if (token_id < 256) {
                continue;
            }

            if (war < lowest_war) {
                lowest_war = war;
                lowest_id = token_id;
            }
        }

        return lowest_id;
    }

    /// Update segmentations after removing a token
    fn updateSegmentationsAfterRemoval(self: *BPE, removed_id: usize) !void {
        const removed_token = self.id_to_token.get(removed_id).?;

        // Find the component tokens this token was formed from
        // In practice, we'd need to track merge history, but for simplicity
        // we'll just fall back to character-level segmentation for affected tokens

        std.debug.print("Updating segmentations after removing token {}: '{s}'\n", .{ removed_id, removed_token.bytes });

        // Get strings where the removed token appears
        const affected_indices = self.inverted_index.get(removed_id).?.items;

        for (affected_indices) |token_count| {
            const string_index = token_count.string_index;
            const string_data = self.string_info.getPtr(string_index).?;
            const old_seg = string_data.segmentation;

            // Create new segmentation, replacing the removed token
            var new_seg = std.ArrayList(usize).init(self.allocator);
            defer new_seg.deinit();

            for (old_seg) |token_id| {
                if (token_id == removed_id) {
                    // Replace with individual bytes
                    for (removed_token.bytes) |byte| {
                        try new_seg.append(byte);
                    }
                } else {
                    try new_seg.append(token_id);
                }
            }

            // Update the segmentation
            self.allocator.free(old_seg);
            const new_seg_slice = try self.allocator.alloc(usize, new_seg.items.len);
            std.mem.copyForwards(usize, new_seg_slice, new_seg.items);
            string_data.segmentation = new_seg_slice;
        }
    }

    /// remove the token with the lowest WAR score
    pub fn removeLowestWARToken(self: *BPE) !void {
        // Find the token with the lowest WAR
        const lowest_id = self.findLowestWARToken();
        if (lowest_id == null) {
            std.debug.print("No suitable token found for removal\n", .{});
            return;
        }

        const removed_id = lowest_id.?;
        const removed_token = self.id_to_token.get(removed_id).?;
        const war_score = self.war_scores.get(removed_id).?;

        std.debug.print("Removing token with lowest WAR: ID {}, '{s}', WAR: {d:.2}\n", .{ removed_id, removed_token.bytes, war_score });

        // update segmentations that used this token
        try self.updateSegmentationsAfterRemoval(removed_id);

        // remove the token from vocabulary
        self.allocator.free(removed_token.bytes);
        _ = self.id_to_token.remove(removed_id);

        // remove from inverted index and war scores
        self.inverted_index.getPtr(removed_id).?.deinit();
        _ = self.inverted_index.remove(removed_id);
        _ = self.war_scores.remove(removed_id);

        std.debug.print("Token removed. New vocabulary size: {}\n", .{self.id_to_token.count()});
    }

    /// Execute the full BPE with WAR pipeline
    pub fn trainWithWAR(self: *BPE, target_vocab_size: usize) !void {
        try self.train(target_vocab_size);

        try self.buildInvertedIndex();

        // Calculate WAR for all tokens
        try self.calculateWAR();

        // Print some WAR stats
        std.debug.print("\nToken WAR Statistics:\n", .{});
        var highest_war: f64 = -std.math.inf(f64);
        var lowest_war: f64 = std.math.inf(f64);
        var highest_id: usize = 0;
        var lowest_id: usize = 0;

        var it = self.war_scores.iterator();
        while (it.next()) |entry| {
            const token_id = entry.key_ptr.*;
            const war = entry.value_ptr.*;

            // Skip byte tokens for stats
            if (token_id < 256) {
                continue;
            }

            if (war > highest_war) {
                highest_war = war;
                highest_id = token_id;
            }

            if (war < lowest_war) {
                lowest_war = war;
                lowest_id = token_id;
            }
        }

        if (highest_id > 0 and lowest_id > 0) {
            const highest_token = self.id_to_token.get(highest_id).?;
            const lowest_token = self.id_to_token.get(lowest_id).?;

            std.debug.print("  Highest WAR: {d:.2} - Token {}: '{s}'\n", .{ highest_war, highest_id, highest_token.bytes });
            std.debug.print("  Lowest WAR: {d:.2} - Token {}: '{s}'\n", .{ lowest_war, lowest_id, lowest_token.bytes });
        }

        // Remove the lowest WAR token
        try self.removeLowestWARToken();

        // Print final vocabulary size
        std.debug.print("\nFinal vocabulary size after WAR pruning: {}\n", .{self.id_to_token.count()});
    }

    /// run the BPE algorithm to build a vocabulary of the target size
    pub fn train(self: *BPE, target_vocab_size: usize) !void {
        std.debug.print("Starting BPE training with target vocabulary size: {}\n", .{target_vocab_size});

        // run standard BPE until reaching target size
        while (self.id_to_token.count() < target_vocab_size) {
            const best_pair = try self.findBestPair();

            if (best_pair == null) {
                std.debug.print("No more pairs to merge\n", .{});
                break;
            }

            std.debug.print("Merging pair ({}, {}) with frequency {}\n", .{ best_pair.?.first, best_pair.?.second, best_pair.?.frequency });
            try self.mergePair(best_pair.?);
            std.debug.print("Vocabulary size: {}\n", .{self.id_to_token.count()});
        }

        // additional merge
        if (self.id_to_token.count() >= target_vocab_size) {
            std.debug.print("Performing one additional merge beyond target size\n", .{});

            const extra_pair = try self.findBestPair();
            if (extra_pair != null) {
                try self.mergePair(extra_pair.?);
            } else {
                std.debug.print("No more pairs to merge for the extra step\n", .{});
            }
        }

        std.debug.print("Final vocabulary size: {}\n", .{self.id_to_token.count()});
    }

    /// print statistics and examples from the vocabulary
    pub fn printVocabularyStats(self: *BPE) void {
        std.debug.print("\nVocabulary Statistics:\n", .{});
        std.debug.print("  Total tokens: {}\n", .{self.id_to_token.count()});

        // print some example tokens
        std.debug.print("\nSample byte tokens:\n", .{});
        var i: usize = 'a';
        while (i <= 'e' and i < 256) : (i += 1) {
            const token = self.id_to_token.get(i).?;
            std.debug.print("  ID {}: '{s}'\n", .{ token.id, token.bytes });
        }

        // print some merged tokens
        std.debug.print("\nSample merged tokens:\n", .{});
        var count: usize = 0;
        var token_id: usize = 256;
        while (count < 5 and token_id < self.next_token_id) : (token_id += 1) {
            if (self.id_to_token.get(token_id)) |token| {
                std.debug.print("  ID {}: '{s}'\n", .{ token.id, token.bytes });
                count += 1;
            }
        }
    }

    /// print examples of tokenized strings
    pub fn printSegmentationExamples(self: *BPE) void {
        std.debug.print("\nSegmentation Examples:\n", .{});

        var shown: usize = 0;
        var string_index: usize = 0;

        // iterate through lengths
        var length_it = self.strings.strings_by_length.iterator();
        while (length_it.next()) |entry| {
            const strings = entry.value_ptr.*.items;

            // show first string of each length
            if (strings.len > 0) {
                const string = strings[0];
                const string_data = self.string_info.get(string_index).?;
                const seg = string_data.segmentation;

                std.debug.print("  '{s}' -> [", .{string.content});

                for (seg, 0..) |token_id, i| {
                    const token = self.id_to_token.get(token_id).?;
                    if (i > 0) std.debug.print(", ", .{});
                    std.debug.print("'{s}'", .{token.bytes});
                }

                std.debug.print("] ({} tokens)\n", .{seg.len});

                shown += 1;
                if (shown >= 5) break;
            }

            string_index += strings.len;
        }
    }
};

/// test function for the BPE implementation with WAR
pub fn testBPEWithWAR() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // sample data for testing
    const test_input =
        \\3,cat,5
        \\3,dog,7
        \\4,frog,3
        \\4,bird,4
        \\5,hello,10
        \\5,world,8
        \\6,planet,2
        \\6,galaxy,3
        \\7,monkeys,5
        \\8,elephant,2
        \\10,strawberry,1
        \\10,watermelon,3
        \\9,pineapple,2
        \\12,hippopotamus,1
        \\13,communication,4
        \\14,transformation,3
        \\13,extraordinary,2
        \\13,international,5
        \\13,comprehensive,3
        \\15,extraordinarily,2
    ;

    var strings = try input_parser.parseInput(allocator, test_input);
    defer strings.deinit();

    std.debug.print("Loaded {} strings across {} different lengths\n", .{ strings.total_count, strings.strings_by_length.count() });

    // initialize and run BPE with WAR
    var bpe = try BPE.init(allocator, &strings);
    defer bpe.deinit();

    const target_size = 270;
    try bpe.trainWithWAR(target_size);

    bpe.printVocabularyStats();
    bpe.printSegmentationExamples();
}
