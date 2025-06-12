const std = @import("std");
const BakaCorasick = @import("baka_corasick.zig").BakaCorasick;
const fineweb = @import("data_loader.zig").FinewebDataLoader;
const InMemoryDataLoader = @import("in_mem_dataloader.zig").InMemoryDataLoader;
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const parallel = @import("parallel.zig");

pub const TokenStats = extern struct {
    str_start_idx: u64,
    str_len: u16,
    len_in_tokens: u16,
    is_in_vocab: bool = false,
    n_uses_for_cover: u64 = 0,
    covered_step: u64 = 0,
    n_covered_occurrences: u64 = 0,
    n_occurrences: u64 = 0,
    n_nonoverlapping_occurrences: u64 = 0,
    sampled_occurrences: u64 = 0,
    sampled_savings: u64 = 0,
    sampled_step: u64 = 0,
    est_total_savings: f64 = 0,

    pub inline fn getCurrentValueBound(self: TokenStats) f64 {
        return self.est_total_savings;
    }
};

pub const SampleStats = struct {
    sampled_occurrences: u64 = 0,
    sampled_savings: u64 = 0,
    token_id: u32,
};

pub const MatchInfo = struct {
    bits: u64,
    pub inline fn init(token_id: u32, end_pos: u32) MatchInfo {
        return .{
            .bits = (@as(u64, token_id) << 32) | end_pos,
        };
    }
    pub inline fn getTokenId(self: MatchInfo) u32 {
        return @intCast(self.bits >> 32);
    }
    pub inline fn getEndPos(self: MatchInfo) u32 {
        return @truncate(self.bits);
    }
};

const CANDIDATE_TOKEN_FLAG: u32 = 1 << 31;

pub const DocumentSampler = struct {
    allocator: Allocator,
    corpus_paths: [][]const u8,
    prng: std.Random,
    debug: bool,

    pub fn init(allocator: Allocator, corpus_paths: []const []const u8, debug: bool) !*DocumentSampler {
        const sampler = try allocator.create(DocumentSampler);

        // Make a copy of the corpus paths
        var paths_copy = try allocator.alloc([]const u8, corpus_paths.len);
        for (corpus_paths, 0..) |path, i| {
            paths_copy[i] = try allocator.dupe(u8, path);
        }

        sampler.* = .{
            .allocator = allocator,
            .corpus_paths = paths_copy,
            .prng = std.crypto.random,
            .debug = debug,
        };

        return sampler;
    }

    pub fn deinit(self: *DocumentSampler) void {
        for (self.corpus_paths) |path| {
            self.allocator.free(path);
        }
        self.allocator.free(self.corpus_paths);
        self.allocator.destroy(self);
    }
};

const STATS_MAGIC = "TOKSTAT".*;
const STATS_HEADER_SIZE = 64; // Increased to 64 bytes for more flexibility

const StatsHeader = extern struct {
    magic: [7]u8,
    pad_a: [1]u8,
    vocab_size: u32,
    n_token_ids: u32,
    file_count: u32,
    pad_b: [4]u8,
    timestamp: i64,
    reserved: [32]u8, // Padding to reach 64 bytes

    // Ensure the struct is exactly 64 bytes
    comptime {
        if (@sizeOf(StatsHeader) != STATS_HEADER_SIZE) {
            @compileError("StatsHeader size mismatch");
        }
    }
};

const VOCAB_MAGIC = "VOCA".*;
const HEADER_SIZE = 28;

// Change from packed struct to regular struct
const VocabHeader = extern struct {
    magic: [4]u8,
    vocab_size: u32,
    reserved: [20]u8,

    // Ensure the struct is exactly 32 bytes
    comptime {
        if (@sizeOf(VocabHeader) != HEADER_SIZE) {
            @compileError("VocabHeader size mismatch");
        }
    }
};

const buildVocabLessThan = struct {
    fn lessThan(ctx: *VocabLearner, a: u32, b: u32) std.math.Order {
        const value_a = ctx.candidate_stats[a].getCurrentValueBound();
        const value_b = ctx.candidate_stats[b].getCurrentValueBound();
        return std.math.order(value_b, value_a);
    }
}.lessThan;

const buildVocabGreaterThan = struct {
    fn lessThan(ctx: *VocabLearner, a: u32, b: u32) std.math.Order {
        const value_a = ctx.candidate_stats[a].getCurrentValueBound();
        const value_b = ctx.candidate_stats[b].getCurrentValueBound();
        return std.math.order(value_a, value_b);
    }
}.lessThan;

const COVER_INTERVAL = 1000;

pub const VocabLearner = struct {
    allocator: Allocator,
    candidate_stats: []TokenStats,
    vocab_automaton: BakaCorasick,
    eval_automaton: BakaCorasick,
    current_step: u32,
    loader: ?*anyopaque = null,
    use_in_memory: bool = false,
    document_sampler: *DocumentSampler,
    n_token_ids: u32,
    vocab_size: u32,
    tokenset_contents: []const u8,
    // Parameters
    max_token_length: u32 = 60,
    max_vocab_size: u32,
    top_k_candidates: u32,
    batch_size: u32,
    sample_size: u32,
    n_candidates_to_tokenize: u32,
    processed_files: std.StringHashMap(void),

    // Tracking
    last_full_corpus_scan: u32,
    debug: bool,

    fn getLoader(self: *VocabLearner, comptime LoaderType: type) *LoaderType {
        return @ptrCast(@alignCast(self.loader.?));
    }

    inline fn initLearner(
        learner: *VocabLearner,
        allocator: Allocator,
        corpus_paths: []const []const u8,
        max_vocab_size: u32,
        use_in_memory: bool,
        debug: bool,
    ) !void {
        learner.* = .{
            .allocator = allocator,
            .candidate_stats = &[_]TokenStats{},
            .vocab_automaton = try BakaCorasick.init(allocator),
            .eval_automaton = try BakaCorasick.init(allocator),
            .current_step = 0,
            .document_sampler = try DocumentSampler.init(allocator, corpus_paths, debug),
            .n_token_ids = 0,
            .vocab_size = 0,
            .tokenset_contents = &[_]u8{},
            .max_vocab_size = max_vocab_size,
            .top_k_candidates = 50,
            .batch_size = 10,
            .sample_size = 40000,
            .n_candidates_to_tokenize = 500,
            .processed_files = std.StringHashMap(void).init(allocator),
            .last_full_corpus_scan = 0,
            .use_in_memory = use_in_memory,
            .debug = debug,
        };
    }

    pub fn init(allocator: Allocator, input_tokenset_path: []const u8, corpus_paths: []const []const u8, max_vocab_size: u32, use_in_memory: bool, debug: bool) !*VocabLearner {
        var learner = try allocator.create(VocabLearner);

        // Initialize fields
        try initLearner(
            learner,
            allocator,
            corpus_paths,
            max_vocab_size,
            use_in_memory,
            debug,
        );

        // Load candidate tokens from tokenset file
        try learner.loadCandidateTokens(input_tokenset_path);

        // Initialize with 256 single-byte tokens
        try learner.initializeWithByteTokens();

        return learner;
    }

    pub fn deinit(self: *VocabLearner) void {
        // Free candidate stats (each contains its own token)
        self.allocator.free(self.candidate_stats);

        var cache_it = self.processed_files.iterator();
        while (cache_it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.processed_files.deinit();

        // Clean up BakaCorasick instances
        self.vocab_automaton.deinit();
        self.eval_automaton.deinit();

        if (self.loader) |_| {
            if (self.use_in_memory) {
                const loader = self.getLoader(InMemoryDataLoader);
                loader.deinit();
            } else {
                const loader = self.getLoader(fineweb);
                loader.deinit();
            }
        }

        // Free document sampler
        self.document_sampler.deinit();
        self.allocator.free(self.tokenset_contents);

        // Free self
        self.allocator.destroy(self);
    }

    pub fn addToVocab(self: *VocabLearner, token_id: u32) void {
        if (!self.candidate_stats[token_id].is_in_vocab) {
            self.candidate_stats[token_id].is_in_vocab = true;
            self.vocab_size += 1;
        }
    }

    pub fn removeFromVocab(self: *VocabLearner, token_id: u32) void {
        if (self.candidate_stats[token_id].is_in_vocab) {
            self.candidate_stats[token_id].is_in_vocab = false;
            self.vocab_size -= 1;
        }
    }

    pub fn getTokenStr(self: *const VocabLearner, token_id: u32) []const u8 {
        const start_idx = self.candidate_stats[token_id].str_start_idx;
        const len = self.candidate_stats[token_id].str_len;
        return self.tokenset_contents[start_idx .. start_idx + len];
    }

    // Initialize with 256 single-byte tokens
    fn initializeWithByteTokens(self: *VocabLearner) !void {
        if (self.debug) {
            std.debug.print("Initializing with 256 single-byte tokens...\n", .{});
        }

        for (0..256) |id_usize| {
            const token_id: u32 = @intCast(id_usize);
            const stats = self.candidate_stats[token_id];
            const token_str = self.getTokenStr(token_id);
            if (stats.is_in_vocab or stats.str_len != 1) {
                return error.InvalidToken;
            }
            self.addToVocab(token_id);
            try self.vocab_automaton.insert(token_str, token_id);
        }

        try self.vocab_automaton.computeSuffixLinks();

        if (self.debug) {
            std.debug.print("Added 256 single-byte tokens to vocabulary.\n", .{});
        }
    }

    // Load candidate tokens from tokenset file
    fn loadCandidateTokens(self: *VocabLearner, input_path: []const u8) !void {
        const start_time = std.time.milliTimestamp();

        if (self.debug) {
            std.debug.print("Loading candidate tokens from {s}...\n", .{input_path});
        }

        const file = try std.fs.cwd().openFile(input_path, .{});
        defer file.close();

        // Get the file size
        const file_size = try file.getEndPos();

        var header: [256]u32 = undefined;
        const header_size = 256 * @sizeOf(u32);
        const bytes_read = try file.readAll(std.mem.asBytes(&header));
        if (bytes_read != header_size) {
            return error.IncompleteHeader;
        }

        // Allocate a buffer of the appropriate size
        const buffer = try self.allocator.alloc(u8, file_size - header_size);
        const bytes_read_again = try file.readAll(buffer);
        self.tokenset_contents = buffer;
        if (bytes_read_again != self.tokenset_contents.len) {
            return error.IncompleteTokenData;
        }

        // check that the file contains all 1-byte values.
        // like, I know we could do this in code, but it's easier to
        // put it in the file, ok.
        var seen_one_byte_values: [256]bool = [_]bool{false} ** 256;
        if (self.tokenset_contents.len < 256) {
            return error.IncompleteTokenData;
        }
        for (0..256) |i| {
            seen_one_byte_values[self.tokenset_contents[i]] = true;
        }
        for (seen_one_byte_values) |seen_one_byte_value| {
            if (!seen_one_byte_value) {
                return error.IncompleteTokenData;
            }
        }

        var total_tokens: usize = 0;
        var total_expected_size_according_to_header: usize = 0;
        for (header, 1..) |count, length| {
            if (length > self.max_token_length) continue;
            total_tokens += count;
            total_expected_size_according_to_header += count * length;
        }
        self.n_token_ids = @intCast(total_tokens);

        if (total_expected_size_according_to_header > self.tokenset_contents.len) {
            return error.IncompleteTokenData;
        }

        self.candidate_stats = try self.allocator.alloc(TokenStats, self.n_token_ids);
        for (self.candidate_stats) |*stats| {
            stats.* = .{
                .str_start_idx = 0,
                .str_len = 0,
                .len_in_tokens = 0,
            };
        }

        if (self.debug) {
            std.debug.print("Found {d} total candidate tokens.\n", .{self.n_token_ids});
        }

        var token_id: u32 = 0;
        var content_offset: usize = 0;

        for (header, 1..) |count, length| {
            if (length > self.max_token_length) continue;

            if (self.debug) {
                std.debug.print("Processing {d} tokens of length {d}...\n", .{ count, length });
            }

            for (0..count) |_| {
                self.candidate_stats[token_id].str_start_idx = @intCast(content_offset);
                self.candidate_stats[token_id].str_len = @intCast(length);
                self.candidate_stats[token_id].len_in_tokens = @intCast(length);
                token_id += 1;
                content_offset += length;
            }
        }

        const elapsed = std.time.milliTimestamp() - start_time;

        if (self.debug) {
            std.debug.print("Loaded {d} candidate tokens in {d}ms.\n", .{ total_tokens, elapsed });
        }
    }

    pub fn checkPhase1Initialization(self: *VocabLearner) !void {
        if (self.vocab_automaton.len < 257) {
            return error.AutomatonInitializationFailed;
        }

        if (self.debug) {
            std.debug.print("Phase 1: Initialized with {d} vocab tokens, {d} total tokens loaded.\n", .{ self.vocab_size, self.n_token_ids });
        }
    }

    fn cleanupBinFiles(self: *VocabLearner, corpus_files: *ArrayList([]const u8)) void {
        for (corpus_files.items) |path| {
            self.allocator.free(path);
        }
        corpus_files.deinit();
    }

    fn initializeLoader(self: *VocabLearner) !void {
        if (self.loader == null) {
            var corpus_files = try self.collectBinFiles();
            defer self.cleanupBinFiles(&corpus_files);

            if (self.use_in_memory) {
                if (self.debug) {
                    std.debug.print("Initializing in-memory data loader...\n", .{});
                }
                const in_memory_loader = try InMemoryDataLoader.init(self.allocator, corpus_files.items);
                self.loader = @ptrCast(in_memory_loader);
            } else {
                if (self.debug) {
                    std.debug.print("Initializing streaming data loader...\n", .{});
                }
                var fineweb_loader = try fineweb.init(self.allocator, corpus_files.items);
                try fineweb_loader.loadVocabulary("vocab.json");
                self.loader = @ptrCast(fineweb_loader);
            }
        }
    }

    inline fn processCorpusBatch(self: *VocabLearner, comptime LoaderType: type, batch_automaton: *BakaCorasick, token_id_to_stats: anytype, position: *u64, tokens_recorded: *u64) !void {
        const loader = self.getLoader(LoaderType);
        try loader.rewind();
        while (try loader.nextDocumentString()) |text| {
            defer self.allocator.free(text);

            var current_state: u32 = 0;
            for (text) |byte| {
                current_state = batch_automaton.transitions[current_state][byte];

                {
                    const token_id = batch_automaton.info[current_state].token_id;
                    if (token_id != BakaCorasick.NO_TOKEN) {
                        const token_len = batch_automaton.info[current_state].depth;
                        if (position.* >= token_id_to_stats[token_id].next_valid_position) {
                            const next_valid_position = position.* + token_len;
                            token_id_to_stats[token_id].next_valid_position = next_valid_position;
                            token_id_to_stats[token_id].n_nonoverlapping_occurrences += 1;
                            tokens_recorded.* += 1;
                        }
                    }
                }

                var suffix = batch_automaton.info[current_state].green;
                while (suffix != 0) {
                    const token_id = batch_automaton.info[suffix].token_id;
                    if (token_id != BakaCorasick.NO_TOKEN) {
                        const token_len = batch_automaton.info[suffix].depth;
                        if (position.* >= token_id_to_stats[token_id].next_valid_position) {
                            const next_valid_position = position.* + token_len;
                            token_id_to_stats[token_id].next_valid_position = next_valid_position;
                            token_id_to_stats[token_id].n_nonoverlapping_occurrences += 1;
                            tokens_recorded.* += 1;
                        }
                    }
                    suffix = batch_automaton.info[suffix].green;
                }
                position.* += 1;
            }
        }
    }

    inline fn countSingleByteTokens(self: *VocabLearner, comptime LoaderType: type, single_byte_counts: []u64) !void {
        const loader = self.getLoader(LoaderType);
        try loader.rewind();

        while (try loader.nextDocumentString()) |text| {
            defer if (LoaderType.NEEDS_DEALLOCATION) self.allocator.free(text);
            for (text) |byte| {
                single_byte_counts[byte] += 1;
            }
        }
    }

    pub fn processCorpus(self: *VocabLearner) !void {
        const start_time = std.time.milliTimestamp();

        if (self.debug) {
            std.debug.print("Processing corpus using raw text conversion...\n", .{});
        }

        try self.initializeLoader();

        const automaton_size = 10_000_000;
        std.debug.print("Using target automaton size of {d} states\n", .{automaton_size});

        const NonoverlappingStats = struct {
            n_nonoverlapping_occurrences: u64,
            next_valid_position: u64,
        };

        const token_id_to_stats = try self.allocator.alloc(NonoverlappingStats, self.n_token_ids);
        defer self.allocator.free(token_id_to_stats);

        const token_ids_buffer = try self.allocator.alloc(u32, self.n_token_ids);
        defer self.allocator.free(token_ids_buffer);

        var batch_automaton = try BakaCorasick.init(self.allocator);
        defer batch_automaton.deinit();

        var token_count: usize = 0;
        for (0..self.n_token_ids) |id_usize| {
            const id: u32 = @intCast(id_usize);
            if (self.getTokenStr(id).len > 1) {
                token_ids_buffer[token_count] = id;
                token_count += 1;
            }
        }

        const tokens_to_process = token_ids_buffer[0..token_count];
        if (self.debug) {
            std.debug.print("Found {d} multi-byte tokens to process\n", .{token_count});
        }

        for (self.candidate_stats) |*stats| {
            stats.n_nonoverlapping_occurrences = 0;
            stats.est_total_savings = 0;
        }

        var batch_num: usize = 1;
        var start_idx: usize = 0;
        var total_tokens_processed: u64 = 0;

        while (start_idx < tokens_to_process.len) {
            const batch_start_time = std.time.milliTimestamp();

            batch_automaton.clear();

            // reset stats for this batch
            @memset(token_id_to_stats, .{ .n_nonoverlapping_occurrences = 0, .next_valid_position = 0 });

            // add tokens until we reach target automaton size
            var current_idx = start_idx;
            var estimated_automaton_size: usize = 0;

            while (current_idx < tokens_to_process.len) {
                const token_id = tokens_to_process[current_idx];
                const token_str = self.getTokenStr(token_id);

                // estimate automaton size increase
                const estimated_increase = token_str.len;

                // stop if adding this token would exceed target size!
                if (estimated_automaton_size > 0 and
                    estimated_automaton_size + estimated_increase > automaton_size)
                {
                    break;
                }

                // add token to the automaton
                try batch_automaton.insert(token_str, token_id);
                estimated_automaton_size = batch_automaton.len;
                current_idx += 1;
            }

            try batch_automaton.computeSuffixLinks();

            const actual_end_idx = current_idx;
            const tokens_in_batch = actual_end_idx - start_idx;

            if (self.debug) {
                std.debug.print("Batch {d}: Processing tokens {d}-{d} ({d} tokens), automaton size: {d} states\n", .{ batch_num, start_idx, actual_end_idx - 1, tokens_in_batch, batch_automaton.len });
            }

            var position: u64 = 0;
            var tokens_recorded: u64 = 0;

            if (self.use_in_memory) {
                try self.processCorpusBatch(InMemoryDataLoader, &batch_automaton, token_id_to_stats, &position, &tokens_recorded);
            } else {
                try self.processCorpusBatch(fineweb, &batch_automaton, token_id_to_stats, &position, &tokens_recorded);
            }

            // update main stats with results from this batch
            for (start_idx..actual_end_idx) |idx| {
                const token_id = tokens_to_process[idx];
                self.candidate_stats[token_id].n_nonoverlapping_occurrences =
                    token_id_to_stats[token_id].n_nonoverlapping_occurrences;
                self.candidate_stats[token_id].est_total_savings = @floatFromInt(token_id_to_stats[token_id].n_nonoverlapping_occurrences *
                    (self.candidate_stats[token_id].str_len - 1));
            }

            total_tokens_processed += tokens_recorded;

            const batch_elapsed_sec = @as(f64, @floatFromInt(std.time.milliTimestamp() - batch_start_time)) / 1000.0;
            if (self.debug) {
                std.debug.print("Batch {d} completed in {d:.2}s. Recorded {d} tokens.\n", .{ batch_num, batch_elapsed_sec, tokens_recorded });
            }

            start_idx = actual_end_idx;
            batch_num += 1;

            if (start_idx >= tokens_to_process.len) break;
        }

        // Process single-byte tokens
        if (self.debug) {
            std.debug.print("Processing single-byte tokens...\n", .{});
        }

        const single_byte_counts = try self.allocator.alloc(u64, 256);
        defer self.allocator.free(single_byte_counts);
        @memset(single_byte_counts, 0);

        if (self.use_in_memory) {
            try self.countSingleByteTokens(InMemoryDataLoader, single_byte_counts);
        } else {
            try self.countSingleByteTokens(fineweb, single_byte_counts);
        }

        for (0..256) |id| {
            const token_id: u32 = @intCast(id);
            self.candidate_stats[token_id].n_nonoverlapping_occurrences = single_byte_counts[id];
            self.candidate_stats[token_id].est_total_savings = 0;
        }

        const elapsed_sec = @as(f64, @floatFromInt(std.time.milliTimestamp() - start_time)) / 1000.0;
        if (self.debug) {
            std.debug.print("\nProcessed total of {d} tokens in {d} batches\n", .{ total_tokens_processed, batch_num - 1 });
            std.debug.print("Completed corpus processing in {d:.2}s\n", .{elapsed_sec});
        }
    }

    fn collectBinFiles(self: *VocabLearner) !ArrayList([]const u8) {
        var result = ArrayList([]const u8).init(self.allocator);

        // Process each corpus path
        for (self.document_sampler.corpus_paths) |corpus_path| {
            if (self.debug) {
                std.debug.print("Looking for .bin files in: {s}\n", .{corpus_path});
            }

            // Try to open as directory
            var dir = std.fs.cwd().openDir(corpus_path, .{ .iterate = true }) catch |err| {
                if (err == error.NotDir) {
                    // It's a file, check if it's a .bin file
                    if (std.mem.endsWith(u8, corpus_path, ".bin")) {
                        try result.append(try self.allocator.dupe(u8, corpus_path));
                        if (self.debug) {
                            std.debug.print("  Added file: {s}\n", .{corpus_path});
                        }
                    }
                } else if (self.debug) {
                    std.debug.print("  Error opening path: {s}\n", .{@errorName(err)});
                }
                continue;
            };
            defer dir.close();

            // Iterate through directory contents (only immediate files)
            var iter = dir.iterate();
            while (try iter.next()) |entry| {
                if (entry.kind == .file) {
                    if (std.mem.endsWith(u8, entry.name, ".bin")) {
                        // Construct full path
                        const full_path = try std.fs.path.join(self.allocator, &[_][]const u8{ corpus_path, entry.name });
                        try result.append(full_path);
                    }
                }
            }
        }

        const lt = struct {
            fn lessThan(_: void, a: []const u8, b: []const u8) bool {
                return std.mem.lessThan(u8, a, b);
            }
        }.lessThan;

        std.sort.pdq([]const u8, result.items, {}, lt);

        for (result.items) |item| {
            if (self.debug) {
                std.debug.print("  Added file: {s}\n", .{item});
            }
        }

        return result;
    }

    pub fn checkPhase2CorpusProcessing(self: *VocabLearner) !void {
        if (self.debug) {
            var tokens_with_occurrences: usize = 0;
            var total_occurrences: u64 = 0;

            for (self.candidate_stats) |stats| {
                if (stats.n_nonoverlapping_occurrences > 0) {
                    tokens_with_occurrences += 1;
                    total_occurrences += stats.n_nonoverlapping_occurrences;
                }
            }

            std.debug.print("Phase 2: Found {d} tokens with occurrences out of {d} candidates.\n", .{ tokens_with_occurrences, self.candidate_stats.len });

            if (tokens_with_occurrences > 0) {
                std.debug.print("         Total occurrences: {d}, avg {d:.1} per token.\n", .{ total_occurrences, @as(f64, @floatFromInt(total_occurrences)) / @as(f64, @floatFromInt(tokens_with_occurrences)) });
            }
        }
    }

    // Check if any suffix of str1 is a prefix of str2
    fn hasSuffixPrefixOverlap(str1: []const u8, str2: []const u8) bool {
        if (str1.len <= 1 or str2.len <= 1) return false; // No meaningful overlaps with empty/single char strings

        // For each position in str1, check if the suffix starting at that position
        // is a prefix of str2
        for (1..str1.len) |i| {
            const suffix_len = str1.len - i;
            // Check if this suffix matches a prefix of str2 (up to the length of the shorter of the two)
            const check_len = @min(suffix_len, str2.len);
            if (check_len <= str2.len and std.mem.eql(u8, str1[i..][0..check_len], str2[0..check_len])) {
                return true;
            }
        }

        return false;
    }

    pub fn tokenIsAlmostIndependentOfTokens(
        self: *VocabLearner,
        token_id: u32,
        accepted_current_step_tokens: []const u32,
    ) bool {
        if (accepted_current_step_tokens.len == 0) return true;

        const token_str = self.getTokenStr(token_id);

        for (accepted_current_step_tokens) |other_token_id| {
            const other_token_str = self.getTokenStr(other_token_id);

            if (std.mem.indexOf(u8, other_token_str, token_str) != null) {
                return false;
            }

            if (std.mem.indexOf(u8, token_str, other_token_str) != null) {
                return false;
            }

            if (hasSuffixPrefixOverlap(token_str, other_token_str)) {
                return false;
            }

            if (hasSuffixPrefixOverlap(other_token_str, token_str)) {
                return false;
            }
        }

        return true;
    }

    fn updateCoveredOccurrences(
        self: *VocabLearner,
        candidate_automaton: *const BakaCorasick,
    ) void {
        for (0..self.candidate_stats.len) |i_usize| {
            const i: u32 = @intCast(i_usize);
            if (self.candidate_stats[i].is_in_vocab) {
                const document = self.getTokenStr(i);
                const increment = self.candidate_stats[i].n_uses_for_cover;
                // Scan text with the automaton
                var can_state: u32 = 0;
                for (document) |byte| {
                    can_state = candidate_automaton.transitions[can_state][byte];

                    // Check if this state represents a match
                    {
                        const token_id = candidate_automaton.info[can_state].token_id;
                        if (token_id != BakaCorasick.NO_TOKEN) {
                            self.candidate_stats[token_id].n_covered_occurrences += increment;
                        }
                    }

                    // Check suffix links for additional matches
                    var suffix = candidate_automaton.info[can_state].green;
                    while (suffix != 0) {
                        const token_id = candidate_automaton.info[suffix].token_id;
                        if (token_id != BakaCorasick.NO_TOKEN) {
                            self.candidate_stats[token_id].n_covered_occurrences += increment;
                        }
                        suffix = candidate_automaton.info[suffix].green;
                    }
                }
            }
        }
    }

    fn ensureSomeCandidatesLookGoodAfterUpdatingBound(
        self: *VocabLearner,
        heap: *std.PriorityQueue(u32, *VocabLearner, buildVocabLessThan),
        tokenize_candidates_heap: *std.PriorityQueue(u32, *VocabLearner, buildVocabGreaterThan),
        n_candidates_to_tokenize: *usize,
        candidates_to_tokenize_arraylist: *std.ArrayList(u32),
        parallel_dp: *parallel.ParallelDP,
        tokenize_candidates_scratch: *std.ArrayList(u32),
        cover_step: u64,
        candidate_automaton: *BakaCorasick,
    ) !void {
        const n_to_tokenize_increase_per_repeat: usize = 500;
        const n_to_tokenize_decrease_when_no_repeats: usize = 100;
        var best_heap_est_savings = self.candidate_stats[heap.peek().?].est_total_savings;
        var repeated = false;
        while (tokenize_candidates_heap.count() < self.top_k_candidates or
            self.candidate_stats[tokenize_candidates_heap.peek().?].est_total_savings <= best_heap_est_savings)
        {
            if (tokenize_candidates_heap.count() > 0) {
                n_candidates_to_tokenize.* += n_to_tokenize_increase_per_repeat;
                repeated = true;
                try candidates_to_tokenize_arraylist.resize(n_candidates_to_tokenize.*);
            }
            var any_candidates_need_covered_update = false;
            candidate_automaton.clear();
            const candidates_to_tokenize = candidates_to_tokenize_arraylist.items;
            for (0..candidates_to_tokenize.len) |i| {
                const id = heap.remove();
                candidates_to_tokenize[i] = id;
                if (self.candidate_stats[id].covered_step != cover_step) {
                    try candidate_automaton.insert(self.getTokenStr(id), id);
                    any_candidates_need_covered_update = true;
                    self.candidate_stats[id].n_covered_occurrences = 0;
                    self.candidate_stats[id].covered_step = cover_step;
                }
            }
            if (any_candidates_need_covered_update) {
                try candidate_automaton.computeSuffixLinks();
                // TODO: maybe this shouldn't be single-threaded.
                self.updateCoveredOccurrences(candidate_automaton);
            }
            if (self.use_in_memory) {
                const loader = self.getLoader(InMemoryDataLoader);
                try parallel_dp.updateMaxSavingsByTokenizingCandidates(candidates_to_tokenize, loader);
            } else {
                const loader = self.getLoader(fineweb);
                try parallel_dp.updateMaxSavingsByTokenizingCandidates(candidates_to_tokenize, loader);
            }
            for (0..candidates_to_tokenize.len) |i| {
                try tokenize_candidates_heap.add(candidates_to_tokenize[i]);
            }
            while (tokenize_candidates_heap.count() > self.top_k_candidates) {
                const token_id = tokenize_candidates_heap.remove();
                try tokenize_candidates_scratch.append(token_id);
            }
            best_heap_est_savings = self.candidate_stats[heap.peek().?].est_total_savings;
        }
        if (!repeated) {
            n_candidates_to_tokenize.* -|= n_to_tokenize_decrease_when_no_repeats;
            n_candidates_to_tokenize.* = @max(n_candidates_to_tokenize.*, self.n_candidates_to_tokenize);
        }
        for (tokenize_candidates_scratch.items) |token_id| {
            try heap.add(token_id);
        }
        tokenize_candidates_scratch.clearRetainingCapacity();
        while (tokenize_candidates_heap.removeOrNull()) |token_id| {
            try heap.add(token_id);
        }
    }

    pub fn buildVocabulary(self: *VocabLearner) !void {
        const start_time = std.time.milliTimestamp();

        if (self.debug) {
            std.debug.print("Starting vocabulary building process...\n", .{});
        }

        var best_corpus_token_count: u64 = ~@as(u64, 0);
        //var late_lookbacks_arraylist = std.ArrayList(u64).init(self.allocator);
        //var late_dp_solution_arraylist = std.ArrayList(u32).init(self.allocator);

        var heap = std.PriorityQueue(u32, *VocabLearner, buildVocabLessThan).init(self.allocator, self);
        defer heap.deinit();
        try heap.ensureTotalCapacity(self.n_token_ids);
        var this_step_heap = std.PriorityQueue(u32, *VocabLearner, buildVocabLessThan).init(self.allocator, self);
        defer this_step_heap.deinit();
        try this_step_heap.ensureTotalCapacity(self.n_token_ids);
        var tokenize_candidates_heap = std.PriorityQueue(u32, *VocabLearner, buildVocabGreaterThan).init(self.allocator, self);
        defer tokenize_candidates_heap.deinit();
        var tokenize_candidates_scratch = std.ArrayList(u32).init(self.allocator);
        defer tokenize_candidates_scratch.deinit();
        // add all the candidate tokens that are not part of vocabulary
        for (0..self.n_token_ids) |id_usize| {
            const id: u32 = @intCast(id_usize);
            if (!self.candidate_stats[id].is_in_vocab) {
                try heap.add(id);
            }
        }

        const rejected_current_step_tokens = try self.allocator.alloc(u32, 1);
        const accepted_current_step_tokens = try self.allocator.alloc(u32, 10);
        const top_k_candidates = try self.allocator.alloc(SampleStats, self.top_k_candidates);

        var candidate_automaton = try BakaCorasick.init(self.allocator);
        defer candidate_automaton.deinit();

        var parallel_dp = try parallel.ParallelDP.init(self.allocator, self, self.debug);
        defer parallel_dp.deinit();

        var candidates_to_tokenize_arraylist = std.ArrayList(u32).init(self.allocator);
        defer candidates_to_tokenize_arraylist.deinit();
        var n_candidates_to_tokenize: usize = self.n_candidates_to_tokenize;
        try candidates_to_tokenize_arraylist.resize(n_candidates_to_tokenize);

        var best_other_savings: f64 = 1e99;

        var cover_step: u64 = 0;

        while (true) {
            {
                const now_cover_step = self.vocab_size / COVER_INTERVAL;
                var token_count: u64 = 0;
                if (now_cover_step > cover_step or self.vocab_size == self.max_vocab_size) {
                    token_count = if (self.use_in_memory) blk: {
                        const loader = self.getLoader(InMemoryDataLoader);
                        break :blk try parallel_dp.getCorpusTokenCount(loader);
                    } else blk: {
                        const loader = self.getLoader(fineweb);
                        break :blk try parallel_dp.getCorpusTokenCount(loader);
                    };
                    cover_step = now_cover_step;
                }
                if (self.vocab_size == self.max_vocab_size) {
                    if (token_count < best_corpus_token_count) {
                        best_corpus_token_count = token_count;
                        std.debug.print("New best corpus token count: {d}\n", .{token_count});
                        try self.saveVocabularyNow();
                    }
                    try self.deleteSomeTokens(&heap);
                }
            }

            const max_acceptable = @min(accepted_current_step_tokens.len, self.max_vocab_size - self.vocab_size);
            //const max_acceptable: usize = 1;
            const iteration_start = std.time.milliTimestamp();
            self.current_step += 1;

            if (self.debug) {
                std.debug.print("\n--- Iteration {d}: Vocabulary size {d}/{d} ---\n", .{ self.current_step, self.vocab_size, self.max_vocab_size });
            }

            var n_candidates: usize = 0;
            //var cleanup_mode = false;

            // At the start of buildVocabulary loop
            while (true) {
                // Extract candidates that haven't been evaluated yet
                //if (!cleanup_mode) {
                for (n_candidates..self.top_k_candidates) |i| {
                    const token_id = heap.remove();
                    top_k_candidates[i] = .{ .token_id = token_id };
                }
                n_candidates = self.top_k_candidates;
                //}

                const automaton_start_time = std.time.milliTimestamp();
                if (self.debug) {
                    std.debug.print("heap.items.len={}, this_step_heap.items.len={}\n", .{ heap.items.len, this_step_heap.items.len });
                    std.debug.print("Creating candidate automaton with just candidate tokens...\n", .{});
                }

                candidate_automaton.clear();
                for (top_k_candidates[0..n_candidates], 0..) |stats, my_idx_usize| {
                    const my_idx: u32 = @intCast(my_idx_usize);
                    const token_id = stats.token_id;
                    const token_str = self.getTokenStr(token_id);
                    try candidate_automaton.insert(token_str, my_idx);
                }

                try candidate_automaton.computeSuffixLinks();

                if (self.debug) {
                    const automaton_elapsed_seconds = @as(f64, @floatFromInt(std.time.milliTimestamp() - automaton_start_time)) / 1000.0;
                    std.debug.print("Candidate automaton created with {d} states in {d:.2}s\n", .{ candidate_automaton.len, automaton_elapsed_seconds });
                    std.debug.print("Added {d} candidate tokens to automaton\n", .{n_candidates});
                }

                if (self.use_in_memory) {
                    const loader = self.getLoader(InMemoryDataLoader);
                    try parallel_dp.processDocuments(loader, self.sample_size, top_k_candidates[0..n_candidates], &candidate_automaton);
                } else {
                    const loader = self.getLoader(fineweb);
                    try parallel_dp.processDocuments(loader, self.sample_size, top_k_candidates[0..n_candidates], &candidate_automaton);
                }

                var write_cursor: usize = 0;
                var best_leftover_savings: f64 = -420;
                for (0..n_candidates) |idx| {
                    const sample_stats = top_k_candidates[idx];
                    const token_id = sample_stats.token_id;
                    const sampled_occurrences = sample_stats.sampled_occurrences;

                    // Only update estimates if we have enough data
                    if (sampled_occurrences >= 5) {
                        const sampled_savings = sample_stats.sampled_savings;
                        const total_occurrences = self.candidate_stats[token_id].n_nonoverlapping_occurrences;
                        const est_savings = @as(f64, @floatFromInt(sampled_savings)) * @as(f64, @floatFromInt(total_occurrences)) / @as(f64, @floatFromInt(sampled_occurrences));
                        const token_count = self.candidate_stats[token_id].len_in_tokens;
                        const max_est_savings: f64 = @floatFromInt(total_occurrences * (token_count - 1));
                        if (est_savings < max_est_savings) {
                            self.candidate_stats[token_id].sampled_occurrences = sampled_occurrences;
                            self.candidate_stats[token_id].sampled_savings = sampled_savings;
                            self.candidate_stats[token_id].est_total_savings = est_savings;
                        }
                        self.candidate_stats[token_id].sampled_step = self.current_step;
                        try this_step_heap.add(token_id);
                    } else {
                        const est_savings = self.candidate_stats[token_id].est_total_savings;
                        best_leftover_savings = @max(best_leftover_savings, est_savings);
                        top_k_candidates[write_cursor] = sample_stats;
                        write_cursor += 1;
                    }
                }
                n_candidates = write_cursor;

                // if this_step_heap has some stuff,
                // and the best one is better than everything else,
                // then we can stop.
                const apparent_best_unsampled_token_id = heap.peek().?;
                best_other_savings = @max(best_leftover_savings, self.candidate_stats[apparent_best_unsampled_token_id].est_total_savings);
                if (this_step_heap.peek()) |elem| {
                    const my_savings = self.candidate_stats[elem].est_total_savings;
                    if (self.debug) {
                        std.debug.print("My savings: {d}, best_other_savings: {d}\n", .{ my_savings, best_other_savings });
                    }
                    if (self.candidate_stats[elem].est_total_savings >= best_other_savings) {
                        break;
                    }
                }
                try self.ensureSomeCandidatesLookGoodAfterUpdatingBound(
                    &heap,
                    &tokenize_candidates_heap,
                    &n_candidates_to_tokenize,
                    &candidates_to_tokenize_arraylist,
                    &parallel_dp,
                    &tokenize_candidates_scratch,
                    cover_step,
                    &candidate_automaton,
                );
            }

            var n_accepted: usize = 0;
            var n_rejected: usize = 0;
            while (n_accepted < max_acceptable and n_rejected < rejected_current_step_tokens.len) {
                if (this_step_heap.peek()) |apparent_best_token_id| {
                    _ = this_step_heap.remove();
                    if (self.candidate_stats[apparent_best_token_id].est_total_savings >= best_other_savings and
                        self.tokenIsAlmostIndependentOfTokens(apparent_best_token_id, accepted_current_step_tokens[0..n_accepted]))
                    {
                        accepted_current_step_tokens[n_accepted] = apparent_best_token_id;
                        n_accepted += 1;
                    } else {
                        rejected_current_step_tokens[n_rejected] = apparent_best_token_id;
                        n_rejected += 1;
                    }
                } else {
                    break;
                }
            }

            for (rejected_current_step_tokens[0..n_rejected]) |token_id| {
                try heap.add(token_id);
            }
            while (this_step_heap.removeOrNull()) |token_id| {
                try heap.add(token_id);
            }
            for (top_k_candidates[0..n_candidates]) |stats| {
                try heap.add(stats.token_id);
            }

            // 5. Add selected tokens to vocabulary
            for (accepted_current_step_tokens[0..n_accepted]) |token_id| {
                const token_str = self.getTokenStr(token_id);
                self.addToVocab(token_id);
                try self.vocab_automaton.insert(token_str, token_id);

                if (self.debug) {
                    std.debug.print("  Added token {d}: \"", .{self.vocab_size});
                    for (token_str) |byte| {
                        if (byte >= 32 and byte < 127) {
                            std.debug.print("{c}", .{byte});
                        } else {
                            std.debug.print("\\x{x:0>2}", .{byte});
                        }
                    }
                    std.debug.print("\" (length: {d}, savings: {d:.2})\n", .{ token_str.len, self.candidate_stats[token_id].est_total_savings });
                }
            }

            try self.vocab_automaton.computeSuffixLinks();

            try self.ensureSomeCandidatesLookGoodAfterUpdatingBound(
                &heap,
                &tokenize_candidates_heap,
                &n_candidates_to_tokenize,
                &candidates_to_tokenize_arraylist,
                &parallel_dp,
                &tokenize_candidates_scratch,
                cover_step,
                &candidate_automaton,
            );

            const iteration_elapsed = std.time.milliTimestamp() - iteration_start;
            if (self.debug) {
                std.debug.print("Iteration {d} completed in {d}ms. Vocabulary size: {d}\n", .{ self.current_step, iteration_elapsed, self.vocab_size });
            }
        }

        const elapsed_sec = @as(f64, @floatFromInt(std.time.milliTimestamp() - start_time)) / 1000.0;
        if (self.debug) {
            std.debug.print("\nVocabulary building completed in {d:.2}s. Final vocabulary size: {d}\n", .{ elapsed_sec, self.vocab_size });
        }
    }

    // Greedy tokenization algorithm
    fn tokenizeGreedy(self: *VocabLearner, automaton: *BakaCorasick, text: []const u8) !ArrayList(u32) {
        var tokens = ArrayList(u32).init(self.allocator);
        errdefer tokens.deinit();

        if (self.debug) {
            std.debug.print("Tokenizing with automaton of {d} states. Text length: {d}\n", .{ automaton.len, text.len });
        }

        var pos = text.len;
        var multi_byte_matches: usize = 0;

        while (pos > 0) {
            var best_len: usize = 0;
            var best_token_id: u32 = 0;

            // Find the longest token that matches at the current position
            var state: u32 = 0;
            var longest_match_pos: usize = 0;
            var longest_match_len: usize = 0;
            var longest_match_id: u32 = 0;

            // Scan backwards to find all possible matches ending at pos
            for (0..@min(32, pos)) |i| { // Limit search depth for efficiency
                const scan_pos = pos - i - 1;
                const byte = text[scan_pos];

                // Get next state
                state = automaton.transitions[state][byte];

                // Check if this state represents a match
                if (automaton.info[state].token_id != BakaCorasick.NO_TOKEN) {
                    const token_id = automaton.info[state].token_id;
                    const token_len = automaton.info[state].depth;

                    // Only consider tokens that fit exactly at the current position
                    if (scan_pos + token_len == pos) {
                        if (token_len > longest_match_len) {
                            longest_match_pos = scan_pos;
                            longest_match_len = token_len;
                            longest_match_id = token_id;
                        }
                    }
                }

                // If we've reached a zero state, no need to continue
                if (state == 0) break;
            }

            // Use the longest match found
            if (longest_match_len > 0) {
                best_len = longest_match_len;
                best_token_id = longest_match_id;

                if (best_len > 1) {
                    multi_byte_matches += 1;
                }
            } else {
                // If no match found, use single byte token
                best_len = 1;
                best_token_id = text[pos - 1];
            }

            // Add token and move position
            try tokens.append(best_token_id);
            pos -= best_len;
        }

        if (self.debug) {
            std.debug.print("Tokenization complete. Total tokens: {d}, Multi-byte matches: {d}\n", .{ tokens.items.len, multi_byte_matches });
        }

        // Reverse tokens (since we tokenized from end to start)
        std.mem.reverse(u32, tokens.items);
        return tokens;
    }

    pub fn checkPhase3MainLoop(self: *VocabLearner) !void {
        if (self.debug) {
            // Count tokens by length
            var single_byte_count: usize = 0;
            var multi_byte_count: usize = 0;

            for (self.candidate_stats) |stats| {
                if (stats.is_in_vocab) {
                    if (stats.str_len == 1) {
                        single_byte_count += 1;
                    } else {
                        multi_byte_count += 1;
                    }
                }
            }

            std.debug.print("Phase 3: Final vocabulary has {d} tokens ({d} single-byte, {d} multi-byte).\n", .{ self.vocab_size, single_byte_count, multi_byte_count });

            const mb_percentage = @as(f64, @floatFromInt(multi_byte_count)) /
                @as(f64, @floatFromInt(self.vocab_size)) * 100.0;

            std.debug.print("         Multi-byte tokens: {d:.1}% of vocabulary.\n", .{mb_percentage});
            std.debug.print("         Candidate tokens remaining: {d}\n", .{self.candidate_stats.len});
        }
    }

    pub fn deleteSomeTokens(
        self: *VocabLearner,
        heap: *std.PriorityQueue(u32, *VocabLearner, buildVocabLessThan),
    ) !void {
        const start_time = std.time.milliTimestamp();

        if (self.debug) {
            std.debug.print("\n=== Starting token deletion ===\n", .{});
        }

        // 1. Collect all non-essential tokens (tokens with ID >= 256)
        var vocab_tokens = std.ArrayList(u32).init(self.allocator);
        defer vocab_tokens.deinit();

        for (0..self.n_token_ids) |id_usize| {
            const id: u32 = @intCast(id_usize);
            if (id >= 256 and self.candidate_stats[id].is_in_vocab) {
                try vocab_tokens.append(id);
            }
        }

        const batch_size = (vocab_tokens.items.len + 2) / 3;

        const available_tokens = vocab_tokens.items.len;
        if (available_tokens < batch_size) {
            if (self.debug) {
                std.debug.print("Not enough non-essential tokens for replacement: {d} available, {d} needed per batch\n", .{ available_tokens, batch_size });
            }
            return;
        }

        // 2. Shuffle the tokens
        var prng = std.Random.DefaultPrng.init(@intCast(std.time.milliTimestamp()));
        const random = prng.random();

        {
            var i = vocab_tokens.items.len - 1;
            while (i > 0) : (i -= 1) {
                const j = random.uintLessThan(usize, i + 1); // Random index from 0 to i inclusive
                const temp = vocab_tokens.items[i];
                vocab_tokens.items[i] = vocab_tokens.items[j];
                vocab_tokens.items[j] = temp;
            }
        }

        // 3. delete the first batch_size tokens
        for (0..batch_size) |i| {
            const token_id = vocab_tokens.items[i];
            self.removeFromVocab(token_id);
        }

        // Rebuild the automaton without deleted tokens
        self.vocab_automaton.clear();
        for (0..self.n_token_ids) |id_usize| {
            const id: u32 = @intCast(id_usize);
            const stats = self.candidate_stats[id];
            if (stats.is_in_vocab and stats.str_len > 1) {
                const token_str = self.getTokenStr(id);
                try self.vocab_automaton.insert(token_str, id);
            }
        }
        try self.vocab_automaton.computeSuffixLinks();

        self.resetTokenEstimates();
        heap.clearRetainingCapacity();
        for (0..self.n_token_ids) |id_usize| {
            const id: u32 = @intCast(id_usize);
            if (!self.candidate_stats[id].is_in_vocab) {
                try heap.add(id);
            }
        }

        const elapsed_ms = std.time.milliTimestamp() - start_time;
        if (self.debug) {
            std.debug.print("Completed deleting some tokens in {d}ms.\n", .{elapsed_ms});
        }
    }

    pub fn saveVocabularyNow(self: *VocabLearner) !void {
        // Get current Unix time in seconds
        const current_time_seconds = std.time.timestamp();

        // Construct filename
        var filename_buffer: [64]u8 = undefined; // Large enough for "vocabulary_" + timestamp + ".bin"
        const filename = try std.fmt.bufPrint(&filename_buffer, "vocabulary_{d}.bin", .{current_time_seconds});

        // Call saveVocabulary with this filename
        try self.saveVocabulary(filename);

        if (self.debug) {
            std.debug.print("Vocabulary saved to {s}\n", .{filename});
        }
    }

    /// Save vocabulary to a binary file
    pub fn saveVocabulary(self: *VocabLearner, path: []const u8) !void {
        const file = try std.fs.cwd().createFile(path, .{});
        defer file.close();

        // Create and write header
        const header = VocabHeader{
            .magic = VOCAB_MAGIC,
            .vocab_size = @intCast(self.vocab_size),
            .reserved = [_]u8{0} ** 20,
        };
        try file.writeAll(std.mem.asBytes(&header));

        // Write each token with sequential IDs
        var sequential_id: u32 = 0;

        // First, ensure the byte tokens (0-255) are saved with their original IDs
        for (0..256) |i| {
            const token_id: u32 = @intCast(i);
            if (!self.candidate_stats[token_id].is_in_vocab) {
                // This shouldn't happen for byte tokens, but just in case
                continue;
            }

            const token_str = self.getTokenStr(token_id);
            const token_length: u32 = @intCast(token_str.len);

            // Write original token ID for byte tokens (0-255)
            var id_bytes: [4]u8 = undefined;
            var len_bytes: [4]u8 = undefined;
            std.mem.writeInt(u32, &id_bytes, sequential_id, .little);
            std.mem.writeInt(u32, &len_bytes, token_length, .little);

            try file.writeAll(&id_bytes);
            try file.writeAll(&len_bytes);
            try file.writeAll(token_str);

            sequential_id += 1;
        }

        // Then save all other tokens with sequential IDs
        for (256..self.candidate_stats.len) |i| {
            const original_id: u32 = @intCast(i);
            if (!self.candidate_stats[original_id].is_in_vocab) continue;

            const token_str = self.getTokenStr(original_id);
            const token_length: u32 = @intCast(token_str.len);

            // Write sequential token ID
            var id_bytes: [4]u8 = undefined;
            var len_bytes: [4]u8 = undefined;
            std.mem.writeInt(u32, &id_bytes, sequential_id, .little);
            std.mem.writeInt(u32, &len_bytes, token_length, .little);

            try file.writeAll(&id_bytes);
            try file.writeAll(&len_bytes);
            try file.writeAll(token_str);

            sequential_id += 1;
        }

        if (self.debug) {
            std.debug.print("Saved vocabulary with {d} tokens to {s}\n", .{ self.vocab_size, path });
        }
    }

    /// Load vocabulary from a binary file
    pub fn loadVocabularyFromFile(
        allocator: Allocator,
        path: []const u8,
        input_tokenset_path: []const u8,
        corpus_paths: []const []const u8,
        max_vocab_size: u32,
        use_in_memory: bool,
        debug: bool,
    ) !*VocabLearner {
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();

        // Read and validate header
        var header: VocabHeader = undefined;
        const bytes_read = try file.readAll(std.mem.asBytes(&header));
        if (bytes_read != @sizeOf(VocabHeader)) {
            return error.IncompleteHeader;
        }

        // Validate magic number
        if (!std.mem.eql(u8, &header.magic, &VOCAB_MAGIC)) {
            return error.InvalidMagicNumber;
        }

        // Create new VocabLearner with empty initialization
        var learner = try allocator.create(VocabLearner);
        errdefer allocator.destroy(learner);

        try initLearner(
            learner,
            allocator,
            corpus_paths,
            max_vocab_size,
            use_in_memory,
            debug,
        );

        // Reserve capacity for vocabulary
        try learner.loadCandidateTokens(input_tokenset_path);
        // Read tokens
        var token_id: u32 = 0;
        var matching_candidate_token_id: u32 = 254;
        while (token_id < header.vocab_size) {
            var id_bytes: [4]u8 = undefined;
            var len_bytes: [4]u8 = undefined;

            // Read token ID and length
            if (try file.readAll(&id_bytes) != 4 or try file.readAll(&len_bytes) != 4) {
                return error.IncompleteHeader;
            }

            const read_token_id = std.mem.readInt(u32, &id_bytes, .little);
            const token_length = std.mem.readInt(u32, &len_bytes, .little);

            // Validate token ID is sequential
            if (read_token_id != token_id) {
                return error.InvalidTokenSequence;
            }

            // Read token content
            const token = try allocator.alloc(u8, token_length);
            errdefer allocator.free(token);

            const token_bytes_read = try file.readAll(token);
            if (token_bytes_read != token_length) {
                return error.IncompleteToken;
            }

            var real_token_id = token_id;
            if (token_id >= 256) {
                matching_candidate_token_id = matching_candidate_token_id + 1;
                while (!std.mem.eql(u8, token[0..], learner.getTokenStr(matching_candidate_token_id))) {
                    matching_candidate_token_id = matching_candidate_token_id + 1;
                    if (matching_candidate_token_id >= learner.n_token_ids) {
                        std.debug.print("No matching candidate for {} '{s}'", .{ token_id, token });
                        return error.UnrecognizedToken;
                    }
                }
                real_token_id = matching_candidate_token_id;
            }

            // Add token to vocabulary
            learner.addToVocab(real_token_id);
            try learner.vocab_automaton.insert(token, real_token_id);
            token_id += 1;
        }

        // Compute suffix links for the automaton
        try learner.vocab_automaton.computeSuffixLinks();

        if (debug) {
            std.debug.print("Loaded vocabulary with {d} tokens from {s}\n", .{ learner.vocab_size, path });
        }

        return learner;
    }

    inline fn getDocumentTokenCountWithoutCountsForCover(
        self: *const VocabLearner,
        document: []const u8,
        lookbacks_arraylist: *std.ArrayList(u64),
        dp_solution_arraylist: *std.ArrayList(u32),
    ) !u64 {
        lookbacks_arraylist.clearRetainingCapacity();
        try lookbacks_arraylist.appendNTimes(0, document.len + 1);
        const lookbacks = lookbacks_arraylist.items;
        dp_solution_arraylist.clearRetainingCapacity();
        try dp_solution_arraylist.appendNTimes(0, document.len + 1);
        const dp_solution = dp_solution_arraylist.items;

        {
            // Scan text with the automata
            const vocab_automaton = &self.vocab_automaton;
            var vocab_state: u32 = 0;
            for (document, 1..) |byte, i| {
                vocab_state = vocab_automaton.transitions[vocab_state][byte];
                var this_lookback: u64 = 0;

                // Check if this state represents a match
                {
                    const token_id = vocab_automaton.info[vocab_state].token_id;
                    if (token_id != BakaCorasick.NO_TOKEN) {
                        const token_len = vocab_automaton.info[vocab_state].depth;
                        this_lookback |= @as(u64, 1) << @intCast(token_len);
                    }
                }

                // Check suffix links for additional matches
                var suffix = vocab_automaton.info[vocab_state].green;
                while (suffix != 0) {
                    const token_id = vocab_automaton.info[suffix].token_id;
                    if (token_id != BakaCorasick.NO_TOKEN) {
                        const token_len = vocab_automaton.info[suffix].depth;
                        this_lookback |= @as(u64, 1) << @intCast(token_len);
                    }
                    suffix = vocab_automaton.info[suffix].green;
                }
                this_lookback &= ~@as(u64, 3);
                lookbacks[i] = this_lookback;
            }
        }

        dp_solution[0] = 0;
        for (1..dp_solution.len) |i| {
            var entry_minus_one = dp_solution[i - 1];
            var mask = lookbacks[i];
            const n_iters = @popCount(mask);
            for (0..n_iters) |_| {
                const lookback = @ctz(mask);
                mask &= mask - 1;
                entry_minus_one = @min(entry_minus_one, dp_solution[i - lookback]);
            }
            dp_solution[i] = entry_minus_one + 1;
        }
        const baseline_cost = dp_solution[dp_solution.len - 1];
        return baseline_cost;
    }

    inline fn getDocumentTokenCountWithCountsForCover(
        self: *const VocabLearner,
        document: []const u8,
        pl_arraylist: *std.ArrayList(u64),
        token_id_arraylist: *std.ArrayList(u32),
        token_lens_arraylist: *std.ArrayList(u32),
        counts_for_cover: []u64,
    ) !u64 {
        pl_arraylist.clearRetainingCapacity();
        try pl_arraylist.appendNTimes(~@as(u64, 0), document.len + 1);
        const pl = pl_arraylist.items;
        token_id_arraylist.clearRetainingCapacity();
        try token_id_arraylist.appendNTimes(0, document.len + 1);
        const token_ids = token_id_arraylist.items;
        token_lens_arraylist.clearRetainingCapacity();
        try token_lens_arraylist.appendNTimes(0, document.len + 1);
        const token_lens = token_lens_arraylist.items;
        pl[0] = 0;

        {
            // Scan text with the automata
            const vocab_automaton = &self.vocab_automaton;
            var vocab_state: u32 = 0;
            for (document, 1..) |byte, i| {
                vocab_state = vocab_automaton.transitions[vocab_state][byte];

                // Check if this state represents a match
                {
                    const token_id = vocab_automaton.info[vocab_state].token_id;
                    if (token_id != BakaCorasick.NO_TOKEN) {
                        const token_len = vocab_automaton.info[vocab_state].depth;
                        const potential_cost = pl[i - token_len] + 1;
                        if (potential_cost < pl[i]) {
                            pl[i] = potential_cost;
                            token_ids[i] = token_id;
                            token_lens[i] = token_len;
                        }
                    }
                }

                // Check suffix links for additional matches
                var suffix = vocab_automaton.info[vocab_state].green;
                while (suffix != 0) {
                    const token_id = vocab_automaton.info[suffix].token_id;
                    if (token_id != BakaCorasick.NO_TOKEN) {
                        const token_len = vocab_automaton.info[suffix].depth;
                        const potential_cost = pl[i - token_len] + 1;
                        if (potential_cost < pl[i]) {
                            pl[i] = potential_cost;
                            token_ids[i] = token_id;
                            token_lens[i] = token_len;
                        }
                    }
                    suffix = vocab_automaton.info[suffix].green;
                }
            }
        }

        const baseline_cost = pl[pl.len - 1];
        var idx = pl.len - 1;
        while (idx > 0) {
            const token_id = token_ids[idx];
            counts_for_cover[token_id] += 1;
            idx -= token_lens[idx];
        }
        return baseline_cost;
    }

    pub fn getDocumentTokenCount(
        self: *const VocabLearner,
        document: []const u8,
        lookbacks_arraylist: *std.ArrayList(u64),
        dp_solution_arraylist: *std.ArrayList(u32),
        token_lens_arraylist: *std.ArrayList(u32),
        counts_for_cover: anytype,
    ) !u64 {
        comptime {
            if (@TypeOf(counts_for_cover) != void and @TypeOf(counts_for_cover) != []u64) {
                @compileError("counts_for_cover must be void or []u64");
            }
        }
        const record_counts = @TypeOf(counts_for_cover) == []u64;
        if (record_counts) {
            return self.getDocumentTokenCountWithCountsForCover(document, lookbacks_arraylist, dp_solution_arraylist, token_lens_arraylist, counts_for_cover);
        } else {
            return self.getDocumentTokenCountWithoutCountsForCover(document, lookbacks_arraylist, dp_solution_arraylist);
        }
    }

    pub fn evaluateCandidatesOnDocumentDP(
        self: *const VocabLearner,
        candidates: []SampleStats,
        candidates_automaton: *const BakaCorasick,
        document: []const u8,
        lookbacks_arraylist: *std.ArrayList(u64),
        dp_solution_arraylist: *std.ArrayList(u32),
        matches_arraylist: *std.ArrayList(MatchInfo),
        token_idx_to_least_end_pos: []u32,
    ) !void {
        lookbacks_arraylist.clearRetainingCapacity();
        try lookbacks_arraylist.appendNTimes(0, document.len + 1);
        const lookbacks = lookbacks_arraylist.items;
        dp_solution_arraylist.clearRetainingCapacity();
        try dp_solution_arraylist.appendNTimes(0, document.len + 1);
        const dp_solution = dp_solution_arraylist.items;
        matches_arraylist.clearRetainingCapacity();

        {
            // Scan text with the automata
            const vocab_automaton = &self.vocab_automaton;
            var vocab_state: u32 = 0;
            var candidates_state: u32 = 0;
            for (document, 1..) |byte, i| {
                vocab_state = vocab_automaton.transitions[vocab_state][byte];
                candidates_state = candidates_automaton.transitions[candidates_state][byte];
                var this_lookback: u64 = 0;

                // Check if this state represents a match
                {
                    const token_id = vocab_automaton.info[vocab_state].token_id;
                    if (token_id != BakaCorasick.NO_TOKEN) {
                        const token_len = vocab_automaton.info[vocab_state].depth;
                        this_lookback |= @as(u64, 1) << @intCast(token_len);
                    }
                }

                // Check if this state represents a match
                {
                    const token_id = candidates_automaton.info[candidates_state].token_id;
                    if (token_id != BakaCorasick.NO_TOKEN) {
                        matches_arraylist.append(MatchInfo.init(token_id, @intCast(i))) catch unreachable;
                    }
                }

                // Check suffix links for additional matches
                var suffix = vocab_automaton.info[vocab_state].green;
                while (suffix != 0) {
                    const token_id = vocab_automaton.info[suffix].token_id;
                    if (token_id != BakaCorasick.NO_TOKEN) {
                        const token_len = vocab_automaton.info[suffix].depth;
                        this_lookback |= @as(u64, 1) << @intCast(token_len);
                    }
                    suffix = vocab_automaton.info[suffix].green;
                }
                this_lookback &= ~@as(u64, 3);
                lookbacks[i] = this_lookback;

                // Check suffix links for additional matches
                suffix = candidates_automaton.info[candidates_state].green;
                while (suffix != 0) {
                    const token_id = candidates_automaton.info[suffix].token_id;
                    if (token_id != BakaCorasick.NO_TOKEN) {
                        matches_arraylist.append(MatchInfo.init(token_id, @intCast(i))) catch unreachable;
                    }
                    suffix = candidates_automaton.info[suffix].green;
                }
            }
        }

        if (matches_arraylist.items.len == 0) {
            return;
        }

        dp_solution[0] = 0;
        for (1..dp_solution.len) |i| {
            var entry_minus_one = dp_solution[i - 1];
            var mask = lookbacks[i];
            const n_iters = @popCount(mask);
            for (0..n_iters) |_| {
                const lookback = @ctz(mask);
                mask &= mask - 1;
                entry_minus_one = @min(entry_minus_one, dp_solution[i - lookback]);
            }
            dp_solution[i] = entry_minus_one + 1;
        }
        const baseline_cost = dp_solution[dp_solution.len - 1];

        @memset(token_idx_to_least_end_pos, ~@as(u32, 0));
        for (matches_arraylist.items) |match| {
            const token_id = match.getTokenId();
            const current = token_idx_to_least_end_pos[token_id];
            token_idx_to_least_end_pos[token_id] = @min(current, match.getEndPos());
        }

        const lt = struct {
            fn lessThan(ctx: []u32, a: MatchInfo, b: MatchInfo) bool {
                const a_id = a.getTokenId();
                const b_id = b.getTokenId();
                if (a_id != b_id) {
                    const end_pos_a = ctx[a_id];
                    const end_pos_b = ctx[b_id];
                    if (end_pos_a != end_pos_b) {
                        return std.math.order(end_pos_b, end_pos_a) == .lt;
                    }
                }
                return std.math.order(a.bits, b.bits) == .lt;
            }
        }.lessThan;
        std.sort.pdq(MatchInfo, matches_arraylist.items, token_idx_to_least_end_pos, lt);

        const NOT_A_TOKEN_ID = ~@as(u32, 0);
        try matches_arraylist.append(MatchInfo.init(NOT_A_TOKEN_ID, 0));
        const matches = matches_arraylist.items;
        var current_candidate_id: u32 = matches[0].getTokenId();
        var current_candidate_start_match_idx: usize = 0;
        //var current_candidate_end_match_idx: usize = 0;
        var current_candidate_start_doc_idx: usize = matches[0].getEndPos();
        var current_candidate_nonoverlapping_count: u64 = 1;
        var current_candidate_global_token_id: u32 = candidates[current_candidate_id].token_id;
        var current_candidate_len: u32 = self.candidate_stats[current_candidate_global_token_id].str_len;
        var current_candidate_next_nonoverlapping_pos: usize = current_candidate_start_doc_idx + current_candidate_len;
        for (1..matches.len) |match_idx_| {
            const match = matches[match_idx_];
            const new_token_id = match.getTokenId();
            if (new_token_id != current_candidate_id) {
                var i = current_candidate_start_doc_idx;
                // solve the dp problem starting from i
                var match_idx = current_candidate_start_match_idx;
                while (i < dp_solution.len) : (i += 1) {
                    var entry_minus_one = dp_solution[i - 1];
                    var mask = lookbacks[i];
                    const n_iters = @popCount(mask);
                    for (0..n_iters) |_| {
                        const lookback = @ctz(mask);
                        mask &= mask - 1;
                        entry_minus_one = @min(entry_minus_one, dp_solution[i - lookback]);
                    }
                    if ( //match_idx < current_candidate_end_match_idx and
                    matches[match_idx].getEndPos() == i) {
                        if (i >= current_candidate_len) {
                            entry_minus_one = @min(entry_minus_one, dp_solution[i - current_candidate_len]);
                        }
                        match_idx += 1;
                    }
                    dp_solution[i] = entry_minus_one + 1;
                }
                const savings = baseline_cost - dp_solution[dp_solution.len - 1];
                // if (self.debug) {
                //     std.debug.print("dp_solution[dp_solution.len-1]={}, baseline_cost={}, len={}, savings={}\n",
                //    .{ dp_solution[dp_solution.len-1], baseline_cost, document.len, savings });
                // }
                candidates[current_candidate_id].sampled_savings += savings;
                candidates[current_candidate_id].sampled_occurrences += current_candidate_nonoverlapping_count;

                if (new_token_id == NOT_A_TOKEN_ID) {
                    break;
                }

                // done with this candidate
                current_candidate_id = new_token_id;
                current_candidate_start_match_idx = match_idx;
                current_candidate_start_doc_idx = match.getEndPos();
                current_candidate_nonoverlapping_count = 1;
                current_candidate_global_token_id = candidates[current_candidate_id].token_id;
                current_candidate_len = self.candidate_stats[current_candidate_global_token_id].str_len;
                current_candidate_next_nonoverlapping_pos = current_candidate_start_doc_idx + current_candidate_len;
            } else {
                const pos = match.getEndPos();
                if (pos >= current_candidate_next_nonoverlapping_pos) {
                    current_candidate_nonoverlapping_count += 1;
                    current_candidate_next_nonoverlapping_pos = pos + current_candidate_len;
                }
            }
        }
    }

    // Reset token estimates to maximum values without expensive recalculation
    fn resetTokenEstimates(self: *VocabLearner) void {
        // Set all non-vocab token estimates to maximum possible value based on length and occurrence
        for (0..self.n_token_ids) |id_usize| {
            const id: u32 = @intCast(id_usize);
            if (!self.candidate_stats[id].is_in_vocab) {
                // Maximum possible savings is if every occurrence saves (length-1) bytes
                const token_len = self.candidate_stats[id].str_len;
                const occurrence_count = self.candidate_stats[id].n_nonoverlapping_occurrences;

                self.candidate_stats[id].est_total_savings = @floatFromInt(occurrence_count * (token_len - 1));
            }
        }
    }

    fn hashFile(file_content: []const u8) u64 {
        var hasher = std.crypto.hash.Sha1.init(.{});

        // Read and hash file content
        hasher.update(file_content);

        var hash_result: [20]u8 = undefined;
        hasher.final(&hash_result);

        return std.mem.readInt(u64, hash_result[0..8], .little);
    }

    pub fn saveCorpusStatistics(self: *VocabLearner, path: []const u8) !void {
        const start_time = std.time.milliTimestamp();

        if (self.debug) {
            std.debug.print("Saving corpus statistics to {s}...\n", .{path});
        }

        // Find all corpus files
        var corpus_files = try self.collectBinFiles();
        defer self.cleanupBinFiles(&corpus_files);

        const file = try std.fs.cwd().createFile(path, .{});
        defer file.close();

        // Create and write header
        const header = StatsHeader{
            .magic = STATS_MAGIC,
            .pad_a = [_]u8{0},
            .pad_b = [_]u8{0} ** 4,
            .vocab_size = @intCast(self.vocab_size),
            .n_token_ids = self.n_token_ids,
            .timestamp = std.time.milliTimestamp(),
            .file_count = @intCast(corpus_files.items.len),
            .reserved = [_]u8{0} ** 32,
        };

        try file.writeAll(std.mem.asBytes(&header));

        // Write file identifiers (hashes of file paths and sizes)
        for (corpus_files.items) |corpus_file| {
            const file_content = std.fs.cwd().readFileAlloc(self.allocator, corpus_file, std.math.maxInt(usize)) catch |err| {
                if (self.debug) {
                    std.debug.print("Error reading file {s}: {s}\n", .{ corpus_file, @errorName(err) });
                }
                return;
            };
            defer self.allocator.free(file_content);

            const hash = hashFile(file_content);
            var hash_bytes: [8]u8 = undefined;
            std.mem.writeInt(u64, &hash_bytes, hash, .little);
            try file.writeAll(&hash_bytes);
        }

        // Write token statistics
        for (0..self.n_token_ids) |id_usize| {
            const id: u32 = @intCast(id_usize);
            const stats = self.candidate_stats[id];

            // Write token ID
            var id_bytes: [4]u8 = undefined;
            std.mem.writeInt(u32, &id_bytes, id, .little);
            try file.writeAll(&id_bytes);

            // Write occurrence count
            var occur_bytes: [8]u8 = undefined;
            std.mem.writeInt(u64, &occur_bytes, stats.n_nonoverlapping_occurrences, .little);
            try file.writeAll(&occur_bytes);

            // Write estimated savings
            var savings_bytes: [8]u8 = undefined;
            const savings_bits: u64 = @bitCast(stats.est_total_savings);
            std.mem.writeInt(u64, &savings_bytes, savings_bits, .little);
            try file.writeAll(&savings_bytes);

            // Write sampled occurrences
            var sampled_occur_bytes: [8]u8 = undefined;
            std.mem.writeInt(u64, &sampled_occur_bytes, stats.sampled_occurrences, .little);
            try file.writeAll(&sampled_occur_bytes);

            // Write sampled savings
            var sampled_savings_bytes: [8]u8 = undefined;
            std.mem.writeInt(u64, &sampled_savings_bytes, stats.sampled_savings, .little);
            try file.writeAll(&sampled_savings_bytes);

            // Write sampled step
            var step_bytes: [8]u8 = undefined;
            std.mem.writeInt(u64, &step_bytes, stats.sampled_step, .little);
            try file.writeAll(&step_bytes);
        }

        const elapsed = std.time.milliTimestamp() - start_time;
        if (self.debug) {
            std.debug.print("Saved corpus statistics for {d} tokens across {d} files in {d}ms.\n", .{ self.n_token_ids, corpus_files.items.len, elapsed });
        }
    }

    pub fn loadCorpusStatistics(self: *VocabLearner, path: []const u8) !bool {
        const start_time = std.time.milliTimestamp();

        if (self.debug) {
            std.debug.print("Attempting to load corpus statistics from {s}...\n", .{path});
        }

        // Try to open the file, return false if it doesn't exist
        const file = std.fs.cwd().openFile(path, .{}) catch |err| {
            if (err == error.FileNotFound) {
                if (self.debug) {
                    std.debug.print("No statistics file found at {s}.\n", .{path});
                }
                return false;
            }
            return err;
        };
        defer file.close();

        // Read header
        var header: StatsHeader = undefined;
        const bytes_read = try file.readAll(std.mem.asBytes(&header));
        if (bytes_read != @sizeOf(StatsHeader)) {
            if (self.debug) {
                std.debug.print("Incomplete header in statistics file.\n", .{});
            }
            return false;
        }

        // Validate magic number
        if (!std.mem.eql(u8, &header.magic, &STATS_MAGIC)) {
            if (self.debug) {
                std.debug.print("Invalid magic number in statistics file.\n", .{});
            }
            return false;
        }

        // Verify token count matches
        if (header.n_token_ids != self.n_token_ids) {
            if (self.debug) {
                std.debug.print("Token count mismatch: file has {d}, expected {d}\n", .{ header.n_token_ids, self.n_token_ids });
            }
            return false;
        }

        // Get current corpus files
        var corpus_files = try self.collectBinFiles();
        defer self.cleanupBinFiles(&corpus_files);

        // Validate file count matches
        if (header.file_count != corpus_files.items.len) {
            if (self.debug) {
                std.debug.print("Corpus file count mismatch: file has {d}, current {d}\n", .{ header.file_count, corpus_files.items.len });
            }
            return false;
        }

        // Read file hashes from stats file
        var saved_hashes = try self.allocator.alloc(u64, header.file_count);
        defer self.allocator.free(saved_hashes);

        for (0..header.file_count) |i| {
            var hash_bytes: [8]u8 = undefined;
            if (try file.readAll(&hash_bytes) != 8) return false;
            saved_hashes[i] = std.mem.readInt(u64, &hash_bytes, .little);
        }

        // Calculate hashes for current files using SHA1
        var current_hashes = try self.allocator.alloc(u64, corpus_files.items.len);
        defer self.allocator.free(current_hashes);

        for (corpus_files.items, 0..) |corpus_file, i| {
            const file_content = std.fs.cwd().readFileAlloc(self.allocator, corpus_file, std.math.maxInt(usize)) catch |err| {
                if (self.debug) {
                    std.debug.print("Error reading file {s}: {s}\n", .{ corpus_file, @errorName(err) });
                }
                return false;
            };
            defer self.allocator.free(file_content);

            current_hashes[i] = hashFile(file_content);
        }

        // Sort both hash arrays for comparison
        std.sort.pdq(u64, saved_hashes, {}, struct {
            fn compare(_: void, a: u64, b: u64) bool {
                return a < b;
            }
        }.compare);

        std.sort.pdq(u64, current_hashes, {}, struct {
            fn compare(_: void, a: u64, b: u64) bool {
                return a < b;
            }
        }.compare);

        // Check if file sets match
        for (saved_hashes, 0..) |saved, i| {
            if (saved != current_hashes[i]) {
                if (self.debug) {
                    std.debug.print("Corpus files have changed, stats file is invalid.\n", .{});
                }
                return false;
            }
        }

        // Load token statistics
        for (0..self.n_token_ids) |_| {
            var id_bytes: [4]u8 = undefined;
            var occur_bytes: [8]u8 = undefined;
            var savings_bytes: [8]u8 = undefined;
            var sampled_occur_bytes: [8]u8 = undefined;
            var sampled_savings_bytes: [8]u8 = undefined;
            var step_bytes: [8]u8 = undefined;

            // Read token ID
            if (try file.readAll(&id_bytes) != 4) return false;
            const id = std.mem.readInt(u32, &id_bytes, .little);

            // Verify ID is within range
            if (id >= self.n_token_ids) {
                if (self.debug) {
                    std.debug.print("Invalid token ID in file: {d}\n", .{id});
                }
                return false;
            }

            // Read statistics
            if (try file.readAll(&occur_bytes) != 8) return false;
            if (try file.readAll(&savings_bytes) != 8) return false;
            if (try file.readAll(&sampled_occur_bytes) != 8) return false;
            if (try file.readAll(&sampled_savings_bytes) != 8) return false;
            if (try file.readAll(&step_bytes) != 8) return false;

            // Update token statistics
            self.candidate_stats[id].n_nonoverlapping_occurrences = std.mem.readInt(u64, &occur_bytes, .little);
            const savings_bits = std.mem.readInt(u64, &savings_bytes, .little);
            self.candidate_stats[id].est_total_savings = @bitCast(savings_bits);
            self.candidate_stats[id].sampled_occurrences = std.mem.readInt(u64, &sampled_occur_bytes, .little);
            self.candidate_stats[id].sampled_savings = std.mem.readInt(u64, &sampled_savings_bytes, .little);
            self.candidate_stats[id].sampled_step = std.mem.readInt(u64, &step_bytes, .little);
        }

        const elapsed = std.time.milliTimestamp() - start_time;
        if (self.debug) {
            std.debug.print("Successfully loaded corpus statistics for {d} tokens from {d} files in {d}ms.\n", .{ self.n_token_ids, corpus_files.items.len, elapsed });
        }

        return true;
    }

    // Integration function to handle cached corpus processing
    pub fn processCorpusWithCache(self: *VocabLearner, stats_path: []const u8) !void {
        // Try to load statistics first
        const loaded = try self.loadCorpusStatistics(stats_path);

        if (loaded) {
            if (self.debug) {
                std.debug.print("Using cached corpus statistics. Skipping corpus processing.\n", .{});
            }

            try self.initializeLoader();
        } else {
            try self.processCorpus();

            // Save results for future runs
            try self.saveCorpusStatistics(stats_path);
        }
    }
};

pub fn main() !void {
    const allocator = std.heap.c_allocator;

    // Parse command line arguments
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 3) {
        std.debug.print("Usage: {s} <tokenset_file> <corpus_path> [corpus_path...] [--stats-path path] [--in-memory]\n", .{args[0]});
        return;
    }

    const tokenset_path = args[1];
    var stats_path: ?[]const u8 = null;
    var vocab_path: ?[]const u8 = null;
    var use_in_memory = false;
    var load_vocab = false;
    var corpus_paths: []const []const u8 = undefined;
    var max_vocab_size: u32 = 50000;

    var non_flag_paths = std.ArrayList([]const u8).init(allocator);
    defer non_flag_paths.deinit();

    var i: usize = 2;
    while (i < args.len) {
        if (std.mem.eql(u8, args[i], "--stats-path")) {
            if (i + 1 >= args.len) {
                std.debug.print("Error: --stats-path requires a value\n", .{});
                return;
            }
            stats_path = args[i + 1];
            i += 2;
        } else if (std.mem.eql(u8, args[i], "--in-memory")) {
            use_in_memory = true;
            i += 1;
        } else if (std.mem.eql(u8, args[i], "--load-vocab")) {
            if (i + 1 >= args.len) {
                std.debug.print("Error: --load-vocab requires a value\n", .{});
                return;
            }
            load_vocab = true;
            vocab_path = args[i + 1];
            i += 2;
        } else if (std.mem.eql(u8, args[i], "--max-vocab-size")) {
            if (i + 1 >= args.len) {
                std.debug.print("Error: --max-vocab-size requires a value\n", .{});
                return;
            }
            max_vocab_size = try std.fmt.parseInt(u32, args[i + 1], 10);
            i += 2;
        } else {
            try non_flag_paths.append(args[i]);
            i += 1;
        }
    }

    corpus_paths = non_flag_paths.items;
    const debug = true;

    var learner: *VocabLearner = undefined;
    if (vocab_path) |path| {
        learner = try VocabLearner.loadVocabularyFromFile(allocator, path, tokenset_path, corpus_paths, max_vocab_size, use_in_memory, debug);
    } else {
        learner = try VocabLearner.init(allocator, tokenset_path, corpus_paths, max_vocab_size, use_in_memory, debug);
    }
    defer learner.deinit();

    // Check if everything initialized properly
    try learner.checkPhase1Initialization();

    // Phase 2: Process corpus and calculate token occurrences
    if (stats_path) |path| {
        try learner.processCorpusWithCache(path);
    } else {
        try learner.processCorpus();
    }
    try learner.checkPhase2CorpusProcessing();

    // Phase 3: Build vocabulary through iterative selection
    try learner.buildVocabulary();
    try learner.checkPhase3MainLoop();

    try learner.saveVocabulary("vocab.bin");
}
