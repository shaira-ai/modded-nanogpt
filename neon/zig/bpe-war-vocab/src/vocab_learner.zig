const std = @import("std");
const BakaCorasick = @import("baka_corasick.zig").BakaCorasick;
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;

pub const TokenStats = struct {
    token: []const u8,
    n_nonoverlapping_occurrences: u32,
    est_n_uses: u32,
    max_gain_for_nonoverlapping_occurrences: i32,
    max_gain_for_nonoverlapping_occurrences_computed_at: u32,
    missed_gain_from_superstring_used: i32,
    missed_gain_from_superstring_used_computed_at: u32,

    pub fn init(allocator: Allocator, token_str: []const u8) !*TokenStats {
        const token_copy = try allocator.dupe(u8, token_str);
        const stats = try allocator.create(TokenStats);
        stats.* = .{
            .token = token_copy,
            .n_nonoverlapping_occurrences = 0,
            .est_n_uses = 0,
            .max_gain_for_nonoverlapping_occurrences = 0,
            .max_gain_for_nonoverlapping_occurrences_computed_at = 0,
            .missed_gain_from_superstring_used = 0,
            .missed_gain_from_superstring_used_computed_at = 0,
        };
        return stats;
    }

    pub fn deinit(self: *TokenStats, allocator: Allocator) void {
        allocator.free(self.token);
        allocator.destroy(self);
    }

    pub fn getCurrentValueBound(self: *TokenStats) i32 {
        return self.max_gain_for_nonoverlapping_occurrences - self.missed_gain_from_superstring_used;
    }
};

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

    pub fn sampleDocuments(self: *DocumentSampler, count: usize) !ArrayList([]const u8) {
        if (self.debug) {
            std.debug.print("Sampling {d} documents\n", .{count});
        }

        var documents = ArrayList([]const u8).init(self.allocator);
        errdefer {
            for (documents.items) |doc| {
                self.allocator.free(doc);
            }
            documents.deinit();
        }

        const corpus_size = self.corpus_paths.len;
        if (corpus_size == 0) return error.EmptyCorpus;

        // For each document sample requested
        for (0..count) |i| {
            _ = i;

            // Choose a random file
            const file_idx = self.prng.uintLessThan(usize, corpus_size);
            const file = try std.fs.cwd().openFile(self.corpus_paths[file_idx], .{});
            defer file.close();

            const file_size = try file.getEndPos();
            const content_size = file_size - @min(file_size, 1024);

            const max_chunk_size = 1024 * 1024;
            const chunk_size = @min(max_chunk_size, content_size);

            const max_offset = if (content_size > chunk_size) content_size - chunk_size else 0;
            const offset = if (max_offset > 0) self.prng.uintLessThan(usize, max_offset) else 0;

            try file.seekTo(1024 + offset); // Skip header + random offset

            var content = try self.allocator.alloc(u8, chunk_size);
            errdefer self.allocator.free(content);

            const bytes_read = try file.readAll(content);
            if (bytes_read < chunk_size) {
                const actual_content = try self.allocator.realloc(content, bytes_read);
                content = actual_content;
            }

            try documents.append(content);
        }

        return documents;
    }

    pub fn loadDocument(self: *DocumentSampler, path: []const u8) ![]const u8 {
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();

        const file_size = try file.getEndPos();

        // Skip the first 1024 bytes (header) if file is large enough
        if (file_size > 1024) {
            try file.seekTo(1024);
        }

        const content_size = file_size - @min(file_size, 1024);
        const content = try self.allocator.alloc(u8, content_size);
        errdefer self.allocator.free(content);

        const bytes_read = try file.readAll(content);
        if (bytes_read < content_size) {
            return self.allocator.realloc(content, bytes_read);
        }

        return content;
    }
};

const VOCAB_MAGIC = "VOCA".*;
const VOCAB_VERSION: u32 = 1;
const HEADER_SIZE = 32;

// Change from packed struct to regular struct
const VocabHeader = struct {
    magic: [4]u8,
    version: u32,
    vocab_size: u32,
    reserved: [20]u8,

    // Ensure the struct is exactly 32 bytes
    comptime {
        if (@sizeOf(VocabHeader) != HEADER_SIZE) {
            @compileError("VocabHeader size mismatch");
        }
    }
};

pub const VocabLearner = struct {
    allocator: Allocator,
    vocab: ArrayList([]const u8),
    vocab_token_ids: std.StringHashMap(u32),
    candidate_stats: std.StringHashMap(*TokenStats),
    vocab_automaton: BakaCorasick,
    eval_automaton: BakaCorasick,
    current_step: u32,
    document_sampler: *DocumentSampler,
    gpt_token_to_string: std.AutoHashMap(u16, []const u8),
    // Parameters
    max_vocab_size: u32,
    top_k_candidates: u32,
    batch_size: u32,
    sample_size: u32,

    // Tracking
    last_full_corpus_scan: u32,
    full_corpus_scan_interval: u32,
    debug: bool,

    pub fn init(allocator: Allocator, input_tokenset_path: []const u8, corpus_paths: []const []const u8, max_vocab_size: u32, debug: bool) !*VocabLearner {
        var learner = try allocator.create(VocabLearner);

        // Initialize fields
        learner.* = .{
            .allocator = allocator,
            .vocab = ArrayList([]const u8).init(allocator),
            .vocab_token_ids = std.StringHashMap(u32).init(allocator),
            .candidate_stats = std.StringHashMap(*TokenStats).init(allocator),
            .vocab_automaton = try BakaCorasick.init(allocator),
            .eval_automaton = try BakaCorasick.init(allocator),
            .current_step = 0,
            .document_sampler = try DocumentSampler.init(allocator, corpus_paths, debug),
            .gpt_token_to_string = std.AutoHashMap(u16, []const u8).init(allocator),
            .max_vocab_size = max_vocab_size,
            .top_k_candidates = 200,
            .batch_size = 10,
            .sample_size = 10000,
            .last_full_corpus_scan = 0,
            .full_corpus_scan_interval = 5000,
            .debug = debug,
        };

        // Initialize with 256 single-byte tokens
        try learner.initializeWithByteTokens();

        // Load candidate tokens from tokenset file
        try learner.loadCandidateTokens(input_tokenset_path);

        return learner;
    }

    fn deinitBakaCorasick(self: *BakaCorasick, allocator: Allocator) void {
        allocator.free(self.transitions[0..self.capacity]);
        allocator.free(self.info[0..self.capacity]);
    }

    pub fn deinit(self: *VocabLearner) void {
        // Free vocabulary tokens
        for (self.vocab.items) |token| {
            self.allocator.free(token);
        }
        self.vocab.deinit();

        // Free candidate stats (each contains its own token)
        var it = self.candidate_stats.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.*.deinit(self.allocator);
        }
        self.candidate_stats.deinit();

        // Free token IDs map
        self.vocab_token_ids.deinit();

        // Clean up BakaCorasick instances
        deinitBakaCorasick(&self.vocab_automaton, self.allocator);
        deinitBakaCorasick(&self.eval_automaton, self.allocator);

        // Free document sampler
        self.document_sampler.deinit();

        // Free self
        self.allocator.destroy(self);
    }

    // Initialize with 256 single-byte tokens
    fn initializeWithByteTokens(self: *VocabLearner) !void {
        if (self.debug) {
            std.debug.print("Initializing with 256 single-byte tokens...\n", .{});
        }

        for (0..256) |i| {
            var token = try self.allocator.alloc(u8, 1);
            token[0] = @intCast(i);

            try self.vocab.append(token);
            try self.vocab_token_ids.put(token, @intCast(i));
            try self.vocab_automaton.insert(token, @intCast(i));
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

        var header: [256]u32 = undefined;
        const header_size = 256 * @sizeOf(u32);
        const bytes_read = try file.readAll(std.mem.asBytes(&header));
        if (bytes_read != header_size) {
            return error.IncompleteHeader;
        }

        var total_tokens: usize = 0;
        for (header) |count| {
            total_tokens += count;
        }

        const estimated_unique = total_tokens / 3;
        try self.candidate_stats.ensureTotalCapacity(@as(u32, @intCast(estimated_unique)));

        if (self.debug) {
            std.debug.print("Found {d} total candidate tokens.\n", .{total_tokens});
        }

        const CHUNK_SIZE = 32 * 1024;

        var buffer = try self.allocator.alloc(u8, CHUNK_SIZE);
        defer self.allocator.free(buffer);

        var tokens_processed: usize = 0;
        var unique_tokens: usize = 0;

        for (header, 0..) |count, i| {
            const length = i + 1;
            if (count == 0) continue;

            if (self.debug) {
                std.debug.print("Processing {d} tokens of length {d}...\n", .{ count, length });
            }

            if (length == 1) {
                try file.seekBy(@as(i64, @intCast(count * length)));
                continue;
            }

            // Calculate how many tokens we can process per chunk with this length
            const tokens_per_chunk = CHUNK_SIZE / length;
            var tokens_left = count;

            while (tokens_left > 0) {
                const tokens_to_read = @min(tokens_left, tokens_per_chunk);
                const bytes_to_read = tokens_to_read * length;

                const chunk_bytes_read = try file.readAll(buffer[0..bytes_to_read]);
                if (chunk_bytes_read == 0) break;

                const complete_tokens = chunk_bytes_read / length;
                if (complete_tokens == 0) break;

                for (0..complete_tokens) |token_idx| {
                    const token_start = token_idx * length;
                    const token_slice = buffer[token_start .. token_start + length];

                    const stats = try TokenStats.init(self.allocator, token_slice);
                    errdefer stats.deinit(self.allocator);

                    const token_copy = stats.token;

                    try self.candidate_stats.put(token_copy, stats);
                    unique_tokens += 1;
                    tokens_processed += 1;
                }

                tokens_left -= @intCast(complete_tokens);
            }
        }

        const elapsed = std.time.milliTimestamp() - start_time;

        if (self.debug) {
            std.debug.print("Loaded {d} unique candidate tokens (from {d} total) in {d}ms.\n", .{ unique_tokens, tokens_processed, elapsed });
        }
    }

    pub fn checkPhase1Initialization(self: *VocabLearner) !void {
        // Validation checks
        if (self.vocab.items.len != 256) {
            return error.VocabularyInitializationFailed;
        }
        if (self.candidate_stats.count() == 0) {
            return error.NoCandidateTokensLoaded;
        }
        if (self.vocab_automaton.len < 257) {
            return error.AutomatonInitializationFailed;
        }

        if (self.debug) {
            std.debug.print("Phase 1: Initialized with {d} byte tokens, {d} candidate tokens loaded.\n", .{ self.vocab.items.len, self.candidate_stats.count() });
        }
    }

    pub fn loadGPTVocabulary(self: *VocabLearner, vocab_path: []const u8) !void {
        const start_time = std.time.milliTimestamp();

        if (self.debug) {
            std.debug.print("Loading GPT vocabulary from {s}...\n", .{vocab_path});
        }

        const file = try std.fs.cwd().openFile(vocab_path, .{});
        defer file.close();

        const file_size = try file.getEndPos();
        const json_buffer = try self.allocator.alloc(u8, file_size);
        defer self.allocator.free(json_buffer);

        const bytes_read = try file.readAll(json_buffer);
        var parsed = try std.json.parseFromSlice(std.json.Value, self.allocator, json_buffer[0..bytes_read], .{});
        defer parsed.deinit();

        const root = parsed.value;
        if (root != .object) {
            return error.InvalidJsonFormat;
        }

        // Clear any existing entries
        self.gpt_token_to_string.clearRetainingCapacity();

        // Add mapping for each token
        var it = root.object.iterator();
        while (it.next()) |entry| {
            const token_str = entry.key_ptr.*;
            const token_id = @as(u16, @intCast(entry.value_ptr.integer));

            // Allocate and store the string
            const str_copy = try self.allocator.dupe(u8, token_str);
            errdefer self.allocator.free(str_copy);

            try self.gpt_token_to_string.put(token_id, str_copy);
        }

        if (self.debug) {
            std.debug.print("Loaded {d} GPT tokens from vocabulary\n", .{self.gpt_token_to_string.count()});
        }

        const elapsed_ms = std.time.milliTimestamp() - start_time;
        if (self.debug) {
            std.debug.print("Loaded GPT vocabulary in {d}ms\n", .{elapsed_ms});
        }
    }

    // Add this function to directly count token ID occurrences
    pub fn processCorpusWithTokenIDs(self: *VocabLearner) !void {
        const start_time = std.time.milliTimestamp();

        if (self.debug) {
            std.debug.print("Processing corpus using direct token ID counting...\n", .{});
        }

        // Load vocabulary if not already loaded
        if (self.gpt_token_to_string.count() == 0) {
            try self.loadGPTVocabulary("vocab.json");
        }

        // Create reverse mapping (string -> token ID)
        var string_to_token_id = std.StringHashMap(u16).init(self.allocator);
        defer string_to_token_id.deinit();

        var vocab_it = self.gpt_token_to_string.iterator();
        while (vocab_it.next()) |entry| {
            try string_to_token_id.put(entry.value_ptr.*, entry.key_ptr.*);
        }

        // Create a set of token IDs we're interested in
        var target_token_ids = std.AutoHashMap(u16, *TokenStats).init(self.allocator);
        defer target_token_ids.deinit();

        // Map candidate tokens to their token IDs
        var candidates_it = self.candidate_stats.iterator();
        while (candidates_it.next()) |entry| {
            const token = entry.key_ptr.*;
            const stats = entry.value_ptr.*;

            if (string_to_token_id.get(token)) |token_id| {
                try target_token_ids.put(token_id, stats);
            }
        }

        if (self.debug) {
            std.debug.print("Mapped {d} candidate tokens to GPT token IDs\n", .{target_token_ids.count()});
        }

        // Process each corpus file
        for (self.document_sampler.corpus_paths, 0..) |corpus_path, file_idx| {
            const file_start_time = std.time.milliTimestamp();

            if (self.debug) {
                std.debug.print("Processing file {d}/{d}: {s}\n", .{ file_idx + 1, self.document_sampler.corpus_paths.len, corpus_path });
            }

            const file = try std.fs.cwd().openFile(corpus_path, .{});
            defer file.close();

            const file_size = try file.getEndPos();
            const data_size = file_size - @min(file_size, 1024);

            if (self.debug) {
                std.debug.print("  File size: {d:.2} MB\n", .{@as(f64, @floatFromInt(data_size)) / (1024.0 * 1024.0)});
            }

            // Skip header
            if (file_size > 1024) {
                try file.seekTo(1024);
            }

            const CHUNK_SIZE = 4 * 1024 * 1024;
            var buffer = try self.allocator.alloc(u8, CHUNK_SIZE);
            defer self.allocator.free(buffer);

            var bytes_processed: usize = 0;
            var last_progress_time = std.time.milliTimestamp();
            const progress_interval_ms = 2000;

            // Process file in chunks
            while (true) {
                const bytes_read = try file.readAll(buffer);
                if (bytes_read == 0) break;

                // Process complete token IDs (2 bytes each)
                const complete_tokens = bytes_read / 2;

                for (0..complete_tokens) |i| {
                    const token_offset = i * 2;
                    const token_id = std.mem.bytesToValue(u16, buffer[token_offset .. token_offset + 2][0..2]);

                    // If this is a token we're interested in, increment its count
                    if (target_token_ids.get(token_id)) |stats| {
                        stats.n_nonoverlapping_occurrences += 1;
                    }
                }

                bytes_processed += bytes_read;

                // Progress reporting
                const current_time = std.time.milliTimestamp();
                if (self.debug and current_time - last_progress_time >= progress_interval_ms) {
                    const mb_processed = @as(f64, @floatFromInt(bytes_processed)) / (1024.0 * 1024.0);
                    const percent_complete = @as(f64, @floatFromInt(bytes_processed)) /
                        @as(f64, @floatFromInt(data_size)) * 100.0;

                    std.debug.print("  Progress: {d:.2}/{d:.2} MB ({d:.1}%)\n", .{ mb_processed, @as(f64, @floatFromInt(data_size)) / (1024.0 * 1024.0), percent_complete });

                    last_progress_time = current_time;
                }
            }

            const file_elapsed_sec = @as(f64, @floatFromInt(std.time.milliTimestamp() - file_start_time)) / 1000.0;
            if (self.debug) {
                std.debug.print("  Completed file: {s} in {d:.2}s\n", .{ corpus_path, file_elapsed_sec });
            }
        }

        // Print token stats if debugging
        if (self.debug) {
            if (self.candidate_stats.get(" professional")) |stats| {
                std.debug.print("After processing: ' professional' occurrences: {d}\n", .{stats.n_nonoverlapping_occurrences});
            }
        }

        // Calculate total processing statistics
        const elapsed_sec = @as(f64, @floatFromInt(std.time.milliTimestamp() - start_time)) / 1000.0;
        if (self.debug) {
            std.debug.print("Completed corpus processing in {d:.2}s\n", .{elapsed_sec});
        }

        // Update initial bounds after counting occurrences
        try self.calculateInitialBounds();
    }

    // Calculate initial bounds for candidate tokens
    fn calculateInitialBounds(self: *VocabLearner) !void {
        const start_time = std.time.milliTimestamp();

        if (self.debug) {
            std.debug.print("Calculating initial bounds for candidate tokens...\n", .{});
        }

        var candidates_processed: usize = 0;
        const total_candidates = self.candidate_stats.count();

        var it = self.candidate_stats.iterator();
        while (it.next()) |entry| {
            const token = entry.key_ptr.*;
            const stats = entry.value_ptr.*;

            // Calculate token length benefit (length - 1) * occurrences
            const token_length = token.len;
            const occurrences = stats.n_nonoverlapping_occurrences;

            // Only tokens of length >= 2 provide compression benefit
            if (token_length >= 2) {
                const max_gain = @as(i32, @intCast(token_length - 1)) * @as(i32, @intCast(occurrences));
                stats.max_gain_for_nonoverlapping_occurrences = max_gain;
            } else {
                stats.max_gain_for_nonoverlapping_occurrences = 0;
            }

            // Set calculation step
            stats.max_gain_for_nonoverlapping_occurrences_computed_at = self.current_step;
            stats.missed_gain_from_superstring_used_computed_at = self.current_step;

            // Initialize other fields
            stats.est_n_uses = 0;
            // Reset missed gain
            stats.missed_gain_from_superstring_used = 0;

            // For each token in the vocabulary, check if our candidate is a substring
            for (self.vocab.items) |vocab_token| {
                // Skip single-byte tokens and tokens shorter than the candidate
                if (vocab_token.len <= 1 or vocab_token.len < token_length) continue;

                // Check if candidate is a substring of the vocabulary token
                if (std.mem.indexOf(u8, vocab_token, token) != null) {
                    // Count how many times the candidate appears in the vocab token
                    var count: u32 = 0;
                    var pos: usize = 0;

                    while (pos <= vocab_token.len - token.len) {
                        const found_pos = std.mem.indexOfPos(u8, vocab_token, pos, token);
                        if (found_pos == null) break;
                        count += 1;
                        pos = found_pos.? + token.len;
                    }

                    //TODO:
                    // Estimate how many times the vocab token will be used
                    const vocab_token_est_uses = @min(100, occurrences / 10);

                    // Calculate missed gain: for each time the vocab token is used,
                    // the candidate token will miss (count * (token_length - 1)) bytes of compression
                    const missed = @as(i32, @intCast(count * vocab_token_est_uses)) *
                        @as(i32, @intCast(token_length - 1));

                    stats.missed_gain_from_superstring_used += missed;

                    if (self.debug) {
                        std.debug.print("  Token '", .{});
                        for (token) |b| {
                            if (b >= 32 and b < 127) {
                                std.debug.print("{c}", .{b});
                            } else {
                                std.debug.print("\\x{x:0>2}", .{b});
                            }
                        }
                        std.debug.print("' is substring of '", .{});
                        for (vocab_token) |b| {
                            if (b >= 32 and b < 127) {
                                std.debug.print("{c}", .{b});
                            } else {
                                std.debug.print("\\x{x:0>2}", .{b});
                            }
                        }
                        std.debug.print("' (missed gain: {d})\n", .{missed});
                    }
                }
            }

            candidates_processed += 1;
        }

        const elapsed_sec = @as(f64, @floatFromInt(std.time.milliTimestamp() - start_time)) / 1000.0;
        if (self.debug) {
            std.debug.print("Calculated initial bounds for {d} tokens in {d:.2}s\n", .{ total_candidates, elapsed_sec });
        }
    }

    pub fn checkPhase2CorpusProcessing(self: *VocabLearner) !void {
        if (self.debug) {
            var tokens_with_occurrences: usize = 0;
            var total_occurrences: u64 = 0;

            var it = self.candidate_stats.iterator();
            while (it.next()) |entry| {
                const stats = entry.value_ptr.*;
                if (stats.n_nonoverlapping_occurrences > 0) {
                    tokens_with_occurrences += 1;
                    total_occurrences += stats.n_nonoverlapping_occurrences;
                }
            }

            std.debug.print("Phase 2: Found {d} tokens with occurrences out of {d} candidates.\n", .{ tokens_with_occurrences, self.candidate_stats.count() });

            if (tokens_with_occurrences > 0) {
                std.debug.print("         Total occurrences: {d}, avg {d:.1} per token.\n", .{ total_occurrences, @as(f64, @floatFromInt(total_occurrences)) / @as(f64, @floatFromInt(tokens_with_occurrences)) });
            }
        }
    }

    pub fn buildVocabulary(self: *VocabLearner) !void {
        const start_time = std.time.milliTimestamp();

        if (self.debug) {
            std.debug.print("Starting vocabulary building process...\n", .{});
        }

        while (self.vocab.items.len < self.max_vocab_size) {
            const iteration_start = std.time.milliTimestamp();
            self.current_step += 1;

            if (self.debug) {
                std.debug.print("\n--- Iteration {d}: Vocabulary size {d}/{d} ---\n", .{ self.current_step, self.vocab.items.len, self.max_vocab_size });
            }

            // 1. Select top candidate tokens based on current value bounds
            const top_candidates = try self.selectTopCandidates(self.top_k_candidates);
            defer top_candidates.deinit();

            // 2. Sample documents from the corpus
            const sample_docs = try self.document_sampler.sampleDocuments(self.sample_size);
            defer {
                for (sample_docs.items) |doc| {
                    self.allocator.free(doc);
                }
                sample_docs.deinit();
            }

            // 3. Evaluate candidates on the sampled documents
            try self.evaluateCandidatesOnDocuments(&top_candidates, sample_docs.items);

            // 4. Select tokens for addition (nearly-non-interdependent batch)
            const tokens_to_add = try self.selectNearlyNonInterdependentBatch(&top_candidates, self.batch_size);
            defer tokens_to_add.deinit();

            // 5. Add selected tokens to vocabulary
            try self.addTokensToVocabulary(&tokens_to_add);

            // 6. Periodically remove random tokens
            if (self.current_step % 500 == 0 and self.vocab.items.len > 300) {
                try self.removeRandomTokens(5);
            }

            // 7. Periodically perform full corpus scan
            if (self.current_step % self.full_corpus_scan_interval == 0) {
                try self.performFullCorpusScan();
                self.last_full_corpus_scan = self.current_step;
            }

            const iteration_elapsed = std.time.milliTimestamp() - iteration_start;
            if (self.debug) {
                std.debug.print("Iteration {d} completed in {d}ms. Vocabulary size: {d}\n", .{ self.current_step, iteration_elapsed, self.vocab.items.len });
            }

            if (tokens_to_add.items.len == 0) {
                if (self.debug) {
                    std.debug.print("No tokens added in this iteration. Exiting.\n", .{});
                }
                break;
            }
        }

        const elapsed_sec = @as(f64, @floatFromInt(std.time.milliTimestamp() - start_time)) / 1000.0;
        if (self.debug) {
            std.debug.print("\nVocabulary building completed in {d:.2}s. Final vocabulary size: {d}\n", .{ elapsed_sec, self.vocab.items.len });
        }
    }

    fn selectTopCandidates(self: *VocabLearner, k: usize) !ArrayList(*TokenStats) {
        const start_time = std.time.milliTimestamp();

        var candidates = ArrayList(*TokenStats).init(self.allocator);
        errdefer candidates.deinit();

        const Context = struct {
            fn lessThan(_: void, a: *TokenStats, b: *TokenStats) std.math.Order {
                const value_a = a.getCurrentValueBound();
                const value_b = b.getCurrentValueBound();
                return std.math.order(value_a, value_b);
            }
        };

        var heap = std.PriorityQueue(*TokenStats, void, Context.lessThan).init(self.allocator, {});
        defer heap.deinit();

        var it = self.candidate_stats.iterator();
        var candidates_processed: usize = 0;

        while (it.next()) |entry| {
            const stats = entry.value_ptr.*;
            const value = stats.getCurrentValueBound();

            if (heap.count() < k) {
                try heap.add(stats);
            } else {
                const worst = heap.peek() orelse unreachable;
                if (value > worst.getCurrentValueBound()) {
                    _ = heap.remove();
                    try heap.add(stats);
                }
            }

            candidates_processed += 1;
        }

        for (2..11) |len| {
            for (candidates.items) |stats| {
                if (stats.token.len == len and stats.getCurrentValueBound() > 0) {
                    std.debug.print("Len {d}: value={d}, occurrences={d}, missed_gain={d}\n", .{ len, stats.getCurrentValueBound(), stats.n_nonoverlapping_occurrences, stats.missed_gain_from_superstring_used });
                    break;
                }
            }
        }

        const heap_size = heap.count();
        try candidates.ensureTotalCapacity(heap_size);

        while (heap.count() > 0) {
            try candidates.append(heap.remove());
        }

        std.mem.reverse(*TokenStats, candidates.items);

        const elapsed_ms = std.time.milliTimestamp() - start_time;
        if (self.debug) {
            std.debug.print("Selected top {d} candidates from {d} total in {d}ms\n", .{ candidates.items.len, candidates_processed, elapsed_ms });
        }

        return candidates;
    }

    fn evaluateCandidatesOnDocuments(self: *VocabLearner, candidates: *const ArrayList(*TokenStats), documents: []const []const u8) !void {
        const start_time = std.time.milliTimestamp();

        if (self.debug) {
            std.debug.print("Evaluating {d} candidates on {d} documents (parallel)...\n", .{ candidates.items.len, documents.len });
        }

        // Reset estimated uses for all candidates
        for (candidates.items) |stats| {
            stats.est_n_uses = 0;
        }

        var candidate_automaton = try BakaCorasick.init(self.allocator);
        defer deinitBakaCorasick(&candidate_automaton, self.allocator);

        // Create a mapping from automaton token IDs to candidate stats indices
        var token_id_to_index = try self.allocator.alloc(usize, candidates.items.len);
        defer self.allocator.free(token_id_to_index);

        // Add all candidates to the automaton
        for (candidates.items, 0..) |stats, i| {
            try candidate_automaton.insert(stats.token, @intCast(i));
            token_id_to_index[i] = i;
        }

        // Compute suffix links for efficient matching
        try candidate_automaton.computeSuffixLinks();

        const num_threads = @min(10, @max(1, documents.len / 100));

        var thread_results = try self.allocator.alloc([]u32, num_threads);
        defer self.allocator.free(thread_results);

        for (0..num_threads) |i| {
            thread_results[i] = try self.allocator.alloc(u32, candidates.items.len);
            @memset(thread_results[i], 0);
        }
        defer {
            for (0..num_threads) |i| {
                self.allocator.free(thread_results[i]);
            }
        }

        var threads = try self.allocator.alloc(std.Thread, num_threads);
        defer self.allocator.free(threads);

        const ThreadContext = struct {
            automaton: *BakaCorasick,
            documents: []const []const u8,
            start_doc: usize,
            end_doc: usize,
            results: []u32,
            debug: bool,
        };

        const workerFn = struct {
            fn process(ctx: ThreadContext) void {
                for (ctx.start_doc..ctx.end_doc) |doc_idx| {
                    const document = ctx.documents[doc_idx];

                    var current_state: u32 = 0;
                    for (document) |byte| {
                        current_state = ctx.automaton.transitions[current_state][byte];

                        if (ctx.automaton.info[current_state].token_id != BakaCorasick.NO_TOKEN) {
                            const token_id = ctx.automaton.info[current_state].token_id;
                            ctx.results[token_id] += 1;
                        }

                        var suffix = ctx.automaton.info[current_state].green;
                        while (suffix != 0) {
                            if (ctx.automaton.info[suffix].token_id != BakaCorasick.NO_TOKEN) {
                                const token_id = ctx.automaton.info[suffix].token_id;
                                ctx.results[token_id] += 1;
                            }
                            suffix = ctx.automaton.info[suffix].green;
                        }
                    }
                }
            }
        }.process;

        const docs_per_thread = (documents.len + num_threads - 1) / num_threads;
        for (0..num_threads) |i| {
            const start = i * docs_per_thread;
            const end = @min(start + docs_per_thread, documents.len);

            if (start >= end) break;

            const context = ThreadContext{
                .automaton = &candidate_automaton,
                .documents = documents,
                .start_doc = start,
                .end_doc = end,
                .results = thread_results[i],
                .debug = self.debug,
            };

            threads[i] = try std.Thread.spawn(.{}, workerFn, .{context});
        }

        for (0..num_threads) |i| {
            if (i * docs_per_thread >= documents.len) break;
            threads[i].join();
        }

        for (candidates.items, 0..) |stats, idx| {
            var total_uses: u32 = 0;
            for (0..num_threads) |i| {
                total_uses += thread_results[i][idx];
            }
            stats.est_n_uses = total_uses;
        }

        const elapsed_ms = std.time.milliTimestamp() - start_time;
        if (self.debug) {
            std.debug.print("Candidate evaluation completed in {d}ms (used {d} threads)\n", .{ elapsed_ms, num_threads });
        }
    }

    fn selectNearlyNonInterdependentBatch(self: *VocabLearner, candidates: *const ArrayList(*TokenStats), batch_size: usize) !ArrayList(*TokenStats) {
        var selected = ArrayList(*TokenStats).init(self.allocator);
        errdefer selected.deinit();

        // Create dependency graph (adjacency list)
        var dependencies = std.StringHashMap(std.BufSet).init(self.allocator);
        defer {
            var it = dependencies.iterator();
            while (it.next()) |entry| {
                entry.value_ptr.deinit();
            }
            dependencies.deinit();
        }

        for (candidates.items) |stats| {
            try dependencies.put(stats.token, std.BufSet.init(self.allocator));
        }

        for (candidates.items) |stats_a| {
            for (candidates.items) |stats_b| {
                if (std.mem.eql(u8, stats_a.token, stats_b.token)) continue;

                const is_prefix = std.mem.startsWith(u8, stats_b.token, stats_a.token);
                const is_suffix = std.mem.endsWith(u8, stats_b.token, stats_a.token);
                const is_substring = std.mem.indexOf(u8, stats_b.token, stats_a.token) != null;

                if (is_prefix or is_suffix or is_substring) {
                    const dep_set = dependencies.getPtr(stats_a.token).?;
                    try dep_set.insert(stats_b.token);
                }
            }
        }

        // Greedy independent set selection
        var remaining = std.BufSet.init(self.allocator);
        defer remaining.deinit();

        // Add all candidates to the remaining set
        for (candidates.items) |stats| {
            try remaining.insert(stats.token);
        }

        // Select tokens until we reach batch_size or run out of candidates
        while (selected.items.len < batch_size and remaining.count() > 0) {
            // Find the best candidate with minimal dependencies
            var best_token: ?[]const u8 = null;
            var best_value: i32 = 0;
            var best_stats: ?*TokenStats = null;

            var it = remaining.iterator();
            while (it.next()) |token_ptr| {
                const token = token_ptr.*;

                // Find the stats for this token
                for (candidates.items) |stats| {
                    if (std.mem.eql(u8, stats.token, token)) {
                        const value = stats.getCurrentValueBound(); // Use the proper value metric

                        // Check if this is the best candidate so far
                        if (best_token == null or value > best_value) {
                            best_token = token;
                            best_value = value;
                            best_stats = stats;
                        }
                        break;
                    }
                }
            }

            if (best_token == null) break;

            // Debug: Check if we can get the dependency set
            const dep_set_opt = dependencies.get(best_token.?);
            if (dep_set_opt) |dep_set| {
                // Remove the selected token from remaining
                remaining.remove(best_token.?);

                // Add the best candidate to the selected set
                try selected.append(best_stats.?);

                // Remove dependencies from remaining
                var dep_it = dep_set.iterator();
                while (dep_it.next()) |dep_token_ptr| {
                    const dep_token = dep_token_ptr.*;
                    if (remaining.contains(dep_token)) {
                        remaining.remove(dep_token);
                    }
                }
            } else {
                // If we can't find the dependency set, skip this token
                remaining.remove(best_token.?);
            }
        }

        if (self.debug) {
            std.debug.print("Selected {d} nearly-non-interdependent tokens for addition\n", .{selected.items.len});
        }

        return selected;
    }

    // 5. Add selected tokens to vocabulary
    fn addTokensToVocabulary(self: *VocabLearner, tokens: *const ArrayList(*TokenStats)) !void {
        const start_time = std.time.milliTimestamp();

        if (tokens.items.len == 0) return;

        var sorted_tokens = ArrayList(*TokenStats).init(self.allocator);
        defer sorted_tokens.deinit();
        try sorted_tokens.appendSlice(tokens.items);

        const LengthComparator = struct {
            pub fn compare(_: void, a: *TokenStats, b: *TokenStats) bool {
                return a.token.len < b.token.len;
            }
        };
        std.sort.block(*TokenStats, sorted_tokens.items, {}, LengthComparator.compare);

        if (self.debug) {
            std.debug.print("Adding {d} tokens to vocabulary:\n", .{sorted_tokens.items.len});
        }

        var tokens_added: usize = 0;

        for (sorted_tokens.items) |stats| {
            const token = stats.token;

            const token_copy = try self.allocator.dupe(u8, token);
            errdefer self.allocator.free(token_copy);

            const token_id = @as(u32, @intCast(self.vocab.items.len));

            try self.vocab_automaton.insert(token_copy, token_id);

            try self.vocab.append(token_copy);
            try self.vocab_token_ids.put(token_copy, token_id);

            // Update missed gain values for candidate tokens that are substrings of this token
            var updated_count: usize = 0;
            var candidates_it = self.candidate_stats.iterator();
            while (candidates_it.next()) |entry| {
                const candidate = entry.key_ptr.*;
                const candidate_stats = entry.value_ptr.*;

                if (candidate.len >= token.len) continue;

                if (std.mem.indexOf(u8, token, candidate) != null) {
                    // Count how many times the candidate appears in the new token
                    var count: u32 = 0;
                    var pos: usize = 0;
                    while (pos <= token.len - candidate.len) {
                        const found_pos = std.mem.indexOfPos(u8, token, pos, candidate);
                        if (found_pos == null) break;
                        count += 1;
                        pos = found_pos.? + candidate.len;
                    }

                    // Update missed gain based on estimated uses of the new token
                    const additional_missed_gain = @as(i32, @intCast(count * stats.est_n_uses)) *
                        @as(i32, @intCast(candidate.len - 1));

                    candidate_stats.missed_gain_from_superstring_used += additional_missed_gain;
                    candidate_stats.missed_gain_from_superstring_used_computed_at = self.current_step;

                    updated_count += 1;
                }
            }

            if (self.debug and updated_count > 0) {
                std.debug.print("  Updated missed gain for {d} candidate tokens\n", .{updated_count});
            }

            // Remove token from candidate stats
            _ = self.candidate_stats.remove(token);

            if (self.debug) {
                std.debug.print("  Added token {d}: ", .{tokens_added + 1});
                for (token) |byte| {
                    if (byte >= 32 and byte < 127) {
                        std.debug.print("{c}", .{byte});
                    } else {
                        std.debug.print("\\x{x:0>2}", .{byte});
                    }
                }
                std.debug.print(" (length: {d}, uses: {d})\n", .{ token.len, stats.est_n_uses });
            }

            // Clean up the token stats
            stats.deinit(self.allocator);

            tokens_added += 1;
        }

        // Rebuild suffix links for the automaton
        try self.vocab_automaton.computeSuffixLinks();

        const elapsed_ms = std.time.milliTimestamp() - start_time;
        if (self.debug) {
            std.debug.print("Added {d} tokens to vocabulary in {d}ms. New vocabulary size: {d}\n", .{ tokens_added, elapsed_ms, self.vocab.items.len });
        }
    }

    fn removeRandomTokens(self: *VocabLearner, count: usize) !void {
        const start_time = std.time.milliTimestamp();

        if (self.debug) {
            std.debug.print("Randomly removing {d} tokens from vocabulary...\n", .{count});
        }

        const vocab_size = self.vocab.items.len;
        const removable_tokens = vocab_size - 256;

        if (removable_tokens <= 0) {
            if (self.debug) {
                std.debug.print("No removable tokens available.\n", .{});
            }
            return;
        }

        const tokens_to_remove = @min(count, removable_tokens);
        var removed_count: usize = 0;

        var indices = try self.allocator.alloc(usize, removable_tokens);
        defer self.allocator.free(indices);

        for (0..removable_tokens) |i| {
            indices[i] = i + 256; // Skip single-byte tokens
        }

        // Shuffle the indices
        for (0..removable_tokens) |i| {
            const j = self.document_sampler.prng.uintLessThan(usize, removable_tokens);
            const temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
        }

        // Remove the first 'tokens_to_remove' tokens
        for (0..tokens_to_remove) |i| {
            const idx = indices[i];
            if (idx >= self.vocab.items.len) continue;

            const token = self.vocab.items[idx];

            const stats = try TokenStats.init(self.allocator, token);
            stats.n_nonoverlapping_occurrences = 100;

            // Add token back to candidate stats - this takes ownership of the token
            try self.candidate_stats.put(token, stats);

            // Remove token from vocabulary token IDs but DON'T free the token
            // because it's now owned by candidate_stats
            _ = self.vocab_token_ids.remove(token);

            removed_count += 1;
        }

        // Rebuild vocabulary array and automaton
        var new_vocab = ArrayList([]const u8).init(self.allocator);
        errdefer {
            for (new_vocab.items) |token| {
                self.allocator.free(token);
            }
            new_vocab.deinit();
        }

        var new_automaton = try BakaCorasick.init(self.allocator);
        errdefer deinitBakaCorasick(&new_automaton, self.allocator);

        for (self.vocab.items) |token| {
            if (!self.vocab_token_ids.contains(token)) {
                self.allocator.free(token);
                continue;
            }

            const token_id = @as(u32, @intCast(new_vocab.items.len));
            try new_vocab.append(token);
            try new_automaton.insert(token, token_id);
        }

        // Update vocabulary and automaton
        self.vocab.deinit();
        self.vocab = new_vocab;

        deinitBakaCorasick(&self.vocab_automaton, self.allocator);
        self.vocab_automaton = new_automaton;

        // Compute suffix links for new automaton
        try self.vocab_automaton.computeSuffixLinks();

        const elapsed_ms = std.time.milliTimestamp() - start_time;
        if (self.debug) {
            std.debug.print("Removed {d} tokens in {d}ms. New vocabulary size: {d}\n", .{ removed_count, elapsed_ms, self.vocab.items.len });
        }
    }

    fn performFullCorpusScan(self: *VocabLearner) !void {
        const start_time = std.time.milliTimestamp();

        if (self.debug) {
            std.debug.print("Performing full corpus scan...\n", .{});
        }

        // Reset occurrence counts
        var it = self.candidate_stats.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.*.n_nonoverlapping_occurrences = 0;
        }

        // Process corpus to update token statistics
        try self.processCorpusWithTokenIDs();

        const elapsed_ms = std.time.milliTimestamp() - start_time;
        if (self.debug) {
            std.debug.print("Full corpus scan completed in {d}ms\n", .{elapsed_ms});
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

            for (self.vocab.items) |token| {
                if (token.len == 1) {
                    single_byte_count += 1;
                } else {
                    multi_byte_count += 1;
                }
            }

            std.debug.print("Phase 3: Final vocabulary has {d} tokens ({d} single-byte, {d} multi-byte).\n", .{ self.vocab.items.len, single_byte_count, multi_byte_count });

            const mb_percentage = @as(f64, @floatFromInt(multi_byte_count)) /
                @as(f64, @floatFromInt(self.vocab.items.len)) * 100.0;

            std.debug.print("         Multi-byte tokens: {d:.1}% of vocabulary.\n", .{mb_percentage});
            std.debug.print("         Candidate tokens remaining: {d}\n", .{self.candidate_stats.count()});
        }
    }

    /// Save vocabulary to a binary file
    pub fn saveVocabulary(self: *VocabLearner, path: []const u8) !void {
        const file = try std.fs.cwd().createFile(path, .{});
        defer file.close();

        // Create and write header
        const header = VocabHeader{
            .magic = VOCAB_MAGIC,
            .version = VOCAB_VERSION,
            .vocab_size = @intCast(self.vocab.items.len),
            .reserved = [_]u8{0} ** 20,
        };
        try file.writeAll(std.mem.asBytes(&header));

        // Write each token
        for (self.vocab.items, 0..) |token, i| {
            const token_id: u32 = @intCast(i);
            const token_length: u32 = @intCast(token.len);

            // Write token ID and length (convert to little-endian bytes)
            var id_bytes: [4]u8 = undefined;
            var len_bytes: [4]u8 = undefined;
            std.mem.writeInt(u32, &id_bytes, token_id, .little);
            std.mem.writeInt(u32, &len_bytes, token_length, .little);

            try file.writeAll(&id_bytes);
            try file.writeAll(&len_bytes);

            // Write token content
            try file.writeAll(token);
        }

        if (self.debug) {
            std.debug.print("Saved vocabulary with {d} tokens to {s}\n", .{ self.vocab.items.len, path });
        }
    }

    /// Load vocabulary from a binary file
    pub fn loadVocabularyFromFile(allocator: Allocator, path: []const u8, debug: bool) !*VocabLearner {
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

        // Check version compatibility
        if (header.version != VOCAB_VERSION) {
            return error.UnsupportedVersion;
        }

        // Create new VocabLearner with empty initialization
        var learner = try allocator.create(VocabLearner);
        errdefer allocator.destroy(learner);

        learner.* = .{
            .allocator = allocator,
            .vocab = ArrayList([]const u8).init(allocator),
            .vocab_token_ids = std.StringHashMap(u32).init(allocator),
            .candidate_stats = std.StringHashMap(*TokenStats).init(allocator),
            .vocab_automaton = try BakaCorasick.init(allocator),
            .eval_automaton = try BakaCorasick.init(allocator),
            .current_step = 0,
            .document_sampler = try DocumentSampler.init(allocator, &[_][]const u8{}, debug),
            .max_vocab_size = header.vocab_size,
            .top_k_candidates = 200,
            .batch_size = 10,
            .sample_size = 5,
            .last_full_corpus_scan = 0,
            .full_corpus_scan_interval = 5000,
            .debug = debug,
            .simple_prng = std.Random.DefaultPrng.init(blk: {
                var seed: u64 = undefined;
                try std.crypto.randomBytes(std.mem.asBytes(&seed));
                break :blk seed;
            }),
        };

        // Reserve capacity for vocabulary
        try learner.vocab.ensureTotalCapacity(header.vocab_size);

        // Read tokens
        var token_id: u32 = 0;
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

            // Add token to vocabulary
            try learner.vocab.append(token);
            try learner.vocab_token_ids.put(token, token_id);
            try learner.vocab_automaton.insert(token, token_id);

            token_id += 1;
        }

        // Compute suffix links for the automaton
        try learner.vocab_automaton.computeSuffixLinks();

        if (debug) {
            std.debug.print("Loaded vocabulary with {d} tokens from {s}\n", .{ learner.vocab.items.len, path });
        }

        return learner;
    }

    pub fn serializeToBuffer(self: *VocabLearner, allocator: Allocator) ![]u8 {
        // Calculate total size needed
        var total_size: usize = HEADER_SIZE;
        for (self.vocab.items) |token| {
            total_size += 8 + token.len; // 4 bytes ID + 4 bytes length + token content
        }

        // Allocate buffer
        const buffer = try allocator.alloc(u8, total_size);
        errdefer allocator.free(buffer);

        // Create and write header
        const header = VocabHeader{
            .magic = VOCAB_MAGIC,
            .version = VOCAB_VERSION,
            .vocab_size = @intCast(self.vocab.items.len),
            .reserved = [_]u8{0} ** 20,
        };
        @memcpy(buffer[0..HEADER_SIZE], std.mem.asBytes(&header));

        // Write tokens to buffer
        var offset: usize = HEADER_SIZE;
        for (self.vocab.items, 0..) |token, i| {
            const token_id: u32 = @intCast(i);
            const token_length: u32 = @intCast(token.len);

            // Write token ID and length
            std.mem.writeInt(u32, buffer[offset..][0..4], token_id, .little);
            offset += 4;
            std.mem.writeInt(u32, buffer[offset..][0..4], token_length, .little);
            offset += 4;

            // Write token content
            @memcpy(buffer[offset..][0..token.len], token);
            offset += token.len;
        }

        return buffer;
    }

    pub fn deserializeFromBuffer(allocator: Allocator, buffer: []const u8, debug: bool) !*VocabLearner {
        if (buffer.len < HEADER_SIZE) {
            return error.BufferTooSmall;
        }

        // Parse header
        const header = @as(*const VocabHeader, @ptrCast(@alignCast(buffer.ptr))).*;

        // Validate magic number
        if (!std.mem.eql(u8, &header.magic, &VOCAB_MAGIC)) {
            return error.InvalidMagicNumber;
        }

        // Check version compatibility
        if (header.version != VOCAB_VERSION) {
            return error.UnsupportedVersion;
        }

        // Create new VocabLearner with empty initialization
        var learner = try allocator.create(VocabLearner);
        errdefer allocator.destroy(learner);

        learner.* = .{
            .allocator = allocator,
            .vocab = ArrayList([]const u8).init(allocator),
            .vocab_token_ids = std.StringHashMap(u32).init(allocator),
            .candidate_stats = std.StringHashMap(*TokenStats).init(allocator),
            .vocab_automaton = try BakaCorasick.init(allocator),
            .eval_automaton = try BakaCorasick.init(allocator),
            .current_step = 0,
            .document_sampler = try DocumentSampler.init(allocator, &[_][]const u8{}, debug),
            .max_vocab_size = header.vocab_size,
            .top_k_candidates = 200,
            .batch_size = 10,
            .sample_size = 5,
            .last_full_corpus_scan = 0,
            .full_corpus_scan_interval = 5000,
            .debug = debug,
            .simple_prng = std.Random.DefaultPrng.init(blk: {
                var seed: u64 = undefined;
                try std.crypto.randomBytes(std.mem.asBytes(&seed));
                break :blk seed;
            }),
        };

        try learner.vocab.ensureTotalCapacity(header.vocab_size);

        // Parse tokens
        var offset: usize = HEADER_SIZE;
        var token_id: u32 = 0;

        while (token_id < header.vocab_size) {
            if (offset + 8 > buffer.len) {
                return error.BufferTooSmall;
            }

            const read_token_id = std.mem.readInt(u32, buffer[offset..][0..4], .little);
            offset += 4;
            const token_length = std.mem.readInt(u32, buffer[offset..][0..4], .little);
            offset += 4;

            if (read_token_id != token_id) {
                return error.InvalidTokenSequence;
            }

            if (offset + token_length > buffer.len) {
                return error.BufferTooSmall;
            }

            const token = try allocator.dupe(u8, buffer[offset..][0..token_length]);
            offset += token_length;

            try learner.vocab.append(token);
            try learner.vocab_token_ids.put(token, token_id);
            try learner.vocab_automaton.insert(token, token_id);

            token_id += 1;
        }

        try learner.vocab_automaton.computeSuffixLinks();

        if (debug) {
            std.debug.print("Deserialized vocabulary with {d} tokens from buffer\n", .{learner.vocab.items.len});
        }

        return learner;
    }
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Parse command line arguments
    var args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 3) {
        std.debug.print("Usage: {s} <tokenset_file> <corpus_path> [corpus_path...]\n", .{args[0]});
        return;
    }

    const tokenset_path = args[1];
    const corpus_paths = args[2..];

    const debug = true;

    var learner = try VocabLearner.init(allocator, tokenset_path, corpus_paths, 10000, debug);
    defer learner.deinit();

    // check if everything initialized properly
    try learner.checkPhase1Initialization();

    // Phase 2: Process corpus and calculate token occurrences
    try learner.processCorpusWithTokenIDs();
    try learner.checkPhase2CorpusProcessing();

    // Phase 3: Build vocabulary through iterative selection
    try learner.buildVocabulary();
    try learner.checkPhase3MainLoop();

    try learner.saveVocabulary("vocab.bin");
}
