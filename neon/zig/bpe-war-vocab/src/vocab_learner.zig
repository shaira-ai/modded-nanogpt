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

const CANDIDATE_TOKEN_FLAG: u32 = 1 << 31;

const TokenMatch = struct {
    len: usize,
    is_candidate: bool,
    id: u32,
};

const DocInfo = struct {
    idx: usize,
    size: usize,
};

// Task struct for worker threads
const DpEvalTask = struct {
    doc_idx: usize,
    document: []const u8,
    token_end_positions: std.AutoHashMap(usize, std.ArrayList(TokenMatch)),
    candidate_positions: []std.ArrayList(usize),
    candidates_in_doc: []usize,
    base_costs: []u32,
    base_masks: []u64,
};

// Result struct for worker threads
const DpEvalResult = struct {
    doc_idx: usize,
    token_savings: []u32,
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

        var gpt_it = self.gpt_token_to_string.iterator();
        while (gpt_it.next()) |entry| {
            self.allocator.free(entry.value_ptr.*);
        }
        self.gpt_token_to_string.deinit();

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
            if (count == 0 or length > 10) continue;

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

    pub fn processCorpus(self: *VocabLearner) !void {
        const start_time = std.time.milliTimestamp();

        if (self.debug) {
            std.debug.print("Processing corpus using raw text conversion...\n", .{});
        }

        // Load GPT vocabulary if needed
        if (self.gpt_token_to_string.count() == 0) {
            try self.loadGPTVocabulary("vocab.json");
        }

        // Find all binary files
        var corpus_files = try self.collectBinFiles();
        defer {
            for (corpus_files.items) |path| {
                self.allocator.free(path);
            }
            corpus_files.deinit();
        }

        if (self.debug) {
            std.debug.print("Found {d} .bin files to process\n", .{corpus_files.items.len});
        }

        // Setup for token counting
        var combined_automaton = try BakaCorasick.init(self.allocator);
        defer deinitBakaCorasick(&combined_automaton, self.allocator);

        // Add all candidate tokens to the automaton
        var token_id_to_stats = std.AutoHashMap(u32, *TokenStats).init(self.allocator);
        defer token_id_to_stats.deinit();

        if (self.debug) {
            std.debug.print("Building search automaton with {d} candidate tokens...\n", .{self.candidate_stats.count()});
        }

        // Add all candidate tokens to the automaton
        var candidates_it = self.candidate_stats.iterator();
        var id: u32 = 0;
        while (candidates_it.next()) |entry| {
            const token = entry.key_ptr.*;
            const stats = entry.value_ptr.*;

            // Reset occurrence counts
            stats.n_nonoverlapping_occurrences = 0;

            // Add to automaton
            try combined_automaton.insert(token, id);
            try token_id_to_stats.put(id, stats);
            id += 1;
        }

        try combined_automaton.computeSuffixLinks();

        if (self.debug) {
            std.debug.print("Automaton built with {d} states\n", .{combined_automaton.len});
        }

        const ThreadContext = struct {
            learner: *VocabLearner,
            corpus_path: []const u8,
            file_idx: usize,
            total_files: usize,
            automaton: *BakaCorasick,
            token_id_to_stats: *std.AutoHashMap(u32, *TokenStats),
            gpt_vocab: *std.AutoHashMap(u16, []const u8),
            mutex: *std.Thread.Mutex,
            progress_mutex: *std.Thread.Mutex,
        };

        var mutex = std.Thread.Mutex{};
        var progress_mutex = std.Thread.Mutex{};

        const workerFn = struct {
            fn processFile(ctx: ThreadContext) void {
                const file_start_time = std.time.milliTimestamp();

                // Report file starting
                ctx.progress_mutex.lock();
                if (ctx.learner.debug) {
                    std.debug.print("\n[{d}/{d}] Starting file: {s}\n", .{ ctx.file_idx + 1, ctx.total_files, ctx.corpus_path });
                }
                ctx.progress_mutex.unlock();

                // Open binary file
                const file = std.fs.cwd().openFile(ctx.corpus_path, .{}) catch |err| {
                    ctx.progress_mutex.lock();
                    if (ctx.learner.debug) {
                        std.debug.print("  Error opening file: {s}\n", .{@errorName(err)});
                    }
                    ctx.progress_mutex.unlock();
                    return;
                };
                defer file.close();

                // Get file size for progress reporting
                const file_size = file.getEndPos() catch |err| {
                    ctx.progress_mutex.lock();
                    if (ctx.learner.debug) {
                        std.debug.print("  Error getting file size: {s}\n", .{@errorName(err)});
                    }
                    ctx.progress_mutex.unlock();
                    return;
                };
                const data_size = file_size - @min(file_size, 1024);

                ctx.progress_mutex.lock();
                if (ctx.learner.debug) {
                    std.debug.print("  File size: {d:.2} MB\n", .{@as(f64, @floatFromInt(data_size)) / (1024.0 * 1024.0)});
                }
                ctx.progress_mutex.unlock();

                // Skip header if present
                if (file_size > 1024) {
                    file.seekTo(1024) catch return;
                }

                // Read in chunks
                const CHUNK_SIZE = 4 * 1024 * 1024; // 4MB chunks
                var buffer = ctx.learner.allocator.alloc(u8, CHUNK_SIZE) catch return;
                defer ctx.learner.allocator.free(buffer);

                // Create text buffer for decoded content
                var text_buffer = std.ArrayList(u8).init(ctx.learner.allocator);
                defer text_buffer.deinit();

                var bytes_processed: usize = 0;
                var last_progress_time = std.time.milliTimestamp();
                const progress_interval_ms = 1000; // Update progress every second

                // Process file in chunks
                while (true) {
                    const bytes_read = file.readAll(buffer) catch break;
                    if (bytes_read == 0) break;

                    // Convert token IDs to text
                    const complete_tokens = bytes_read / 2;
                    for (0..complete_tokens) |i| {
                        const token_offset = i * 2;
                        const token_id = std.mem.bytesToValue(u16, buffer[token_offset..][0..2]);

                        // Get string for this token
                        if (ctx.gpt_vocab.get(token_id)) |token_str| {
                            text_buffer.appendSlice(token_str) catch continue;
                        }
                    }

                    bytes_processed += bytes_read;

                    // Progress reporting
                    const current_time = std.time.milliTimestamp();
                    if (ctx.learner.debug and current_time - last_progress_time >= progress_interval_ms) {
                        const mb_processed = @as(f64, @floatFromInt(bytes_processed)) / (1024.0 * 1024.0);
                        const percent_complete = if (data_size > 0)
                            @as(f64, @floatFromInt(bytes_processed)) / @as(f64, @floatFromInt(data_size)) * 100.0
                        else
                            100.0;

                        ctx.progress_mutex.lock();
                        // Print progress bar: [=====>    ] 45.5%
                        std.debug.print("\r[{d}/{d}] ", .{ ctx.file_idx + 1, ctx.total_files });
                        std.debug.print("[", .{});
                        const bar_width = 20;
                        const filled_width = @as(usize, @intFromFloat(@min(percent_complete / 100.0 * @as(f64, @floatFromInt(bar_width)), @as(f64, @floatFromInt(bar_width)))));

                        for (0..filled_width) |_| {
                            std.debug.print("=", .{});
                        }

                        if (filled_width < bar_width) {
                            std.debug.print(">", .{});
                            for (filled_width + 1..bar_width) |_| {
                                std.debug.print(" ", .{});
                            }
                        }

                        std.debug.print("] {d:.1}% - {d:.2}/{d:.2} MB", .{ percent_complete, mb_processed, @as(f64, @floatFromInt(data_size)) / (1024.0 * 1024.0) });
                        ctx.progress_mutex.unlock();

                        last_progress_time = current_time;
                    }

                    // Process accumulated text when buffer gets large
                    if (text_buffer.items.len > CHUNK_SIZE) {
                        processText(ctx, text_buffer.items);
                        text_buffer.clearRetainingCapacity();
                    }
                }

                // Process any remaining text
                if (text_buffer.items.len > 0) {
                    processText(ctx, text_buffer.items);
                }

                const file_elapsed_sec = @as(f64, @floatFromInt(std.time.milliTimestamp() - file_start_time)) / 1000.0;
                ctx.progress_mutex.lock();
                if (ctx.learner.debug) {
                    // Clear line first
                    std.debug.print("\r                                                                   \r", .{});
                    std.debug.print("  Completed file {d}/{d}: {s} in {d:.2}s\n", .{ ctx.file_idx + 1, ctx.total_files, ctx.corpus_path, file_elapsed_sec });
                }
                ctx.progress_mutex.unlock();
            }

            fn processText(ctx: ThreadContext, text: []const u8) void {
                // Only do processing if we have text
                if (text.len == 0) return;

                // Find all tokens using Aho-Corasick
                var token_counts = std.AutoHashMap(u32, u32).init(ctx.learner.allocator);
                defer token_counts.deinit();

                // Keep track of used positions in the text
                var used_positions = ctx.learner.allocator.alloc(bool, text.len) catch return;
                defer ctx.learner.allocator.free(used_positions);
                @memset(used_positions, false);

                // First pass: collect best matches at each position
                var position_matches = ctx.learner.allocator.alloc(?struct { token_id: u32, len: usize }, text.len) catch return;
                defer ctx.learner.allocator.free(position_matches);
                for (0..position_matches.len) |i| {
                    position_matches[i] = null;
                }

                // Scan text with the automaton
                var current_state: u32 = 0;
                for (text, 0..) |byte, pos| {
                    current_state = ctx.automaton.transitions[current_state][byte];

                    // Check if this state represents a match
                    if (ctx.automaton.info[current_state].token_id != BakaCorasick.NO_TOKEN) {
                        const token_id = ctx.automaton.info[current_state].token_id;
                        const token_len = ctx.automaton.info[current_state].depth;

                        // Store this match if it's longer than existing match at this end position
                        if (position_matches[pos] == null or position_matches[pos].?.len < token_len) {
                            position_matches[pos] = .{ .token_id = token_id, .len = token_len };
                        }
                    }

                    // Check suffix links for additional matches
                    var suffix = ctx.automaton.info[current_state].green;
                    while (suffix != 0) {
                        if (ctx.automaton.info[suffix].token_id != BakaCorasick.NO_TOKEN) {
                            const token_id = ctx.automaton.info[suffix].token_id;
                            const token_len = ctx.automaton.info[suffix].depth;

                            // Store this match if it's longer than existing match at this end position
                            if (position_matches[pos] == null or position_matches[pos].?.len < token_len) {
                                position_matches[pos] = .{ .token_id = token_id, .len = token_len };
                            }
                        }
                        suffix = ctx.automaton.info[suffix].green;
                    }
                }

                // Second pass: process matches in order of end position (which is naturally the case)
                for (position_matches, 0..) |match_opt, pos| {
                    if (match_opt) |match| {
                        const start_pos = pos + 1 - match.len;

                        // Check if any position in this match is already used
                        var is_valid = true;
                        for (start_pos..pos + 1) |i| {
                            if (used_positions[i]) {
                                is_valid = false;
                                break;
                            }
                        }

                        if (is_valid) {
                            // Count this match
                            if (token_counts.getPtr(match.token_id)) |count_ptr| {
                                count_ptr.* += 1;
                            } else {
                                token_counts.put(match.token_id, 1) catch continue;
                            }

                            // Mark positions as used
                            for (start_pos..pos + 1) |i| {
                                used_positions[i] = true;
                            }
                        }
                    }
                }

                // Update global counts with thread mutex
                ctx.mutex.lock();
                defer ctx.mutex.unlock();

                var it = token_counts.iterator();
                while (it.next()) |entry| {
                    const token_id = entry.key_ptr.*;
                    const count = entry.value_ptr.*;

                    if (ctx.token_id_to_stats.get(token_id)) |stats| {
                        stats.n_nonoverlapping_occurrences += count;
                    }
                }
            }
        }.processFile;

        // Process files in parallel
        const available_cores = try std.Thread.getCpuCount();
        const num_threads = @min(available_cores, corpus_files.items.len);

        if (self.debug) {
            std.debug.print("Processing {d} files using {d} threads\n", .{ corpus_files.items.len, num_threads });
        }

        // Process files in batches
        const batch_size = num_threads;
        var batch_start: usize = 0;
        var batch_num: usize = 0;
        const total_batches = (corpus_files.items.len + batch_size - 1) / batch_size;

        while (batch_start < corpus_files.items.len) {
            batch_num += 1;
            const batch_end = @min(batch_start + batch_size, corpus_files.items.len);
            const current_batch_size = batch_end - batch_start;

            if (self.debug) {
                std.debug.print("\nProcessing batch {d}/{d} ({d} files)\n", .{ batch_num, total_batches, current_batch_size });
            }

            var threads = try self.allocator.alloc(std.Thread, current_batch_size);
            defer self.allocator.free(threads);

            // Start worker threads
            for (batch_start..batch_end) |i| {
                const thread_idx = i - batch_start;
                const context = ThreadContext{
                    .learner = self,
                    .corpus_path = corpus_files.items[i],
                    .file_idx = i,
                    .total_files = corpus_files.items.len,
                    .automaton = &combined_automaton,
                    .token_id_to_stats = &token_id_to_stats,
                    .gpt_vocab = &self.gpt_token_to_string,
                    .mutex = &mutex,
                    .progress_mutex = &progress_mutex,
                };

                threads[thread_idx] = try std.Thread.spawn(.{}, workerFn, .{context});
            }

            // Wait for threads to complete
            for (threads) |thread| {
                thread.join();
            }

            // Report batch completion
            if (self.debug) {
                std.debug.print("Completed batch {d}/{d}\n", .{ batch_num, total_batches });
            }

            batch_start = batch_end;
        }

        // Calculate bounds after counting
        try self.calculateInitialBounds();

        const elapsed_sec = @as(f64, @floatFromInt(std.time.milliTimestamp() - start_time)) / 1000.0;
        if (self.debug) {
            std.debug.print("\nCompleted corpus processing in {d:.2}s\n", .{elapsed_sec});
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

                        if (self.debug) {
                            std.debug.print("  Added file: {s}\n", .{full_path});
                        }
                    }
                }
            }
        }

        return result;
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
        try self.processCorpus();

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

    fn evaluateCandidatesOnDocumentsDP(self: *VocabLearner, candidates: *const ArrayList(*TokenStats), documents: []const []const u8) !void {
        const start_time = std.time.milliTimestamp();

        if (self.debug) {
            std.debug.print("Evaluating {d} candidates on {d} documents using parallelized DP...\n", .{ candidates.items.len, documents.len });
        }

        // Reset estimated uses for all candidates
        for (candidates.items) |stats| {
            stats.est_n_uses = 0;
        }

        // Early-stopping thresholds (tokens with more savings than this can stop early)
        const early_stop_threshold: u32 = 500;
        var tokens_reached_threshold = std.AutoHashMap(usize, bool).init(self.allocator);
        defer tokens_reached_threshold.deinit();

        // Sample a smaller number of documents (we don't need to process all 10k)
        const sample_size = @min(1000, documents.len); // Using 200 documents instead of 1000
        const sample_docs = try self.allocator.alloc([]const u8, sample_size);
        defer self.allocator.free(sample_docs);

        // Select sample documents, prioritizing smaller ones for faster processing
        var doc_sizes = try self.allocator.alloc(DocInfo, documents.len);
        defer self.allocator.free(doc_sizes);

        for (documents, 0..) |doc, i| {
            doc_sizes[i] = DocInfo{ .idx = i, .size = doc.len };
        }

        // Sort documents by size (ascending)
        std.sort.block(DocInfo, doc_sizes, {}, struct {
            fn lessThan(_: void, a: DocInfo, b: DocInfo) bool {
                // Skip empty documents
                if (a.size == 0) return false;
                if (b.size == 0) return true;
                return a.size < b.size;
            }
        }.lessThan);

        // Choose smaller documents with a preference, but include some larger ones
        const smaller_docs_count = sample_size / 5 * 4; // Calculate 80% of the sample, avoid overflow

        for (0..sample_size) |i| {
            // Use 80% smaller documents, 20% random documents
            var doc_idx: usize = 0;

            if (i < smaller_docs_count) {
                // For the first 80%, use smaller documents
                if (i < doc_sizes.len) {
                    doc_idx = doc_sizes[i].idx;
                } else {
                    doc_idx = i % documents.len;
                }
            } else {
                // For the remaining 20%, use some larger documents
                if (documents.len > 0) {
                    // Safely take from the end of the array
                    const offset = i % documents.len;
                    if (offset < documents.len) {
                        doc_idx = doc_sizes[documents.len - 1 - offset].idx;
                    } else {
                        doc_idx = i % documents.len;
                    }
                } else {
                    doc_idx = 0;
                }
            }

            sample_docs[i] = documents[doc_idx];
        }

        // Track documents processed and candidates with impact
        var docs_processed: usize = 0;
        var candidates_with_impact: usize = 0;

        // 1. Create a single combined automaton (vocab + candidates) - ONLY ONCE
        if (self.debug) {
            std.debug.print("Creating combined automaton with vocabulary and candidate tokens...\n", .{});
        }

        var combined_automaton = try BakaCorasick.init(self.allocator);
        defer deinitBakaCorasick(&combined_automaton, self.allocator);

        // Add vocabulary tokens
        for (self.vocab.items, 0..) |token, i| {
            try combined_automaton.insert(token, @intCast(i));
        }

        // Add candidate tokens with flag
        for (candidates.items, 0..) |stats, i| {
            try combined_automaton.insert(stats.token, CANDIDATE_TOKEN_FLAG | @as(u32, @intCast(i)));
        }

        try combined_automaton.computeSuffixLinks();

        if (self.debug) {
            std.debug.print("Combined automaton created with {d} states\n", .{combined_automaton.len});
        }

        // Parallelize document processing using thread pool
        // Determine number of threads based on CPU cores
        const num_threads = 10;

        if (self.debug) {
            std.debug.print("Using {d} threads for parallel processing\n", .{num_threads});
        }

        // Process documents in batches
        const batch_size = @min(num_threads * 2, sample_size);
        var batch_start: usize = 0;

        // Create a mutex for thread-safe updates to the token statistics
        var mutex = std.Thread.Mutex{};

        // Track global stopping condition
        var all_tokens_reached_threshold = false;

        while (batch_start < sample_size and !all_tokens_reached_threshold) {
            const batch_end = @min(batch_start + batch_size, sample_size);
            const current_batch_size = batch_end - batch_start;

            // Prepare thread contexts
            var threads = try self.allocator.alloc(std.Thread, current_batch_size);
            defer self.allocator.free(threads);

            var results = try self.allocator.alloc(std.AutoHashMap(usize, u32), current_batch_size);
            defer self.allocator.free(results);

            // Initialize result maps
            for (0..current_batch_size) |i| {
                results[i] = std.AutoHashMap(usize, u32).init(self.allocator);
            }
            defer {
                for (0..current_batch_size) |i| {
                    results[i].deinit();
                }
            }

            // Thread worker function
            const ThreadContext = struct {
                learner: *VocabLearner,
                candidates: *const ArrayList(*TokenStats),
                document: []const u8,
                doc_idx: usize,
                automaton: *BakaCorasick,
                result_map: *std.AutoHashMap(usize, u32),
                mutex: *std.Thread.Mutex,
                tokens_reached_threshold: *std.AutoHashMap(usize, bool),
                early_stop_threshold: u32,
            };

            const workerFn = struct {
                fn processDocument(ctx: ThreadContext) void {
                    if (ctx.document.len == 0) return;

                    // For very large documents, only process the first part
                    const max_doc_size = 30000; // 30KB is enough for a representative sample
                    const doc_size = @min(ctx.document.len, max_doc_size);
                    const doc = ctx.document[0..doc_size];

                    // 1. Find all token positions in a single scan
                    var token_end_positions = std.AutoHashMap(usize, std.ArrayList(TokenMatch)).init(ctx.learner.allocator);
                    defer {
                        var it = token_end_positions.iterator();
                        while (it.next()) |entry| {
                            entry.value_ptr.*.deinit();
                        }
                        token_end_positions.deinit();
                    }

                    // Also track which candidates appear in this document and their positions
                    var candidate_positions = ctx.learner.allocator.alloc(std.ArrayList(usize), ctx.candidates.items.len) catch return;
                    defer {
                        for (candidate_positions) |*pos_list| {
                            pos_list.deinit();
                        }
                        ctx.learner.allocator.free(candidate_positions);
                    }

                    for (0..ctx.candidates.items.len) |i| {
                        candidate_positions[i] = std.ArrayList(usize).init(ctx.learner.allocator);
                    }

                    // Scan document with combined automaton
                    var current_state: u32 = 0;
                    for (doc, 0..) |byte, i| {
                        current_state = ctx.automaton.transitions[current_state][byte];

                        if (ctx.automaton.info[current_state].token_id != BakaCorasick.NO_TOKEN) {
                            const token_id = ctx.automaton.info[current_state].token_id;
                            const token_len = ctx.automaton.info[current_state].depth;
                            const end_pos = i + 1;

                            const is_candidate = (token_id & CANDIDATE_TOKEN_FLAG) != 0;
                            const actual_id = token_id & ~CANDIDATE_TOKEN_FLAG;

                            // Record this match
                            var matches = token_end_positions.get(end_pos);
                            if (matches == null) {
                                token_end_positions.put(end_pos, std.ArrayList(TokenMatch).init(ctx.learner.allocator)) catch continue;
                                matches = token_end_positions.get(end_pos);
                            }

                            matches.?.append(TokenMatch{
                                .len = token_len,
                                .is_candidate = is_candidate,
                                .id = actual_id,
                            }) catch continue;

                            // If it's a candidate, also track its position
                            if (is_candidate and actual_id < candidate_positions.len) {
                                // Check if this token has already reached threshold
                                ctx.mutex.lock();
                                const reached_threshold = ctx.tokens_reached_threshold.get(actual_id) != null;
                                ctx.mutex.unlock();

                                if (!reached_threshold) {
                                    candidate_positions[actual_id].append(end_pos) catch continue;
                                }
                            }
                        }

                        var suffix = ctx.automaton.info[current_state].green;
                        while (suffix != 0) {
                            if (ctx.automaton.info[suffix].token_id != BakaCorasick.NO_TOKEN) {
                                const token_id = ctx.automaton.info[suffix].token_id;
                                const token_len = ctx.automaton.info[suffix].depth;
                                const end_pos = i + 1;

                                const is_candidate = (token_id & CANDIDATE_TOKEN_FLAG) != 0;
                                const actual_id = token_id & ~CANDIDATE_TOKEN_FLAG;

                                // Record this match
                                var matches = token_end_positions.get(end_pos);
                                if (matches == null) {
                                    token_end_positions.put(end_pos, std.ArrayList(TokenMatch).init(ctx.learner.allocator)) catch continue;
                                    matches = token_end_positions.get(end_pos);
                                }

                                matches.?.append(TokenMatch{
                                    .len = token_len,
                                    .is_candidate = is_candidate,
                                    .id = actual_id,
                                }) catch continue;

                                // If it's a candidate, also track its position
                                if (is_candidate and actual_id < candidate_positions.len) {
                                    // Check if this token has already reached threshold
                                    ctx.mutex.lock();
                                    const reached_threshold = ctx.tokens_reached_threshold.get(actual_id) != null;
                                    ctx.mutex.unlock();

                                    if (!reached_threshold) {
                                        candidate_positions[actual_id].append(end_pos) catch continue;
                                    }
                                }
                            }
                            suffix = ctx.automaton.info[suffix].green;
                        }
                    }

                    // Find which candidates appear in this document
                    var candidates_in_doc = std.ArrayList(usize).init(ctx.learner.allocator);
                    defer candidates_in_doc.deinit();

                    for (0..ctx.candidates.items.len) |i| {
                        // Check if this token has already reached threshold
                        ctx.mutex.lock();
                        const reached_threshold = ctx.tokens_reached_threshold.get(i) != null;
                        ctx.mutex.unlock();

                        if (!reached_threshold and candidate_positions[i].items.len > 0) {
                            candidates_in_doc.append(i) catch continue;
                        }
                    }

                    if (candidates_in_doc.items.len == 0) return;

                    // 2. Solve base DP problem with current vocabulary
                    var costs = ctx.learner.allocator.alloc(u32, doc.len + 1) catch return;
                    defer ctx.learner.allocator.free(costs);

                    var masks = ctx.learner.allocator.alloc(u64, doc.len + 1) catch return;
                    defer ctx.learner.allocator.free(masks);

                    // Initialize arrays
                    @memset(costs, std.math.maxInt(u32));
                    costs[0] = 0;
                    @memset(masks, 0);

                    // Solve base DP
                    for (1..doc.len + 1) |i| {
                        // Always consider 1-byte token (implicit)
                        if (i >= 1 and costs[i - 1] + 1 < costs[i]) {
                            costs[i] = costs[i - 1] + 1;
                        }

                        // Consider vocabulary tokens ending at position i
                        if (token_end_positions.get(i)) |token_matches| {
                            for (token_matches.items) |match| {
                                // Skip candidate tokens for base solution
                                if (match.is_candidate) continue;

                                const start = i - match.len;
                                if (costs[start] + 1 < costs[i]) {
                                    costs[i] = costs[start] + 1;

                                    // Set bit for this token length
                                    if (match.len >= 2 and match.len <= 65) {
                                        masks[i] |= (@as(u64, 1) << @intCast(match.len - 2));
                                    }
                                }
                            }
                        }
                    }

                    // Store base token count
                    const base_token_count = costs[doc.len];

                    // 3. For each candidate, recalculate from where it first appears
                    for (candidates_in_doc.items) |candidate_idx| {
                        // Check again if this token has already reached threshold
                        ctx.mutex.lock();
                        const reached_threshold = ctx.tokens_reached_threshold.get(candidate_idx) != null;
                        ctx.mutex.unlock();

                        if (reached_threshold) continue;

                        const token = ctx.candidates.items[candidate_idx].token;
                        const token_positions = candidate_positions[candidate_idx].items;

                        if (token_positions.len == 0) continue;

                        // Copy costs array
                        var candidate_costs = ctx.learner.allocator.dupe(u32, costs) catch continue;
                        defer ctx.learner.allocator.free(candidate_costs);

                        // Find first position
                        var first_pos = doc.len;
                        for (token_positions) |pos| {
                            if (pos < first_pos) {
                                first_pos = pos;
                            }
                        }

                        // Recalculate from first position
                        for (first_pos..doc.len + 1) |i| {
                            // Check if this position is end of candidate token
                            var is_token_end = false;
                            for (token_positions) |pos| {
                                if (pos == i) {
                                    is_token_end = true;
                                    break;
                                }
                            }

                            if (is_token_end) {
                                // Consider using candidate token
                                const start = i - token.len;
                                if (start < candidate_costs.len and
                                    candidate_costs[start] + 1 < candidate_costs[i])
                                {
                                    candidate_costs[i] = candidate_costs[start] + 1;
                                }
                            }

                            // Process existing tokens using bitmasks
                            const mask = masks[i];
                            const cnt = @popCount(mask);

                            if (cnt > 0) {
                                var temp_mask = mask;
                                for (0..cnt) |_| {
                                    const tz = @ctz(temp_mask);
                                    const lookback_amt = tz + 2;

                                    if (i >= lookback_amt) {
                                        const start = i - lookback_amt;
                                        if (candidate_costs[start] + 1 < candidate_costs[i]) {
                                            candidate_costs[i] = candidate_costs[start] + 1;
                                        }
                                    }

                                    // More efficient bit clearing as suggested by sasuke
                                    temp_mask &= temp_mask - 1;
                                }
                            }

                            // Always consider 1-byte token
                            if (i >= 1 and candidate_costs[i - 1] + 1 < candidate_costs[i]) {
                                candidate_costs[i] = candidate_costs[i - 1] + 1;
                            }
                        }

                        // Calculate tokens saved
                        const candidate_token_count = candidate_costs[doc.len];

                        if (base_token_count > candidate_token_count) {
                            const tokens_saved = base_token_count - candidate_token_count;

                            // Store result in thread-local hashmap
                            ctx.result_map.put(candidate_idx, @intCast(tokens_saved)) catch continue;

                            // Check for early stopping threshold
                            if (tokens_saved >= ctx.early_stop_threshold) {
                                ctx.mutex.lock();
                                ctx.tokens_reached_threshold.put(candidate_idx, true) catch {};
                                ctx.mutex.unlock();
                            }
                        }
                    }
                }
            }.processDocument;

            // Start worker threads
            for (0..current_batch_size) |i| {
                const thread_doc_idx = batch_start + i;
                const document = sample_docs[thread_doc_idx];

                const thread_context = ThreadContext{
                    .learner = self,
                    .candidates = candidates,
                    .document = document,
                    .doc_idx = thread_doc_idx,
                    .automaton = &combined_automaton,
                    .result_map = &results[i],
                    .mutex = &mutex,
                    .tokens_reached_threshold = &tokens_reached_threshold,
                    .early_stop_threshold = early_stop_threshold,
                };

                threads[i] = try std.Thread.spawn(.{}, workerFn, .{thread_context});
            }

            // Wait for worker threads to complete
            for (0..current_batch_size) |i| {
                threads[i].join();
            }

            // Update global stats from thread results
            var batch_docs_processed: usize = 0;

            for (0..current_batch_size) |i| {
                var thread_result = &results[i];

                if (thread_result.count() > 0) {
                    batch_docs_processed += 1;

                    var it = thread_result.iterator();
                    while (it.next()) |entry| {
                        const candidate_idx = entry.key_ptr.*;
                        const tokens_saved = entry.value_ptr.*;

                        mutex.lock();
                        candidates.items[candidate_idx].est_n_uses += tokens_saved;
                        mutex.unlock();

                        candidates_with_impact += 1;
                    }
                }
            }

            docs_processed += batch_docs_processed;

            // Check if all tokens have reached threshold
            mutex.lock();
            all_tokens_reached_threshold = tokens_reached_threshold.count() == candidates.items.len;
            mutex.unlock();

            if (all_tokens_reached_threshold and self.debug) {
                std.debug.print("  All tokens reached early stopping threshold\n", .{});
            }

            batch_start = batch_end;
        }

        // Scale results to match the sample size
        const scale_factor = if (docs_processed > 0)
            @as(f32, @floatFromInt(sample_size)) / @as(f32, @floatFromInt(docs_processed))
        else
            1.0;

        if (scale_factor != 1.0) {
            for (candidates.items) |stats| {
                stats.est_n_uses = @intCast(@as(u32, @intFromFloat(@as(f32, @floatFromInt(stats.est_n_uses)) * scale_factor)));
            }
        }

        const elapsed_ms = std.time.milliTimestamp() - start_time;
        if (self.debug) {
            std.debug.print("Evaluated {d} documents with DP, {d} candidates had impact. Completed in {d}ms\n", .{ docs_processed, candidates_with_impact, elapsed_ms });
        }
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

    var learner = try VocabLearner.init(allocator, tokenset_path, corpus_paths, 2000, debug);
    defer learner.deinit();

    // check if everything initialized properly
    try learner.checkPhase1Initialization();

    // Phase 2: Process corpus and calculate token occurrences
    try learner.processCorpus();
    try learner.checkPhase2CorpusProcessing();

    // Phase 3: Build vocabulary through iterative selection
    try learner.buildVocabulary();
    try learner.checkPhase3MainLoop();

    try learner.saveVocabulary("vocab.bin");
}
