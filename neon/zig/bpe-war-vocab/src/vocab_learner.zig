const std = @import("std");
const BakaCorasick = @import("baka_corasick.zig").BakaCorasick;
const fineweb = @import("data_loader.zig").FinewebDataLoader;
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;

pub const TokenStats = extern struct {
    str_start_idx: u64,
    str_len: u16,
    is_in_vocab: bool = false,
    n_nonoverlapping_occurrences: u64 = 0,
    sampled_occurrences: u64 = 0,
    sampled_savings: u64 = 0,
    sampled_step: u64 = 0,
    est_total_savings: f64 = 0,
    //est_n_uses: u32 = 0,
    //max_gain_for_nonoverlapping_occurrences: u32 = 0,
    //max_gain_for_nonoverlapping_occurrences_computed_at: u32 = 0,
    //missed_gain_from_superstring_used: u32 = 0,
    //missed_gain_from_superstring_used_computed_at: u32 = 0,

    pub inline fn getCurrentValueBound(self: TokenStats) f64 {
        return self.est_total_savings;
    }
};

const SampleStats = struct {
    sampled_occurrences: u64 = 0,
    sampled_savings: u64 = 0,
    token_id: u32,
};

const MatchInfo = struct {
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
    doc_range: [2]usize,
    documents: []const []const u8,
    doc_infos: []const DocInfo,
    automaton: *BakaCorasick,
    candidates: *const ArrayList(*TokenStats),
    vocab_size: u32,
    candidates_size: u32,
    progress_mutex: *std.Thread.Mutex,
    total_docs_processed: *usize,
    total_docs: usize,
    debug: bool,
    allocator: Allocator, // Added this field
};

// Result struct for worker threads
const DpEvalResult = struct {
    tokens_saved: []u32,
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

    pub fn sampleDocuments(self: *DocumentSampler, count: usize, max_doc_size: ?usize) !ArrayList([]const u8) {
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

        // First, build a list of actual files from corpus_paths
        var all_files = ArrayList([]const u8).init(self.allocator);
        defer {
            for (all_files.items) |path| {
                self.allocator.free(path);
            }
            all_files.deinit();
        }

        // Process each corpus path
        for (self.corpus_paths) |corpus_path| {
            // Check if path is a directory
            var is_dir_path = false;
            {
                var dir_check = std.fs.cwd().openDir(corpus_path, .{}) catch |err| {
                    if (err == error.NotDir) {
                        // It's a file, check if it's a .bin file
                        if (std.mem.endsWith(u8, corpus_path, ".bin")) {
                            try all_files.append(try self.allocator.dupe(u8, corpus_path));
                        }
                    }
                    continue;
                };
                defer dir_check.close();
                is_dir_path = true;
            }

            // If it's a directory, find all .bin files in it
            if (is_dir_path) {
                var dir = try std.fs.cwd().openDir(corpus_path, .{ .iterate = true });
                defer dir.close();

                var iter = dir.iterate();
                while (try iter.next()) |entry| {
                    if (entry.kind == .file) {
                        if (std.mem.endsWith(u8, entry.name, ".bin")) {
                            // Construct full path
                            const full_path = try std.fs.path.join(self.allocator, &[_][]const u8{ corpus_path, entry.name });
                            try all_files.append(full_path);
                        }
                    }
                }
            }
        }

        if (all_files.items.len == 0) {
            if (self.debug) {
                std.debug.print("Warning: No .bin files found in corpus paths\n", .{});
            }
            return error.NoFilesFound;
        }

        if (self.debug) {
            std.debug.print("Found {d} .bin files for sampling\n", .{all_files.items.len});
        }

        for (0..count) |_| {
            // Choose a random file
            const file_idx = self.prng.uintLessThan(usize, all_files.items.len);
            const file_path = all_files.items[file_idx];

            const file = try std.fs.cwd().openFile(file_path, .{});
            defer file.close();

            const file_size = try file.getEndPos();

            // Skip header
            const header_size = 1024;
            const content_size = file_size - @min(file_size, header_size);

            if (content_size == 0) continue;

            const max_chunk_size = max_doc_size orelse (1024 * 1024);
            const chunk_size = @min(max_chunk_size, content_size);

            const max_offset = if (content_size > chunk_size) content_size - chunk_size else 0;
            const offset = if (max_offset > 0) self.prng.uintLessThan(usize, max_offset) else 0;

            try file.seekTo(header_size + offset);

            var content = try self.allocator.alloc(u8, chunk_size);
            errdefer self.allocator.free(content);

            const bytes_read = try file.readAll(content);
            if (bytes_read == 0) {
                self.allocator.free(content);
                continue;
            }

            if (bytes_read < chunk_size) {
                const actual_content = try self.allocator.realloc(content, bytes_read);
                content = actual_content;
            }

            try documents.append(content);
        }

        if (self.debug) {
            var total_size: usize = 0;
            var empty_count: usize = 0;
            for (documents.items) |doc| {
                total_size += doc.len;
                if (doc.len == 0) empty_count += 1;
            }

            std.debug.print("Sampled {d} documents ({d} empty), total size: {d} bytes\n", .{ documents.items.len, empty_count, total_size });
        }

        return documents;
    }

    // Helper function to check if content is valid text
    fn containsValidText(content: []const u8) bool {
        if (content.len == 0) return false;

        // Look for printable ASCII as a simple heuristic
        var printable_count: usize = 0;
        for (content) |byte| {
            if (byte >= 32 and byte < 127) {
                printable_count += 1;
            }
        }

        // If at least 30% is printable text, consider it valid
        return printable_count >= content.len / 3;
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
    candidate_stats: []TokenStats,
    vocab_automaton: BakaCorasick,
    eval_automaton: BakaCorasick,
    current_step: u32,
    loader: ?*fineweb = null,
    document_sampler: *DocumentSampler,
    n_token_ids: u32,
    vocab_size: u32,
    tokenset_contents: []const u8,
    // Parameters
    max_token_length: u32 = 10,
    max_vocab_size: u32,
    top_k_candidates: u32,
    batch_size: u32,
    sample_size: u32,
    processed_files: std.StringHashMap(void),

    // Tracking
    last_full_corpus_scan: u32,
    debug: bool,

    pub fn init(allocator: Allocator, input_tokenset_path: []const u8, corpus_paths: []const []const u8, max_vocab_size: u32, debug: bool) !*VocabLearner {
        var learner = try allocator.create(VocabLearner);

        // Initialize fields
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
            .top_k_candidates = 100,
            .batch_size = 10,
            .sample_size = 3000,
            .processed_files = std.StringHashMap(void).init(allocator),
            .last_full_corpus_scan = 0,
            .debug = debug,
        };

        // Load candidate tokens from tokenset file
        try learner.loadCandidateTokens(input_tokenset_path);

        // Initialize with 256 single-byte tokens
        try learner.initializeWithByteTokens();

        return learner;
    }

    fn deinitBakaCorasick(self: *BakaCorasick, allocator: Allocator) void {
        allocator.free(self.transitions[0..self.capacity]);
        allocator.free(self.info[0..self.capacity]);
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
        deinitBakaCorasick(&self.vocab_automaton, self.allocator);
        deinitBakaCorasick(&self.eval_automaton, self.allocator);

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
        return self.tokenset_contents[start_idx..start_idx + len];
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

    pub fn processCorpus(self: *VocabLearner) !void {
        const start_time = std.time.milliTimestamp();

        if (self.debug) {
            std.debug.print("Processing corpus using raw text conversion...\n", .{});
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

        // Initialize data loader with the files
        var loader = try fineweb.init(self.allocator, corpus_files.items);
        try loader.loadVocabulary("vocab.json");
        self.loader = loader;

        // Setup for token counting
        var combined_automaton = try BakaCorasick.init(self.allocator);
        defer deinitBakaCorasick(&combined_automaton, self.allocator);

        if (self.debug) {
            std.debug.print("Building search automaton with {d} candidate tokens ({d})...\n", .{ self.candidate_stats.len, self.n_token_ids });
        }

        for (0..self.n_token_ids) |id_usize| {
            const id: u32 = @intCast(id_usize);
            const token = self.getTokenStr(id);
            if (token.len > 1) {
                try combined_automaton.insert(token, id);
            }
        }

        try combined_automaton.computeSuffixLinks();

        if (self.debug) {
            const elapsed_sec = @as(f64, @floatFromInt(std.time.milliTimestamp() - start_time)) / 1000.0;
            std.debug.print("Automaton built with {d} states in {d:.2}s\n", .{combined_automaton.len, elapsed_sec});
        }

        const NonoverlappingStats = struct {
            n_nonoverlapping_occurrences: u64,
            next_valid_position: u64,
        };

        const token_id_to_stats = try self.allocator.alloc(NonoverlappingStats, self.n_token_ids);
        defer self.allocator.free(token_id_to_stats);
        @memset(token_id_to_stats, .{.n_nonoverlapping_occurrences = 0, .next_valid_position = 0});
        var position: u64 = 0;
        var tokens_recorded: u64 = 0;

        while (try loader.nextDocumentString()) |text| {
            // Scan text with the automaton
            var current_state: u32 = 0;
            for (text) |byte| {
                current_state = combined_automaton.transitions[current_state][byte];

                // Check if this state represents a match
                {
                    const token_id = combined_automaton.info[current_state].token_id;
                    if (token_id != BakaCorasick.NO_TOKEN) {
                        const token_len = combined_automaton.info[current_state].depth;
                        if (position >= token_id_to_stats[token_id].next_valid_position) {
                            const next_valid_position = position + token_len;
                            token_id_to_stats[token_id].next_valid_position = next_valid_position;
                            token_id_to_stats[token_id].n_nonoverlapping_occurrences += 1;
                            tokens_recorded += 1;
                        }
                    }
                }

                // Check suffix links for additional matches
                var suffix = combined_automaton.info[current_state].green;
                while (suffix != 0) {
                    const token_id = combined_automaton.info[suffix].token_id;
                    if (token_id != BakaCorasick.NO_TOKEN) {
                        const token_len = combined_automaton.info[suffix].depth;
                        if (position >= token_id_to_stats[token_id].next_valid_position) {
                            const next_valid_position = position + token_len;
                            token_id_to_stats[token_id].next_valid_position = next_valid_position;
                            token_id_to_stats[token_id].n_nonoverlapping_occurrences += 1;
                            tokens_recorded += 1;
                        }
                    }
                    suffix = combined_automaton.info[suffix].green;
                }
                position += 1;
            }
        }

        if (self.debug) {
            std.debug.print("Phase 1: Processed {d} tokens from {d} bytes\n", .{ tokens_recorded, position });
        }

        for (self.candidate_stats, token_id_to_stats) |*stats, my_stats| {
            stats.n_nonoverlapping_occurrences = my_stats.n_nonoverlapping_occurrences;
            stats.est_total_savings = @floatFromInt(my_stats.n_nonoverlapping_occurrences * (stats.str_len - 1));
        }

        // Calculate bounds after counting
        // try self.calculateInitialBounds();

        const elapsed_sec = @as(f64, @floatFromInt(std.time.milliTimestamp() - start_time)) / 1000.0;
        if (self.debug) {
            std.debug.print("\nCompleted corpus processing in {d:.2}s\n", .{elapsed_sec});
        }

        // const ThreadContext = struct {
        //     learner: *VocabLearner,
        //     corpus_path: []const u8,
        //     file_idx: usize,
        //     total_files: usize,
        //     automaton: *BakaCorasick,
        //     token_id_to_n_nonoverlapping_occurrences: []usize,
        //     gpt_vocab: *std.AutoHashMap(u16, []const u8),
        //     mutex: *std.Thread.Mutex,
        //     progress_mutex: *std.Thread.Mutex,
        // };

        // var mutex = std.Thread.Mutex{};
        // var progress_mutex = std.Thread.Mutex{};

        // const workerFn = struct {
        //     fn processFile(ctx: ThreadContext) void {
        //         const file_start_time = std.time.milliTimestamp();

        //         // Lock once at the beginning
        //         ctx.progress_mutex.lock();
        //         if (ctx.learner.debug) {
        //             std.debug.print("\n[{d}/{d}] Starting file: {s}\n", .{ ctx.file_idx + 1, ctx.total_files, ctx.corpus_path });
        //         }

        //         // Check if this file has already been processed - use a separate lock
        //         var is_already_processed = false;
        //         ctx.mutex.lock();
        //         if (ctx.learner.processed_files.contains(ctx.corpus_path)) {
        //             is_already_processed = true;
        //         }
        //         ctx.mutex.unlock();

        //         if (is_already_processed) {
        //             if (ctx.learner.debug) {
        //                 std.debug.print("  Skipping previously processed file: {s}\n", .{ctx.corpus_path});
        //             }
        //             ctx.progress_mutex.unlock(); // Unlock before returning
        //             return;
        //         }
        //         ctx.progress_mutex.unlock(); // Unlock after printing debug info

        //         const file = std.fs.cwd().openFile(ctx.corpus_path, .{}) catch |err| {
        //             ctx.progress_mutex.lock();
        //             if (ctx.learner.debug) {
        //                 std.debug.print("  Error opening file: {s}\n", .{@errorName(err)});
        //             }
        //             ctx.progress_mutex.unlock();
        //             return;
        //         };
        //         defer file.close();

        //         // Get file size for progress reporting
        //         const file_size = file.getEndPos() catch |err| {
        //             ctx.progress_mutex.lock();
        //             if (ctx.learner.debug) {
        //                 std.debug.print("  Error getting file size: {s}\n", .{@errorName(err)});
        //             }
        //             ctx.progress_mutex.unlock();
        //             return;
        //         };
        //         const data_size = file_size - @min(file_size, 1024);

        //         ctx.progress_mutex.lock();
        //         if (ctx.learner.debug) {
        //             std.debug.print("  File size: {d:.2} MB\n", .{@as(f64, @floatFromInt(data_size)) / (1024.0 * 1024.0)});
        //         }
        //         ctx.progress_mutex.unlock();

        //         // Skip header if present
        //         if (file_size > 1024) {
        //             file.seekTo(1024) catch return;
        //         }

        //         const CHUNK_SIZE = 4 * 1024 * 1024; // 4MB chunks
        //         var buffer = ctx.learner.allocator.alloc(u8, CHUNK_SIZE) catch return;
        //         defer ctx.learner.allocator.free(buffer);

        //         // Create text buffer for decoded content
        //         var text_buffer = std.ArrayList(u8).init(ctx.learner.allocator);
        //         defer text_buffer.deinit();

        //         var bytes_processed: usize = 0;
        //         var last_progress_time = std.time.milliTimestamp();
        //         const progress_interval_ms = 1000;

        //         while (true) {
        //             const bytes_read = file.readAll(buffer) catch break;
        //             if (bytes_read == 0) break;

        //             // Convert token IDs to text
        //             const complete_tokens = bytes_read / 2;
        //             for (0..complete_tokens) |i| {
        //                 const token_offset = i * 2;
        //                 const token_id = std.mem.bytesToValue(u16, buffer[token_offset..][0..2]);

        //                 // Get string for this token
        //                 if (ctx.gpt_vocab.get(token_id)) |token_str| {
        //                     text_buffer.appendSlice(token_str) catch continue;
        //                 }
        //             }

        //             bytes_processed += bytes_read;

        //             // Progress reporting
        //             const current_time = std.time.milliTimestamp();
        //             if (ctx.learner.debug and current_time - last_progress_time >= progress_interval_ms) {
        //                 const mb_processed = @as(f64, @floatFromInt(bytes_processed)) / (1024.0 * 1024.0);
        //                 const percent_complete = if (data_size > 0)
        //                     @as(f64, @floatFromInt(bytes_processed)) / @as(f64, @floatFromInt(data_size)) * 100.0
        //                 else
        //                     100.0;

        //                 ctx.progress_mutex.lock();
        //                 // Print progress bar: [=====>    ] 45.5%
        //                 std.debug.print("\r[{d}/{d}] ", .{ ctx.file_idx + 1, ctx.total_files });
        //                 std.debug.print("[", .{});
        //                 const bar_width = 20;
        //                 const filled_width = @as(usize, @intFromFloat(@min(percent_complete / 100.0 * @as(f64, @floatFromInt(bar_width)), @as(f64, @floatFromInt(bar_width)))));

        //                 for (0..filled_width) |_| {
        //                     std.debug.print("=", .{});
        //                 }

        //                 if (filled_width < bar_width) {
        //                     std.debug.print(">", .{});
        //                     for (filled_width + 1..bar_width) |_| {
        //                         std.debug.print(" ", .{});
        //                     }
        //                 }

        //                 std.debug.print("] {d:.1}% - {d:.2}/{d:.2} MB", .{ percent_complete, mb_processed, @as(f64, @floatFromInt(data_size)) / (1024.0 * 1024.0) });
        //                 ctx.progress_mutex.unlock();

        //                 last_progress_time = current_time;
        //             }

        //             // Process accumulated text when buffer gets large
        //             if (text_buffer.items.len > CHUNK_SIZE) {
        //                 processText(ctx, text_buffer.items);
        //                 text_buffer.clearRetainingCapacity();
        //             }
        //         }

        //         // Process any remaining text
        //         if (text_buffer.items.len > 0) {
        //             processText(ctx, text_buffer.items);
        //         }

        //         const file_elapsed_sec = @as(f64, @floatFromInt(std.time.milliTimestamp() - file_start_time)) / 1000.0;
        //         ctx.mutex.lock();
        //         const path_copy = ctx.learner.allocator.dupe(u8, ctx.corpus_path) catch {
        //             ctx.mutex.unlock();
        //             return;
        //         };
        //         ctx.learner.processed_files.put(path_copy, {}) catch {
        //             ctx.learner.allocator.free(path_copy);
        //             ctx.mutex.unlock();
        //             return;
        //         };
        //         ctx.mutex.unlock();
        //         if (ctx.learner.debug) {
        //             // Clear line first
        //             std.debug.print("\r                                                                   \r", .{});
        //             std.debug.print("  Completed file {d}/{d}: {s} in {d:.2}s\n", .{ ctx.file_idx + 1, ctx.total_files, ctx.corpus_path, file_elapsed_sec });
        //         }
        //     }

        //     fn processText(ctx: ThreadContext, text: []const u8) void {
        //         // Only do processing if we have text
        //         if (text.len == 0) return;

        //         var max_token_id: u32 = 0;
        //         var id_it = ctx.token_id_to_stats.iterator();
        //         while (id_it.next()) |entry| {
        //             const token_id = entry.key_ptr.*;
        //             if (token_id > max_token_id) {
        //                 max_token_id = token_id;
        //             }
        //         }

        //         var last_used_positions = ctx.learner.allocator.alloc(usize, max_token_id + 1) catch return;
        //         defer ctx.learner.allocator.free(last_used_positions);

        //         var token_counts = ctx.learner.allocator.alloc(u32, max_token_id + 1) catch return;
        //         defer ctx.learner.allocator.free(token_counts);

        //         // Initialize arrays
        //         @memset(last_used_positions, 0);
        //         @memset(token_counts, 0);

        //         // Scan text with the automaton
        //         var current_state: u32 = 0;
        //         for (text, 0..) |byte, pos| {
        //             current_state = ctx.automaton.transitions[current_state][byte];

        //             // Check if this state represents a match
        //             if (ctx.automaton.info[current_state].token_id != BakaCorasick.NO_TOKEN) {
        //                 const token_id = ctx.automaton.info[current_state].token_id;
        //                 if (token_id <= max_token_id) {
        //                     const token_len = ctx.automaton.info[current_state].depth;
        //                     const start_pos = pos + 1 - token_len;

        //                     if (start_pos >= last_used_positions[token_id]) {
        //                         token_counts[token_id] += 1;
        //                         last_used_positions[token_id] = pos + 1;
        //                     }
        //                 }
        //             }

        //             // Check suffix links for additional matches
        //             var suffix = ctx.automaton.info[current_state].green;
        //             while (suffix != 0) {
        //                 if (ctx.automaton.info[suffix].token_id != BakaCorasick.NO_TOKEN) {
        //                     const token_id = ctx.automaton.info[suffix].token_id;
        //                     if (token_id <= max_token_id) {
        //                         const token_len = ctx.automaton.info[suffix].depth;
        //                         const start_pos = pos + 1 - token_len;

        //                         if (start_pos >= last_used_positions[token_id]) {
        //                             token_counts[token_id] += 1;
        //                             last_used_positions[token_id] = pos + 1;
        //                         }
        //                     }
        //                 }
        //                 suffix = ctx.automaton.info[suffix].green;
        //             }
        //         }

        //         ctx.mutex.lock();
        //         defer ctx.mutex.unlock();

        //         for (token_counts, 0..) |count, token_id_usize| {
        //             if (count > 0) {
        //                 const token_id: u32 = @intCast(token_id_usize);
        //                 if (ctx.token_id_to_stats.get(token_id)) |stats| {
        //                     stats.n_nonoverlapping_occurrences += count;
        //                 }
        //             }
        //         }
        //     }
        // }.processFile;

        // const available_cores = try std.Thread.getCpuCount();
        // const num_threads = @min(available_cores, corpus_files.items.len);

        // if (self.debug) {
        //     std.debug.print("Processing {d} files using {d} threads\n", .{ corpus_files.items.len, num_threads });
        // }

        // const batch_size = num_threads;
        // var batch_start: usize = 0;
        // var batch_num: usize = 0;
        // const total_batches = (corpus_files.items.len + batch_size - 1) / batch_size;

        // while (batch_start < corpus_files.items.len) {
        //     batch_num += 1;
        //     const batch_end = @min(batch_start + batch_size, corpus_files.items.len);
        //     const current_batch_size = batch_end - batch_start;

        //     if (self.debug) {
        //         std.debug.print("\nProcessing batch {d}/{d} ({d} files)\n", .{ batch_num, total_batches, current_batch_size });
        //     }

        //     var threads = try self.allocator.alloc(std.Thread, current_batch_size);
        //     defer self.allocator.free(threads);

        //     // Start worker threads
        //     for (batch_start..batch_end) |i| {
        //         const thread_idx = i - batch_start;
        //         const context = ThreadContext{
        //             .learner = self,
        //             .corpus_path = corpus_files.items[i],
        //             .file_idx = i,
        //             .total_files = corpus_files.items.len,
        //             .automaton = &combined_automaton,
        //             .token_id_to_stats = &token_id_to_stats,
        //             .gpt_vocab = &self.gpt_token_to_string,
        //             .mutex = &mutex,
        //             .progress_mutex = &progress_mutex,
        //         };

        //         threads[thread_idx] = try std.Thread.spawn(.{}, workerFn, .{context});
        //     }

        //     // Wait for threads to complete
        //     for (threads) |thread| {
        //         thread.join();
        //     }

        //     // Report batch completion
        //     if (self.debug) {
        //         std.debug.print("Completed batch {d}/{d}\n", .{ batch_num, total_batches });
        //     }

        //     batch_start = batch_end;
        // }
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
    // fn calculateInitialBounds(self: *VocabLearner) !void {
    //     var token_it = self.candidate_stats.iterator();
    //     for (self.candidate_stats, 0..) |*stats, token_id| {

    //         // if (self.vocab_token_ids.contains(token_str)) continue;
    //         // TODO: token_id should exist here
    //         if (self.candidate_stats[token_id].is_in_vocab) continue;

    //         const token_length = stats.str_len;
    //         const occurrences = stats.n_nonoverlapping_occurrences;
    //         stats.max_gain_for_nonoverlapping_occurrences = @as(i32, @intCast(occurrences)) *
    //             @as(i32, @intCast(token_length - 1));

    //         stats.missed_gain_from_superstring_used = 0;

    //         // Look for vocabulary tokens that contain this candidate as substring
    //         for (self.vocab.items) |vocab_token| {
    //             if (vocab_token.len <= 1 or vocab_token.len < token_length) continue;

    //             var count: u32 = 0;
    //             var pos: usize = 0;

    //             // while (pos <= vocab_token.len - token_length) {
    //             //     const found_pos = std.mem.indexOfPos(u8, vocab_token, pos, token_str);
    //             //     if (found_pos == null) break;
    //             //     count += 1;
    //             //     pos = found_pos.? + token_str.len;
    //             // }

    //             if (count > 0) {
    //                 var vocab_token_actual_uses: u32 = 0;

    //                 if (self.candidate_stats.get(vocab_token)) |vocab_stats| {
    //                     vocab_token_actual_uses = vocab_stats.n_nonoverlapping_occurrences;
    //                 }

    //                 const missed = @as(i32, @intCast(count * vocab_token_actual_uses)) *
    //                     @as(i32, @intCast(token_str.len - 1));

    //                 stats.missed_gain_from_superstring_used += missed;
    //             }
    //         }

    //         // Update timestamps
    //         stats.max_gain_for_nonoverlapping_occurrences_computed_at = self.current_step;
    //         stats.missed_gain_from_superstring_used_computed_at = self.current_step;
    //     }
    // }

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

    fn tokenIsAlmostIndependentOfTokens(
        self: *VocabLearner,
        token_id: u32,
        accepted_current_step_tokens: []const u32,
    ) bool {
        _ = self;
        _ = token_id;
        return accepted_current_step_tokens.len == 0;
        // TODO: Implement this function
    }

    pub fn buildVocabulary(self: *VocabLearner) !void {
        const start_time = std.time.milliTimestamp();

        if (self.debug) {
            std.debug.print("Starting vocabulary building process...\n", .{});
        }

        const Context = struct {
            fn lessThan(ctx: *VocabLearner, a: u32, b: u32) std.math.Order {
                const value_a = ctx.candidate_stats[a].getCurrentValueBound();
                const value_b = ctx.candidate_stats[b].getCurrentValueBound();
                return std.math.order(value_b, value_a);
            }
        };

        var heap = std.PriorityQueue(u32, *VocabLearner, Context.lessThan).init(self.allocator, self);
        defer heap.deinit();
        try heap.ensureTotalCapacity(self.n_token_ids);
        for (0..self.n_token_ids) |id_usize| {
            const id: u32 = @intCast(id_usize);
            if (!self.candidate_stats[id].is_in_vocab) {
                try heap.add(id);
            }
        }

        const rejected_current_step_tokens = try self.allocator.alloc(u32, 1000);
        const accepted_current_step_tokens = try self.allocator.alloc(u32, 10);
        const top_k_candidates = try self.allocator.alloc(SampleStats, self.top_k_candidates);

        var candidate_automaton = try BakaCorasick.init(self.allocator);
        defer deinitBakaCorasick(&candidate_automaton, self.allocator);

        var lookbacks = std.ArrayList(u64).init(self.allocator);
        var dp_solution = std.ArrayList(u32).init(self.allocator);
        var matches = std.ArrayList(MatchInfo).init(self.allocator);
        const token_idx_to_least_end_pos = try self.allocator.alloc(u32, self.top_k_candidates);

        while (self.vocab_size < self.max_vocab_size) {
            const max_acceptable = @min(accepted_current_step_tokens.len, self.max_vocab_size - self.vocab_size);
            const iteration_start = std.time.milliTimestamp();
            self.current_step += 1;

            if (self.debug) {
                std.debug.print("\n--- Iteration {d}: Vocabulary size {d}/{d} ---\n", .{ self.current_step, self.vocab_size, self.max_vocab_size });
            }

            var apparent_best_token_id = heap.peek().?;
            while (self.candidate_stats[apparent_best_token_id].sampled_step < self.current_step) {
                for (0..self.top_k_candidates) |i| {
                    top_k_candidates[i] = .{ .token_id = heap.remove() };
                }

                {
                    const automaton_start_time = std.time.milliTimestamp();
                    if (self.debug) {
                        std.debug.print("Creating candidate automaton with just candidate tokens...\n", .{});
                    }

                    // Add candidate tokens with flag
                    for (top_k_candidates, 0..) |stats, my_idx_usize| {
                        const my_idx: u32 = @intCast(my_idx_usize);
                        const token_id = stats.token_id;
                        const token_str = self.getTokenStr(token_id);
                        try candidate_automaton.insert(token_str, my_idx);
                    }

                    try candidate_automaton.computeSuffixLinks();

                    if (self.debug) {
                        const automaton_elapsed_seconds = @as(f64, @floatFromInt(std.time.milliTimestamp() - automaton_start_time)) / 1000.0;
                        std.debug.print("Candidate automaton created with {d} states in {d:.2}s\n", .{candidate_automaton.len, automaton_elapsed_seconds});
                        std.debug.print("Added {d} candidate tokens to automaton\n", .{ top_k_candidates.len });
                    }
                }

                // TODO: sample some documents
                // 2. Sample documents from the corpus
                const sample_size = self.sample_size;
                for (0..sample_size) |_| {
                    const doc = try self.loader.?.nextDocumentStringLoop();
                    try self.evaluateCandidatesOnDocumentDP(
                        top_k_candidates,
                        &candidate_automaton,
                        doc,
                        &lookbacks,
                        &dp_solution,
                        &matches,
                        token_idx_to_least_end_pos,
                    );
                }

                for (top_k_candidates) |sample_stats| {
                    const token_id = sample_stats.token_id;
                    const sampled_occurrences = sample_stats.sampled_occurrences;
                    if (sampled_occurrences >= 5) {
                        const sampled_savings = sample_stats.sampled_savings;
                        const total_occurrences = self.candidate_stats[token_id].n_nonoverlapping_occurrences;
                        const est_savings = @as(f64, @floatFromInt(sampled_savings)) * @as(f64, @floatFromInt(total_occurrences)) / @as(f64, @floatFromInt(sampled_occurrences));
                        // if (self.debug) {
                        //     std.debug.print("token_id: {d}, sampled_occurrences: {d}, sampled_savings: {d}, total_occurrences: {d}, est_savings: {d:.2}\n", .{ token_id, sampled_occurrences, sampled_savings, total_occurrences, est_savings });
                        // }
                        self.candidate_stats[token_id].sampled_occurrences = sampled_occurrences;
                        self.candidate_stats[token_id].sampled_savings = sampled_savings;
                        self.candidate_stats[token_id].est_total_savings = est_savings;
                        self.candidate_stats[token_id].sampled_step = self.current_step;
                    }
                    try heap.add(token_id);
                }

                apparent_best_token_id = heap.peek().?;
            }

            var n_accepted: usize = 0;
            var n_rejected: usize = 0;
            while (n_accepted < max_acceptable and n_rejected < rejected_current_step_tokens.len) {
                apparent_best_token_id = heap.peek().?;
                if (self.candidate_stats[apparent_best_token_id].sampled_step < self.current_step) {
                    break;
                }
                _ = heap.remove();
                if (self.tokenIsAlmostIndependentOfTokens(apparent_best_token_id, accepted_current_step_tokens[0..n_accepted])) {
                    accepted_current_step_tokens[n_accepted] = apparent_best_token_id;
                    n_accepted += 1;
                } else {
                    rejected_current_step_tokens[n_rejected] = apparent_best_token_id;
                    n_rejected += 1;
                }
            }

            for (rejected_current_step_tokens[0..n_rejected]) |token_id| {
                try heap.add(token_id);
            }

            // // 1. Select top candidate tokens based on current value bounds
            // const top_candidates = try self.selectTopCandidates(self.top_k_candidates);
            // defer top_candidates.deinit();


            // // 4. Select tokens for addition (nearly-non-interdependent batch)
            // const tokens_to_add = try self.selectNearlyNonInterdependentBatch(&top_candidates, self.batch_size);
            // defer tokens_to_add.deinit();

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

            // // 6. Periodically remove random tokens
            // if (self.current_step % 500 == 0 and self.vocab_size > 300) {
            //     try self.removeRandomTokens(5);
            // }

            const iteration_elapsed = std.time.milliTimestamp() - iteration_start;
            if (self.debug) {
                std.debug.print("Iteration {d} completed in {d}ms. Vocabulary size: {d}\n", .{ self.current_step, iteration_elapsed, self.vocab_size });
            }

            // if (tokens_to_add.items.len == 0) {
            //     if (self.debug) {
            //         std.debug.print("No tokens added in this iteration. Exiting.\n", .{});
            //     }
            //     break;
            // }
        }

        const elapsed_sec = @as(f64, @floatFromInt(std.time.milliTimestamp() - start_time)) / 1000.0;
        if (self.debug) {
            std.debug.print("\nVocabulary building completed in {d:.2}s. Final vocabulary size: {d}\n", .{ elapsed_sec, self.vocab_size });
        }
    }

    // fn selectTopCandidates(self: *VocabLearner, k: usize) !ArrayList(u32) {
    //     if (true) {
    //         @panic("qq");
    //     }
    //     const start_time = std.time.milliTimestamp();

    //     var candidates = ArrayList(*TokenStats).init(self.allocator);
    //     errdefer candidates.deinit();

    //     const Context = struct {
    //         fn lessThan(_: void, a: *TokenStats, b: *TokenStats) std.math.Order {
    //             const value_a = a.getCurrentValueBound();
    //             const value_b = b.getCurrentValueBound();
    //             return std.math.order(value_a, value_b);
    //         }
    //     };

    //     var heap = std.PriorityQueue(*TokenStats, void, Context.lessThan).init(self.allocator, {});
    //     defer heap.deinit();

    //     var candidates_processed: usize = 0;

    //     for (self.candidate_stats, 0..) |stats, id_usize| {
    //         const id: u32 = @intCast(id_usize);
    //         _ = id;
    //         const value = stats.getCurrentValueBound();

    //         if (heap.count() < k) {
    //             try heap.add(stats);
    //         } else {
    //             const worst = heap.peek() orelse unreachable;
    //             if (value > worst.getCurrentValueBound()) {
    //                 _ = heap.remove();
    //                 try heap.add(stats);
    //             }
    //         }

    //         candidates_processed += 1;
    //     }

    //     // Output debugging info for each token length
    //     for (2..11) |len| {
    //         for (candidates.items) |stats| {
    //             if (stats.token.len == len and stats.getCurrentValueBound() > 0) {
    //                 std.debug.print("Len {d}: value={d}, occurrences={d}, missed_gain={d}\n", .{ len, stats.getCurrentValueBound(), stats.n_nonoverlapping_occurrences, stats.missed_gain_from_superstring_used });
    //                 break;
    //             }
    //         }
    //     }

    //     const heap_size = heap.count();
    //     try candidates.ensureTotalCapacity(heap_size);

    //     while (heap.count() > 0) {
    //         try candidates.append(heap.remove());
    //     }

    //     std.mem.reverse(*TokenStats, candidates.items);

    //     const elapsed_ms = std.time.milliTimestamp() - start_time;
    //     if (self.debug) {
    //         std.debug.print("Selected top {d} candidates from {d} total in {d}ms\n", .{ candidates.items.len, candidates_processed, elapsed_ms });
    //     }

    //     return candidates;
    // }

    // fn evaluateCandidatesOnDocuments(self: *VocabLearner, candidates: *const ArrayList(*TokenStats), documents: []const []const u8) !void {
    //     const start_time = std.time.milliTimestamp();

    //     if (self.debug) {
    //         std.debug.print("Evaluating {d} candidates on {d} documents (parallel)...\n", .{ candidates.items.len, documents.len });
    //     }

    //     // Reset estimated uses for all candidates
    //     for (candidates.items) |stats| {
    //         stats.est_n_uses = 0;
    //     }

    //     var candidate_automaton = try BakaCorasick.init(self.allocator);
    //     defer deinitBakaCorasick(&candidate_automaton, self.allocator);

    //     // Create a mapping from automaton token IDs to candidate stats indices
    //     var token_id_to_index = try self.allocator.alloc(usize, candidates.items.len);
    //     defer self.allocator.free(token_id_to_index);

    //     // Add all candidates to the automaton
    //     for (candidates.items, 0..) |stats, i| {
    //         try candidate_automaton.insert(stats.token, @intCast(i));
    //         token_id_to_index[i] = i;
    //     }

    //     // Compute suffix links for efficient matching
    //     try candidate_automaton.computeSuffixLinks();

    //     const num_threads = @min(10, @max(1, documents.len / 100));

    //     var thread_results = try self.allocator.alloc([]u32, num_threads);
    //     defer self.allocator.free(thread_results);

    //     for (0..num_threads) |i| {
    //         thread_results[i] = try self.allocator.alloc(u32, candidates.items.len);
    //         @memset(thread_results[i], 0);
    //     }
    //     defer {
    //         for (0..num_threads) |i| {
    //             self.allocator.free(thread_results[i]);
    //         }
    //     }

    //     var threads = try self.allocator.alloc(std.Thread, num_threads);
    //     defer self.allocator.free(threads);

    //     const ThreadContext = struct {
    //         automaton: *BakaCorasick,
    //         documents: []const []const u8,
    //         start_doc: usize,
    //         end_doc: usize,
    //         results: []u32,
    //         debug: bool,
    //     };

    //     const workerFn = struct {
    //         fn process(ctx: ThreadContext) void {
    //             for (ctx.start_doc..ctx.end_doc) |doc_idx| {
    //                 const document = ctx.documents[doc_idx];

    //                 var current_state: u32 = 0;
    //                 for (document) |byte| {
    //                     current_state = ctx.automaton.transitions[current_state][byte];

    //                     if (ctx.automaton.info[current_state].token_id != BakaCorasick.NO_TOKEN) {
    //                         const token_id = ctx.automaton.info[current_state].token_id;
    //                         ctx.results[token_id] += 1;
    //                     }

    //                     var suffix = ctx.automaton.info[current_state].green;
    //                     while (suffix != 0) {
    //                         if (ctx.automaton.info[suffix].token_id != BakaCorasick.NO_TOKEN) {
    //                             const token_id = ctx.automaton.info[suffix].token_id;
    //                             ctx.results[token_id] += 1;
    //                         }
    //                         suffix = ctx.automaton.info[suffix].green;
    //                     }
    //                 }
    //             }
    //         }
    //     }.process;

    //     const docs_per_thread = (documents.len + num_threads - 1) / num_threads;
    //     for (0..num_threads) |i| {
    //         const start = i * docs_per_thread;
    //         const end = @min(start + docs_per_thread, documents.len);

    //         if (start >= end) break;

    //         const context = ThreadContext{
    //             .automaton = &candidate_automaton,
    //             .documents = documents,
    //             .start_doc = start,
    //             .end_doc = end,
    //             .results = thread_results[i],
    //             .debug = self.debug,
    //         };

    //         threads[i] = try std.Thread.spawn(.{}, workerFn, .{context});
    //     }

    //     for (0..num_threads) |i| {
    //         if (i * docs_per_thread >= documents.len) break;
    //         threads[i].join();
    //     }

    //     for (candidates.items, 0..) |stats, idx| {
    //         var total_uses: u32 = 0;
    //         for (0..num_threads) |i| {
    //             total_uses += thread_results[i][idx];
    //         }
    //         stats.est_n_uses = total_uses;
    //     }

    //     const elapsed_ms = std.time.milliTimestamp() - start_time;
    //     if (self.debug) {
    //         std.debug.print("Candidate evaluation completed in {d}ms (used {d} threads)\n", .{ elapsed_ms, num_threads });
    //     }
    // }

    // fn selectNearlyNonInterdependentBatch(self: *VocabLearner, candidates: *const ArrayList(u32), batch_size: usize) !ArrayList(u32) {
    //     if (true) {
    //         @panic("qq");
    //     }
    //     var selected = ArrayList(*TokenStats).init(self.allocator);
    //     errdefer selected.deinit();

    //     // Create dependency graph (adjacency list)
    //     var dependencies = std.StringHashMap(std.BufSet).init(self.allocator);
    //     defer {
    //         var it = dependencies.iterator();
    //         while (it.next()) |entry| {
    //             entry.value_ptr.deinit();
    //         }
    //         dependencies.deinit();
    //     }

    //     for (candidates.items) |stats| {
    //         try dependencies.put(stats.token, std.BufSet.init(self.allocator));
    //     }

    //     for (candidates.items) |stats_a| {
    //         for (candidates.items) |stats_b| {
    //             if (std.mem.eql(u8, stats_a.token, stats_b.token)) continue;

    //             const is_prefix = std.mem.startsWith(u8, stats_b.token, stats_a.token);
    //             const is_suffix = std.mem.endsWith(u8, stats_b.token, stats_a.token);
    //             const is_substring = std.mem.indexOf(u8, stats_b.token, stats_a.token) != null;

    //             if (is_prefix or is_suffix or is_substring) {
    //                 const dep_set = dependencies.getPtr(stats_a.token).?;
    //                 try dep_set.insert(stats_b.token);
    //             }
    //         }
    //     }

    //     // Greedy independent set selection
    //     var remaining = std.BufSet.init(self.allocator);
    //     defer remaining.deinit();

    //     // Add all candidates to the remaining set
    //     for (candidates.items) |stats| {
    //         try remaining.insert(stats.token);
    //     }

    //     // Select tokens until we reach batch_size or run out of candidates
    //     while (selected.items.len < batch_size and remaining.count() > 0) {
    //         // Find the best candidate with minimal dependencies
    //         var best_token: ?[]const u8 = null;
    //         var best_value: i32 = 0;
    //         var best_stats: ?*TokenStats = null;

    //         var it = remaining.iterator();
    //         while (it.next()) |token_ptr| {
    //             const token = token_ptr.*;

    //             // Find the stats for this token
    //             for (candidates.items) |stats| {
    //                 if (std.mem.eql(u8, stats.token, token)) {
    //                     const value = stats.getCurrentValueBound(); // Use the proper value metric

    //                     // Check if this is the best candidate so far
    //                     if (best_token == null or value > best_value) {
    //                         best_token = token;
    //                         best_value = value;
    //                         best_stats = stats;
    //                     }
    //                     break;
    //                 }
    //             }
    //         }

    //         if (best_token == null) break;

    //         // Debug: Check if we can get the dependency set
    //         const dep_set_opt = dependencies.get(best_token.?);
    //         if (dep_set_opt) |dep_set| {
    //             // Remove the selected token from remaining
    //             remaining.remove(best_token.?);

    //             // Add the best candidate to the selected set
    //             try selected.append(best_stats.?);

    //             // Remove dependencies from remaining
    //             var dep_it = dep_set.iterator();
    //             while (dep_it.next()) |dep_token_ptr| {
    //                 const dep_token = dep_token_ptr.*;
    //                 if (remaining.contains(dep_token)) {
    //                     remaining.remove(dep_token);
    //                 }
    //             }
    //         } else {
    //             // If we can't find the dependency set, skip this token
    //             remaining.remove(best_token.?);
    //         }
    //     }

    //     if (self.debug) {
    //         std.debug.print("Selected {d} nearly-non-interdependent tokens for addition\n", .{selected.items.len});
    //     }

    //     return selected;
    // }

    // 5. Add selected tokens to vocabulary
    fn addTokensToVocabulary(self: *VocabLearner, tokens: *const ArrayList(u32)) !void {
        if (true) {
            @panic("qq");
        }
        _ = tokens;
        _ = self;
        // const start_time = std.time.milliTimestamp();

        // if (tokens.items.len == 0) return;

        // var sorted_tokens = ArrayList(u32).init(self.allocator);
        // defer sorted_tokens.deinit();
        // try sorted_tokens.appendSlice(tokens.items);

        // const LengthComparator = struct {
        //     pub fn compare(ctx: *VocabLearner, a_id: u32, b_id: u32) bool {
        //         const a = ctx.candidate_stats[a_id];
        //         const b = ctx.candidate_stats[b_id];
        //         return a.str_len < b.str_len;
        //     }
        // };
        // std.sort.pdq(u32, sorted_tokens.items, self, LengthComparator.compare);

        // if (self.debug) {
        //     std.debug.print("Adding {d} tokens to vocabulary:\n", .{sorted_tokens.items.len});
        // }

        // var tokens_added: usize = 0;
        // var tokens_skipped: usize = 0;

        // for (sorted_tokens.items) |token_id| {
        //     const token_str = self.getTokenStr(token_id);
        //     const stats = self.candidate_stats[token_id];

        //     // Skip tokens with zero uses - THIS IS THE KEY CHANGE
        //     if (stats.est_n_uses == 0) {
        //         if (self.debug) {
        //             std.debug.print("  Skipped token: ", .{});
        //             for (token_str) |byte| {
        //                 if (byte >= 32 and byte < 127) {
        //                     std.debug.print("{c}", .{byte});
        //                 } else {
        //                     std.debug.print("\\x{x:0>2}", .{byte});
        //                 }
        //             }
        //             std.debug.print(" (length: {d}, uses: 0)\n", .{token_str.len});
        //         }
        //         tokens_skipped += 1;
        //         continue; // Skip this token completely
        //     }

        //     try self.vocab_automaton.insert(token_str, token_id);
        //     self.addToVocab(token_id);

        //     // Update missed gain values for candidate tokens that are substrings of this token
        //     // var updated_count: usize = 0;
        //     // var candidates_it = self.candidate_stats.iterator();
        //     // while (candidates_it.next()) |entry| {
        //     //     const candidate = entry.key_ptr.*;
        //     //     const candidate_stats = entry.value_ptr.*;

        //     //     if (candidate.len >= token.len) continue;

        //     //     if (std.mem.indexOf(u8, token, candidate) != null) {
        //     //         // Count how many times the candidate appears in the new token
        //     //         var count: u32 = 0;
        //     //         var pos: usize = 0;
        //     //         while (pos <= token.len - candidate.len) {
        //     //             const found_pos = std.mem.indexOfPos(u8, token, pos, candidate);
        //     //             if (found_pos == null) break;
        //     //             count += 1;
        //     //             pos = found_pos.? + candidate.len;
        //     //         }

        //     //         // Update missed gain based on estimated uses of the new token
        //     //         const additional_missed_gain = @as(i32, @intCast(count * stats.est_n_uses)) *
        //     //             @as(i32, @intCast(candidate.len - 1));

        //     //         candidate_stats.missed_gain_from_superstring_used += additional_missed_gain;
        //     //         candidate_stats.missed_gain_from_superstring_used_computed_at = self.current_step;

        //     //         updated_count += 1;
        //     //     }
        //     // }

        //     // if (self.debug and updated_count > 0) {
        //     //     std.debug.print("  Updated missed gain for {d} candidate tokens\n", .{updated_count});
        //     // }


        //     if (self.debug) {
        //         std.debug.print("  Added token {d}: ", .{tokens_added + 1});
        //         for (token_str) |byte| {
        //             if (byte >= 32 and byte < 127) {
        //                 std.debug.print("{c}", .{byte});
        //             } else {
        //                 std.debug.print("\\x{x:0>2}", .{byte});
        //             }
        //         }
        //         std.debug.print(" (length: {d}, uses: {d})\n", .{ token_str.len, stats.est_n_uses });
        //     }

        //     tokens_added += 1;
        // }

        // // Rebuild suffix links for the automaton
        // try self.vocab_automaton.computeSuffixLinks();

        // const elapsed_ms = std.time.milliTimestamp() - start_time;
        // if (self.debug) {
        //     std.debug.print("Added {d} tokens to vocabulary in {d}ms. New vocabulary size: {d}\n", .{ tokens_added, elapsed_ms, self.vocab_size });
        //     if (tokens_skipped > 0) {
        //         std.debug.print("  (Skipped {d} tokens with zero uses)\n", .{tokens_skipped});
        //     }
        // }
    }

    fn removeRandomTokens(self: *VocabLearner, count: usize) !void {
        const start_time = std.time.milliTimestamp();

        if (self.debug) {
            std.debug.print("Randomly removing {d} tokens from vocabulary...\n", .{count});
        }

        if (self.vocab_size < 256) {
            if (self.debug) {
                std.debug.print("Vocabulary size is already below 256, weird!!.\n", .{});
                @panic("oh no!");
            }
            return error.VocabularySizeTooSmall;
        }
        const tokens_to_remove = @min(self.vocab_size - 256, count);

        if (tokens_to_remove <= 0) {
            if (self.debug) {
                std.debug.print("No removable tokens available.\n", .{});
            }
            return;
        }

        if (tokens_to_remove < count) {
            if (self.debug) {
                std.debug.print("Removing {d} tokens instead of {d} requested.\n", .{ tokens_to_remove, count });
            }
        }

        var ids_to_remove = try self.allocator.alloc(u32, tokens_to_remove);
        defer self.allocator.free(ids_to_remove);
        var n_found: usize = 0;
        var can_idx: u32 = 256;
        while (n_found < count) : (can_idx += 1) {
            if (self.candidate_stats[can_idx].is_in_vocab) {
                ids_to_remove[n_found] = can_idx;
                n_found += 1;
            }
        }
        while (can_idx < self.n_token_ids) : (can_idx += 1) {
            if (self.candidate_stats[can_idx].is_in_vocab) {
                const replace_idx = self.document_sampler.prng.uintLessThan(usize, n_found);
                ids_to_remove[replace_idx] = can_idx;
            }
        }

        // Remove the first 'tokens_to_remove' tokens
        for (ids_to_remove) |id| {
            self.removeFromVocab(id);
        }

        self.vocab_automaton.clear();
        for (self.candidate_stats, 0..) |stats, id_usize| {
            if (stats.is_in_vocab) {
                const id: u32 = @intCast(id_usize);
                const slice = self.getTokenStr(id);
                try self.vocab_automaton.insert(slice, id);
            }
        }
        try self.vocab_automaton.computeSuffixLinks();

        const elapsed_ms = std.time.milliTimestamp() - start_time;
        if (self.debug) {
            std.debug.print("Removed {d} tokens in {d}ms. New vocabulary size: {d}\n", .{ tokens_to_remove, elapsed_ms, self.vocab_size });
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

    /// Save vocabulary to a binary file
    pub fn saveVocabulary(self: *VocabLearner, path: []const u8) !void {
        const file = try std.fs.cwd().createFile(path, .{});
        defer file.close();

        // Create and write header
        const header = VocabHeader{
            .magic = VOCAB_MAGIC,
            .version = VOCAB_VERSION,
            .vocab_size = @intCast(self.vocab_size),
            .reserved = [_]u8{0} ** 20,
        };
        try file.writeAll(std.mem.asBytes(&header));

        // Write each token
        for (self.candidate_stats, 0..) |stats, i| {
            if (!stats.is_in_vocab) continue;
            const token_id: u32 = @intCast(i);
            const token_str = self.getTokenStr(token_id);
            const token_length: u32 = @intCast(token_str.len);

            // Write token ID and length (convert to little-endian bytes)
            var id_bytes: [4]u8 = undefined;
            var len_bytes: [4]u8 = undefined;
            std.mem.writeInt(u32, &id_bytes, token_id, .little);
            std.mem.writeInt(u32, &len_bytes, token_length, .little);

            try file.writeAll(&id_bytes);
            try file.writeAll(&len_bytes);

            // Write token content
            try file.writeAll(token_str);
        }

        if (self.debug) {
            std.debug.print("Saved vocabulary with {d} tokens to {s}\n", .{ self.vocab_size, path });
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
            learner.candidate_stats[token_id].is_in_vocab = true;
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
            .vocab_size = @intCast(self.vocab_size),
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
            learner.candidate_stats[token_id].is_in_vocab = true;
            try learner.vocab_automaton.insert(token, token_id);

            token_id += 1;
        }

        try learner.vocab_automaton.computeSuffixLinks();

        if (debug) {
            std.debug.print("Deserialized vocabulary with {d} tokens from buffer\n", .{learner.vocab.items.len});
        }

        return learner;
    }

    fn evaluateCandidatesOnDocumentDP(
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
        try lookbacks_arraylist.appendNTimes(0, document.len+1);
        const lookbacks = lookbacks_arraylist.items;
        dp_solution_arraylist.clearRetainingCapacity();
        try dp_solution_arraylist.appendNTimes(0, document.len+1);
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

        dp_solution[0]=0;
        for (1..dp_solution.len) |i| {
            var entry_minus_one = dp_solution[i-1];
            var mask = lookbacks[i];
            const n_iters = @popCount(mask);
            for (0..n_iters) |_| {
                const lookback = @ctz(mask);
                mask &= mask - 1;
                entry_minus_one = @min(entry_minus_one, dp_solution[i-lookback]);
            }
            dp_solution[i] = entry_minus_one + 1;
        }
        const baseline_cost = dp_solution[dp_solution.len-1];

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
                    var entry_minus_one = dp_solution[i-1];
                    var mask = lookbacks[i];
                    const n_iters = @popCount(mask);
                    for (0..n_iters) |_| {
                        const lookback = @ctz(mask);
                        mask &= mask - 1;
                        entry_minus_one = @min(entry_minus_one, dp_solution[i-lookback]);
                    }
                    if (//match_idx < current_candidate_end_match_idx and
                        matches[match_idx].getEndPos() == i) {
                        entry_minus_one = @min(entry_minus_one, dp_solution[i-current_candidate_len]);
                        match_idx += 1;
                    }
                    dp_solution[i] = entry_minus_one + 1;
                }
                const savings = baseline_cost - dp_solution[dp_solution.len-1];
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


    // fn evaluateCandidatesOnDocumentDP(self: *VocabLearner, candidates: *const ArrayList(u32), documents: []const []const u8) !void {
    //     if (true) {
    //         @panic("qq");
    //     }
    //     const start_time = std.time.milliTimestamp();

    //     if (self.debug) {
    //         std.debug.print("Evaluating {d} candidates on {d} documents using DP...\n", .{ candidates.items.len, documents.len });
    //     }

    //     for (candidates.items) |stats| {
    //         stats.est_n_uses = 0;
    //     }

    //     // Convert binary token IDs to text for proper evaluation
    //     var converted_documents = ArrayList([]const u8).init(self.allocator);
    //     defer {
    //         for (converted_documents.items) |doc| {
    //             self.allocator.free(doc);
    //         }
    //         converted_documents.deinit();
    //     }

    //     for (documents) |doc_data| {
    //         var text_buffer = ArrayList(u8).init(self.allocator);

    //         const complete_tokens = doc_data.len / 2;
    //         for (0..complete_tokens) |i| {
    //             const token_offset = i * 2;
    //             if (token_offset + 2 > doc_data.len) break;

    //             const token_id = std.mem.bytesToValue(u16, doc_data[token_offset..][0..2]);
    //             _ = token_id;

    //             // if (self.gpt_token_to_string.get(token_id)) |token_str| {
    //             //     try text_buffer.appendSlice(token_str);
    //             // }
    //         }

    //         const owned_slice = try text_buffer.toOwnedSlice();
    //         try converted_documents.append(owned_slice);
    //     }

    //     if (converted_documents.items.len == 0) {
    //         if (self.debug) {
    //             std.debug.print("No valid documents for DP evaluation, skipping\n", .{});
    //         }
    //         return;
    //     }

    //     if (self.debug) {
    //         std.debug.print("Creating combined automaton with vocabulary and candidate tokens...\n", .{});
    //     }

    //     var combined_automaton = try BakaCorasick.init(self.allocator);
    //     defer deinitBakaCorasick(&combined_automaton, self.allocator);

    //     // Add vocabulary tokens
    //     for (self.vocab.items, 0..) |token, i| {
    //         try combined_automaton.insert(token, @intCast(i));
    //     }

    //     // Add candidate tokens with flag
    //     for (candidates.items, 0..) |stats, i| {
    //         try combined_automaton.insert(stats.token, CANDIDATE_TOKEN_FLAG | @as(u32, @intCast(i)));
    //     }

    //     try combined_automaton.computeSuffixLinks();

    //     if (self.debug) {
    //         std.debug.print("Combined automaton created with {d} states\n", .{combined_automaton.len});
    //         std.debug.print("Added {d} vocabulary tokens and {d} candidate tokens to automaton\n", .{ self.vocab_size, candidates.items.len });
    //     }

    //     const available_cores = try std.Thread.getCpuCount();
    //     const num_threads = @max(1, available_cores);

    //     // Allocate results array to store token savings counts
    //     var total_tokens_saved = try self.allocator.alloc(u32, candidates.items.len);
    //     defer self.allocator.free(total_tokens_saved);
    //     @memset(total_tokens_saved, 0);

    //     if (self.debug) {
    //         std.debug.print("Using {d} threads for parallel DP evaluation\n", .{num_threads});
    //     }

    //     // Create threads and per-thread results
    //     var threads = try self.allocator.alloc(std.Thread, num_threads);
    //     defer self.allocator.free(threads);

    //     var per_thread_results = try self.allocator.alloc([]u32, num_threads);
    //     defer self.allocator.free(per_thread_results);

    //     for (0..num_threads) |i| {
    //         per_thread_results[i] = try self.allocator.alloc(u32, candidates.items.len);
    //         @memset(per_thread_results[i], 0);
    //     }
    //     defer {
    //         for (0..num_threads) |i| {
    //             self.allocator.free(per_thread_results[i]);
    //         }
    //     }

    //     var progress_mutex = std.Thread.Mutex{};
    //     var total_processed: usize = 0;

    //     const docs_per_thread = (converted_documents.items.len + num_threads - 1) / num_threads;

    //     for (0..num_threads) |thread_idx| {
    //         const start_doc = thread_idx * docs_per_thread;
    //         const end_doc = @min(start_doc + docs_per_thread, converted_documents.items.len);

    //         if (start_doc >= end_doc) continue;

    //         const ThreadContext = struct {
    //             learner: *VocabLearner,
    //             automaton: *BakaCorasick,
    //             candidates: *const ArrayList(*TokenStats),
    //             documents: []const []const u8,
    //             start_doc: usize,
    //             end_doc: usize,
    //             results: []u32,
    //             progress_mutex: *std.Thread.Mutex,
    //             total_processed: *usize,
    //             total_docs: usize,
    //             debug: bool,
    //         };

    //         const context = ThreadContext{
    //             .learner = self,
    //             .automaton = &combined_automaton,
    //             .candidates = candidates,
    //             .documents = converted_documents.items,
    //             .start_doc = start_doc,
    //             .end_doc = end_doc,
    //             .results = per_thread_results[thread_idx],
    //             .progress_mutex = &progress_mutex,
    //             .total_processed = &total_processed,
    //             .total_docs = converted_documents.items.len,
    //             .debug = self.debug,
    //         };

    //         threads[thread_idx] = try std.Thread.spawn(.{}, struct {
    //             fn processDocumentRange(ctx: ThreadContext) void {
    //                 // Process each document in the assigned range
    //                 for (ctx.start_doc..ctx.end_doc) |doc_idx| {
    //                     const document = ctx.documents[doc_idx];

    //                     // Skip empty documents
    //                     if (document.len == 0) continue;

    //                     processDocumentDp(ctx.learner, ctx.automaton, ctx.candidates, document, ctx.results);

    //                     // Update progress
    //                     ctx.progress_mutex.lock();
    //                     ctx.total_processed.* += 1;

    //                     if (ctx.debug) {
    //                         const progress = ctx.total_processed.*;
    //                         printProgressBar(progress, ctx.total_docs, 50);
    //                     }
    //                     ctx.progress_mutex.unlock();
    //                 }
    //             }
    //         }.processDocumentRange, .{context});
    //     }

    //     // Wait for all threads to complete
    //     for (0..num_threads) |i| {
    //         if (i * docs_per_thread < converted_documents.items.len) {
    //             threads[i].join();
    //         }
    //     }

    //     // Combine results from all threads
    //     for (0..candidates.items.len) |candidate_idx| {
    //         for (0..num_threads) |thread_idx| {
    //             total_tokens_saved[candidate_idx] += per_thread_results[thread_idx][candidate_idx];
    //         }
    //     }

    //     if (self.debug) {
    //         std.debug.print("\nParallel processing complete\n", .{});
    //     }

    //     // Update candidate token stats with results
    //     for (candidates.items, 0..) |stats, idx| {
    //         stats.est_n_uses += total_tokens_saved[idx];
    //     }

    //     const elapsed_ms = std.time.milliTimestamp() - start_time;
    //     if (self.debug) {
    //         std.debug.print("\nDP evaluation completed in {d}ms\n", .{elapsed_ms});

    //         // Display tokens with most savings
    //         std.debug.print("\nTop tokens by tokenization savings:\n", .{});

    //         // Count tokens with non-zero savings
    //         var tokens_with_savings: usize = 0;
    //         var total_savings: u32 = 0;
    //         for (candidates.items) |stats| {
    //             if (stats.est_n_uses > 0) {
    //                 tokens_with_savings += 1;
    //                 total_savings += stats.est_n_uses;
    //             }
    //         }

    //         std.debug.print("Tokens with non-zero savings: {d}/{d}\n", .{ tokens_with_savings, candidates.items.len });
    //         std.debug.print("Total token savings across all documents: {d}\n", .{total_savings});

    //         // Display top tokens
    //         var indices = try self.allocator.alloc(usize, candidates.items.len);
    //         defer self.allocator.free(indices);

    //         for (0..candidates.items.len) |i| {
    //             indices[i] = i;
    //         }

    //         const SortContext = struct {
    //             candidates: *const ArrayList(*TokenStats),
    //             pub fn lessThan(ctx: @This(), a: usize, b: usize) bool {
    //                 return ctx.candidates.items[a].est_n_uses > ctx.candidates.items[b].est_n_uses;
    //             }
    //         };

    //         // Initialize the context and call the sort
    //         std.sort.pdq(usize, indices, SortContext{ .candidates = candidates }, SortContext.lessThan);

    //         const show_count = @min(10, candidates.items.len);
    //         for (0..show_count) |i| {
    //             const idx = indices[i];
    //             const stats = candidates.items[idx];
    //             std.debug.print("  {d}. Token: \"", .{i + 1});
    //             for (stats.token) |byte| {
    //                 if (byte >= 32 and byte < 127) {
    //                     std.debug.print("{c}", .{byte});
    //                 } else {
    //                     std.debug.print("\\x{x:0>2}", .{byte});
    //                 }
    //             }
    //             std.debug.print("\" (len={d}), tokens saved: {d}\n", .{ stats.token.len, stats.est_n_uses });
    //         }
    //     }
    // }

    // Core DP processing function for a single document - used by both parallel and sequential paths
    fn processDocumentDp(self: *VocabLearner, automaton: *BakaCorasick, candidates: *const ArrayList(*TokenStats), document: []const u8, results: []u32) void {
        // 1. Identify candidates that appear in this document
        var candidates_in_doc = std.ArrayList(usize).init(self.allocator);
        defer candidates_in_doc.deinit();

        var candidate_positions = self.allocator.alloc(std.ArrayList(usize), candidates.items.len) catch return;
        defer {
            for (candidate_positions) |*pos_list| {
                pos_list.deinit();
            }
            self.allocator.free(candidate_positions);
        }

        // Initialize position lists for each candidate
        for (0..candidates.items.len) |i| {
            candidate_positions[i] = std.ArrayList(usize).init(self.allocator);
        }

        // Find all candidate token matches in the document
        for (candidates.items, 0..) |stats, i| {
            const token = stats.token;
            var pos: usize = 0;

            while (pos <= document.len - token.len) {
                const found_pos = std.mem.indexOfPos(u8, document, pos, token);
                if (found_pos == null) break;

                const match_pos = found_pos.? + token.len;
                candidate_positions[i].append(match_pos) catch break;
                pos = found_pos.? + 1;
            }

            if (candidate_positions[i].items.len > 0) {
                candidates_in_doc.append(i) catch continue;
            }
        }

        if (candidates_in_doc.items.len == 0) return; // No candidates in this document

        // 2. Find vocabulary token positions for DP
        var token_end_positions = std.AutoHashMap(usize, std.ArrayList(TokenMatch)).init(self.allocator);
        defer {
            var it = token_end_positions.iterator();
            while (it.next()) |entry| {
                entry.value_ptr.*.deinit();
            }
            token_end_positions.deinit();
        }

        // Scan document with automaton for vocabulary tokens
        var current_state: u32 = 0;
        for (document, 0..) |byte, pos| {
            current_state = automaton.transitions[current_state][byte];

            if (automaton.info[current_state].token_id != BakaCorasick.NO_TOKEN) {
                const token_id = automaton.info[current_state].token_id;
                const token_len = automaton.info[current_state].depth;
                const end_pos = pos + 1;

                // Only add vocabulary tokens here (not candidates)
                if ((token_id & CANDIDATE_TOKEN_FLAG) == 0) {
                    var matches = token_end_positions.get(end_pos);
                    if (matches == null) {
                        const list = std.ArrayList(TokenMatch).init(self.allocator);
                        token_end_positions.put(end_pos, list) catch continue;
                        matches = token_end_positions.get(end_pos);
                    }

                    matches.?.append(TokenMatch{
                        .len = token_len,
                        .is_candidate = false,
                        .id = token_id,
                    }) catch continue;
                }
            }

            // Check suffix links for additional matches
            var suffix = automaton.info[current_state].green;
            while (suffix != 0) {
                if (automaton.info[suffix].token_id != BakaCorasick.NO_TOKEN) {
                    const token_id = automaton.info[suffix].token_id;
                    const token_len = automaton.info[suffix].depth;
                    const end_pos = pos + 1;

                    // Only add vocabulary tokens
                    if ((token_id & CANDIDATE_TOKEN_FLAG) == 0) {
                        var matches = token_end_positions.get(end_pos);
                        if (matches == null) {
                            const list = std.ArrayList(TokenMatch).init(self.allocator);
                            token_end_positions.put(end_pos, list) catch continue;
                            matches = token_end_positions.get(end_pos);
                        }

                        matches.?.append(TokenMatch{
                            .len = token_len,
                            .is_candidate = false,
                            .id = token_id,
                        }) catch continue;
                    }
                }
                suffix = automaton.info[suffix].green;
            }
        }

        // 3. Solve base DP problem with current vocabulary
        var base_costs = self.allocator.alloc(u32, document.len + 1) catch return;
        defer self.allocator.free(base_costs);

        var base_masks = self.allocator.alloc(u64, document.len + 1) catch return;
        defer self.allocator.free(base_masks);

        // Initialize DP arrays
        for (0..base_costs.len) |i| base_costs[i] = std.math.maxInt(u32);
        base_costs[0] = 0;
        @memset(base_masks, 0);

        // Solve base tokenization using DP
        for (1..document.len + 1) |i| {
            // 1-byte token (always available)
            if (i >= 1 and base_costs[i - 1] + 1 < base_costs[i]) {
                base_costs[i] = base_costs[i - 1] + 1;
            }

            // Check for vocabulary tokens ending at position i
            if (token_end_positions.get(i)) |token_matches| {
                for (token_matches.items) |match| {
                    // Only use vocabulary tokens for base problem
                    if (match.is_candidate) continue;

                    const start = i - match.len;
                    if (base_costs[start] + 1 < base_costs[i]) {
                        base_costs[i] = base_costs[start] + 1;

                        // Set bit for this token length in the mask
                        if (match.len >= 2 and match.len <= 65) {
                            base_masks[i] |= (@as(u64, 1) << @intCast(match.len - 2));
                        }
                    }
                }
            }
        }

        // Store base token count
        const base_token_count = base_costs[document.len];

        // 4. Evaluate each candidate by solving DP problem with it included
        for (candidates_in_doc.items) |candidate_idx| {
            // Skip candidates with no occurrences
            if (candidate_positions[candidate_idx].items.len == 0) continue;

            const token = candidates.items[candidate_idx].token;
            const token_len = token.len;

            // Sort positions in ascending order
            std.sort.pdq(usize, candidate_positions[candidate_idx].items, {}, struct {
                fn compare(_: void, a: usize, b: usize) bool {
                    return a < b;
                }
            }.compare);

            // Find first position where this candidate appears
            const positions = candidate_positions[candidate_idx].items;
            if (positions.len == 0) continue;

            const first_pos = positions[0];
            if (first_pos < token_len) continue;

            const start_pos = first_pos - token_len;

            // Copy costs array for this candidate
            var candidate_costs = self.allocator.dupe(u32, base_costs) catch continue;
            defer self.allocator.free(candidate_costs);

            // Recalculate DP from the first occurrence
            for (start_pos + token_len..document.len + 1) |i| {
                // Check if this position is the end of our candidate token
                var is_candidate_end = false;
                for (positions) |pos| {
                    if (pos == i) {
                        is_candidate_end = true;
                        break;
                    }
                }

                if (is_candidate_end) {
                    // Consider using candidate token
                    const start = i - token_len;
                    if (candidate_costs[start] + 1 < candidate_costs[i]) {
                        candidate_costs[i] = candidate_costs[start] + 1;
                    }
                }

                // Process existing vocab tokens using bitmask
                var mask = base_masks[i];
                while (mask != 0) {
                    const tz = @ctz(mask);
                    const lookback_amt = tz + 2;

                    if (i >= lookback_amt) {
                        const start = i - lookback_amt;
                        if (candidate_costs[start] + 1 < candidate_costs[i]) {
                            candidate_costs[i] = candidate_costs[start] + 1;
                        }
                    }

                    // Clear bit efficiently and continue (mask &= mask - 1)
                    mask &= mask - 1;
                }

                // Always consider 1-byte token
                if (i >= 1 and candidate_costs[i - 1] + 1 < candidate_costs[i]) {
                    candidate_costs[i] = candidate_costs[i - 1] + 1;
                }
            }

            // Calculate tokens saved
            const new_token_count = candidate_costs[document.len];

            if (base_token_count > new_token_count) {
                const tokens_saved = base_token_count - new_token_count;
                results[candidate_idx] += @intCast(tokens_saved);
            }
        }
    }

    fn printProgressBar(progress: usize, total: usize, width: usize) void {
        if (total == 0) return;

        const percent = @as(f64, @floatFromInt(progress)) / @as(f64, @floatFromInt(total));
        const chars_complete = @as(usize, @intFromFloat(percent * @as(f64, @floatFromInt(width))));

        std.debug.print("\r[", .{});
        for (0..width) |i| {
            if (i < chars_complete) {
                std.debug.print("=", .{});
            } else if (i == chars_complete and progress < total) {
                std.debug.print(">", .{});
            } else {
                std.debug.print(" ", .{});
            }
        }

        std.debug.print("] {d:.1}% ({d}/{d})", .{ percent * 100.0, progress, total });
    }
};

fn debugComputeSuffixLinks(allocator: Allocator) !void {
    var a = try BakaCorasick.init(allocator);
    var b = try BakaCorasick.init(allocator);
    try a.insert(" ", 1);
    try b.insert(" ", 1);
    try a.insert("t", 2);
    try b.insert("t", 2);
    try a.insert("h", 3);
    try b.insert("h", 3);
    try a.insert("e", 4);
    try b.insert("e", 4);
    try a.computeSuffixLinks();
    try a.insert(" the", 5);
    try b.insert(" the", 5);
    try a.computeSuffixLinks();
    try b.computeSuffixLinks();
    std.debug.print("a.len={}\n", .{a.len});
    std.debug.print("b.len={}\n", .{b.len});
    for (0..a.len) |i| {
        std.debug.print("{}\n", .{a.info[i]});
        std.debug.print("{}\n", .{b.info[i]});
        std.debug.print("{any}\n", .{a.transitions[i]});
        std.debug.print("{any}\n", .{b.transitions[i]});
        std.debug.print("{}\n", .{std.mem.eql(u32, &a.transitions[i], &b.transitions[i])});
    }
}

pub fn main() !void {
    const allocator = std.heap.c_allocator;

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

    var learner = try VocabLearner.init(allocator, tokenset_path, corpus_paths, 300, debug);
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
