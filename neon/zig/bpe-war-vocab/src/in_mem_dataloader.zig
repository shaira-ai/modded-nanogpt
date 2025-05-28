const std = @import("std");
const gpt_encode = @import("gpt_encode.zig");
const time = std.time;
const fs = std.fs;
const Allocator = std.mem.Allocator;

pub const SEPARATOR_TOKEN_ID: usize = 50256;

fn reportFunctionTime(function_name: []const u8, elapsed_ns: i128) void {
    const elapsed_u64 = if (elapsed_ns > 0)
        @as(u64, @intCast(@min(elapsed_ns, std.math.maxInt(u64))))
    else
        0;

    const elapsed_ms = @as(f64, @floatFromInt(elapsed_u64)) / time.ns_per_ms;
    std.debug.print("[TIMING] {s}: {d:.2}ms\n", .{ function_name, elapsed_ms });
}

pub const InMemoryDataLoader = struct {
    pub const NEEDS_DEALLOCATION = false;

    allocator: std.mem.Allocator,

    documents: [][]const u8,
    current_index: std.atomic.Value(usize),
    reached_end: bool,

    // File information for compatibility
    file_paths: [][]const u8,

    token_bytes: []?[]const u8,

    pub fn init(allocator: Allocator, file_paths: []const []const u8) !*InMemoryDataLoader {
        const start_time = time.nanoTimestamp();

        if (file_paths.len == 0) {
            std.debug.print("[ERROR] No files provided to InMemoryDataLoader.init\n", .{});
            return error.NoFilesProvided;
        }

        const self = try allocator.create(InMemoryDataLoader);
        errdefer allocator.destroy(self);

        // Make a copy of the file paths
        var paths_copy = try allocator.alloc([]const u8, file_paths.len);
        errdefer allocator.free(paths_copy);

        for (file_paths, 0..) |path, i| {
            paths_copy[i] = try allocator.dupe(u8, path);
        }

        // Allocate token_bytes array for vocabulary
        const token_bytes = try allocator.alloc(?[]const u8, 50257);
        errdefer allocator.free(token_bytes);

        for (token_bytes) |*ptr| {
            ptr.* = null;
        }

        self.* = .{
            .allocator = allocator,
            .documents = &[_][]const u8{},
            .current_index = std.atomic.Value(usize).init(0),
            .reached_end = false,
            .file_paths = paths_copy,
            .token_bytes = token_bytes,
        };

        // Load vocabulary for token conversion
        try self.loadVocabulary("vocab.json");

        // Load all documents into memory
        try self.loadAllDocuments();

        const elapsed = time.nanoTimestamp() - start_time;
        reportFunctionTime("InMemoryDataLoader.init", elapsed);

        return self;
    }

    pub fn deinit(self: *InMemoryDataLoader) void {
        const start_time = time.nanoTimestamp();

        // Free all documents
        for (self.documents) |doc| {
            self.allocator.free(doc);
        }
        self.allocator.free(self.documents);

        // Free file paths
        for (self.file_paths) |path| {
            self.allocator.free(path);
        }
        self.allocator.free(self.file_paths);

        // Free token bytes
        for (self.token_bytes) |maybe_str| {
            if (maybe_str) |str| {
                self.allocator.free(str);
            }
        }
        self.allocator.free(self.token_bytes);

        self.allocator.destroy(self);

        const elapsed = time.nanoTimestamp() - start_time;
        reportFunctionTime("InMemoryDataLoader.deinit", elapsed);
    }

    /// Load GPT-2 vocabulary from a JSON file
    fn loadVocabulary(self: *InMemoryDataLoader, vocab_path: []const u8) !void {
        const start_time = time.nanoTimestamp();

        const vocab_file = try std.fs.cwd().openFile(vocab_path, .{});
        defer vocab_file.close();

        const file_size = try vocab_file.getEndPos();
        const json_buffer = try self.allocator.alloc(u8, file_size);
        defer self.allocator.free(json_buffer);

        const bytes_read = try vocab_file.readAll(json_buffer);
        var parsed = try std.json.parseFromSlice(std.json.Value, self.allocator, json_buffer[0..bytes_read], .{});
        defer parsed.deinit();

        const root = parsed.value;
        if (root != .object) {
            return error.InvalidJsonFormat;
        }

        // Reset all entries to null
        for (self.token_bytes) |*token_entry| {
            token_entry.* = null;
        }

        var it = root.object.iterator();
        while (it.next()) |entry| {
            const token_str = entry.key_ptr.*;
            const token_id = @as(usize, @intCast(entry.value_ptr.integer));

            const decoded_size = gpt_encode.get_decoded_len(token_str);
            const bytes_buffer = try self.allocator.alloc(u8, decoded_size);
            const decoded_bytes = try gpt_encode.decode(bytes_buffer, token_str);

            // Store in direct lookup array
            if (token_id < self.token_bytes.len) {
                self.token_bytes[token_id] = decoded_bytes;
            }
        }

        const elapsed = time.nanoTimestamp() - start_time;
        reportFunctionTime("InMemoryDataLoader.loadVocabulary", elapsed);
    }

    /// Load all documents from all files into memory
    fn loadAllDocuments(self: *InMemoryDataLoader) !void {
        const start_time = time.nanoTimestamp();

        var document_list = std.ArrayList([]const u8).init(self.allocator);
        defer document_list.deinit();

        // Process each file
        for (self.file_paths) |file_path| {
            std.debug.print("[INFO] Loading file into memory: {s}\n", .{file_path});

            const file = try std.fs.cwd().openFile(file_path, .{});
            defer file.close();

            // Create buffered reader
            var buffered_reader = std.io.bufferedReaderSize(2 * 1024 * 1024, file.reader());

            // Skip header (first 1024 bytes)
            const HEADER_SIZE = 1024;
            var skip_buffer: [HEADER_SIZE]u8 = undefined;
            const bytes_read = try buffered_reader.reader().readAll(&skip_buffer);
            if (bytes_read < HEADER_SIZE) {
                return error.IncompleteHeader;
            }

            // Read all documents from this file
            while (true) {
                const document = self.readNextDocumentFromFile(&buffered_reader) catch |err| {
                    if (err == error.EndOfFile) break;
                    return err;
                };

                if (document.len > 0) {
                    try document_list.append(document);
                }
            }
        }

        // Convert to owned slice
        self.documents = try document_list.toOwnedSlice();

        const elapsed = time.nanoTimestamp() - start_time;
        reportFunctionTime("InMemoryDataLoader.loadAllDocuments", elapsed);

        std.debug.print("[INFO] Loaded {d} documents into memory\n", .{self.documents.len});
    }

    /// Read next document from a file and convert to string
    fn readNextDocumentFromFile(self: *InMemoryDataLoader, buffered_reader: anytype) ![]const u8 {
        var document_tokens = std.ArrayList(usize).init(self.allocator);
        defer document_tokens.deinit();

        const CHUNK_SIZE = 1;
        var byte_buffer = try self.allocator.alloc(u8, CHUNK_SIZE * 2);
        defer self.allocator.free(byte_buffer);

        var found_document = false;
        var token_count: usize = 0;

        // Read tokens until we find a separator or end of file
        while (!found_document) {
            const bytes_read = try buffered_reader.reader().readAll(byte_buffer);
            if (bytes_read == 0) {
                if (token_count == 0) {
                    return error.EndOfFile;
                } else {
                    found_document = true;
                    break;
                }
            }

            // Process tokens
            const complete_tokens = bytes_read / 2;
            for (0..complete_tokens) |i| {
                const token_offset = i * 2;
                const token_id = @as(usize, std.mem.bytesToValue(u16, byte_buffer[token_offset .. token_offset + 2][0..2]));

                if (token_id == SEPARATOR_TOKEN_ID) {
                    if (token_count > 0) {
                        found_document = true;
                        break;
                    }
                    continue;
                }

                try document_tokens.append(token_id);
                token_count += 1;
            }
        }

        // Convert tokens to string
        if (document_tokens.items.len > 0) {
            return try self.tokensToString(document_tokens.items);
        } else {
            return try self.allocator.dupe(u8, "");
        }
    }

    /// Convert token IDs to string
    fn tokensToString(self: *InMemoryDataLoader, tokens: []const usize) ![]u8 {
        // First pass - calculate total size needed
        var total_size: usize = 0;
        for (tokens) |token_id| {
            if (token_id < self.token_bytes.len and self.token_bytes[token_id] != null) {
                total_size += self.token_bytes[token_id].?.len;
            } else {
                total_size += 1; // placeholder for unknown tokens
            }
        }

        // Allocate buffer
        const buffer = try self.allocator.alloc(u8, total_size);
        errdefer self.allocator.free(buffer);

        // Second pass - fill the buffer
        var pos: usize = 0;
        for (tokens) |token_id| {
            if (token_id < self.token_bytes.len and self.token_bytes[token_id] != null) {
                const bytes = self.token_bytes[token_id].?;
                @memcpy(buffer[pos .. pos + bytes.len], bytes);
                pos += bytes.len;
            } else {
                buffer[pos] = '?';
                pos += 1;
            }
        }

        return buffer[0..pos];
    }

    /// Get next document string
    pub fn nextDocumentString(self: *InMemoryDataLoader) !?[]u8 {
        if (self.reached_end) return null;

        const index = self.current_index.fetchAdd(1, .monotonic);

        if (index >= self.documents.len) {
            self.reached_end = true;
            return null;
        }

        return try self.allocator.dupe(u8, self.documents[index]);
    }

    pub fn nextDocumentStringLoop(self: *InMemoryDataLoader) ![]u8 {
        if (self.documents.len == 0) {
            return error.YourCorpusHasNoDocuments;
        }

        const index = self.current_index.fetchAdd(1, .monotonic);
        const doc_index = index % self.documents.len;

        // Return a copy of the document
        return try self.allocator.dupe(u8, self.documents[doc_index]);
    }

    /// Rewind to start
    pub fn rewind(self: *InMemoryDataLoader) !void {
        self.current_index.store(0, .monotonic);
        self.reached_end = false;
    }

    /// Get file status (for compatibility)
    pub fn getFileStatus(self: *InMemoryDataLoader) struct {
        current_file_index: usize,
        total_files: usize,
        current_file_path: []const u8,
        reached_end: bool,
    } {
        return .{
            .current_file_index = 0, // All files loaded
            .total_files = self.file_paths.len,
            .current_file_path = if (self.file_paths.len > 0) self.file_paths[0] else "",
            .reached_end = self.reached_end,
        };
    }
};
