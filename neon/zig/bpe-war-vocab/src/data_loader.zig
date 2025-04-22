const std = @import("std");
const input_parser = @import("input_parser.zig");
const gpt_encode = @import("gpt_encode.zig");
const time = std.time;

pub const SEPARATOR_TOKEN_ID: usize = 50256;

pub fn reportFunctionTime(function_name: []const u8, elapsed_ns: i128) void {
    const elapsed_u64 = if (elapsed_ns > 0)
        @as(u64, @intCast(@min(elapsed_ns, std.math.maxInt(u64))))
    else
        0;

    const elapsed_ms = @as(f64, @floatFromInt(elapsed_u64)) / time.ns_per_ms;
    std.debug.print("[TIMING] {s}: {d:.2}ms\n", .{ function_name, elapsed_ms });
}

pub const FinewebDataLoader = struct {
    // File handling
    allocator: std.mem.Allocator,
    file: std.fs.File,
    file_path: []const u8,

    // Buffered reader with large buffer
    buffered_reader: std.io.BufferedReader(2*1024*1024, std.fs.File.Reader),

    // Token handling
    token_map: std.AutoHashMap(usize, []const u8),
    byte_to_token: std.StringHashMap(usize),

    // Direct token lookup array
    token_bytes: []?[]const u8,

    // Document state tracking
    current_document: ?[]usize,
    added_fake_separator: bool,
    reached_end: bool,
    header_skipped: bool,

    pub fn init(allocator: std.mem.Allocator, file_path: []const u8) !*FinewebDataLoader {
        const start_time = time.nanoTimestamp();

        const self = try allocator.create(FinewebDataLoader);

        // Open the file
        const file = try std.fs.cwd().openFile(file_path, .{});

        // Allocate token_bytes array
        const token_bytes = try allocator.alloc(?[]const u8, 50257);
        for (token_bytes) |*ptr| {
            ptr.* = null;
        }

        self.* = .{
            .allocator = allocator,
            .file = file,
            .file_path = try allocator.dupe(u8, file_path),
            .buffered_reader = std.io.bufferedReaderSize(2*1024*1024, file.reader()),
            .token_map = std.AutoHashMap(usize, []const u8).init(allocator),
            .byte_to_token = std.StringHashMap(usize).init(allocator),
            .token_bytes = token_bytes,
            .current_document = null,
            .added_fake_separator = false,
            .reached_end = false,
            .header_skipped = false,
        };

        // Print timing information
        const elapsed = time.nanoTimestamp() - start_time;
        reportFunctionTime("init", elapsed);

        return self;
    }

    pub fn deinit(self: *FinewebDataLoader) void {
        const start_time = time.nanoTimestamp();

        self.file.close();
        self.allocator.free(self.file_path);

        var token_it = self.token_map.iterator();
        while (token_it.next()) |entry| {
            self.allocator.free(entry.value_ptr.*);
        }
        self.token_map.deinit();

        var byte_token_it = self.byte_to_token.iterator();
        while (byte_token_it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }

        self.byte_to_token.deinit();
        self.allocator.free(self.token_bytes);
        if (self.current_document) |doc| {
            self.allocator.free(doc);
        }

        self.allocator.destroy(self);

        // Print timing information
        const elapsed = time.nanoTimestamp() - start_time;
        reportFunctionTime("deinit", elapsed);
    }

    pub const DEFAULT_VOCAB_PATH = "vocab.json";

    /// Load GPT-2 vocabulary from a JSON file
    pub fn loadVocabulary(self: *FinewebDataLoader, vocab_path: []const u8) !void {
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

        self.token_map.clearRetainingCapacity();
        self.byte_to_token.clearRetainingCapacity();

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

            try self.token_map.put(token_id, decoded_bytes);
            const str_copy = try self.allocator.dupe(u8, token_str);
            try self.byte_to_token.put(str_copy, token_id);

            // Store in direct lookup array
            if (token_id < self.token_bytes.len) {
                self.token_bytes[token_id] = decoded_bytes;
            }
        }

        // Print timing information
        const elapsed = time.nanoTimestamp() - start_time;
        reportFunctionTime("loadVocabulary", elapsed);
    }

    /// Skip the file header (first 1024 bytes)
    fn skipHeader(self: *FinewebDataLoader) !void {
        const start_time = time.nanoTimestamp();

        if (self.header_skipped) return;

        const HEADER_SIZE = 1024; // Based on our examination of the file

        // Use BufferedReader to skip bytes
        var skip_buffer: [HEADER_SIZE]u8 = undefined;
        const bytes_read = try self.buffered_reader.reader().readAll(&skip_buffer);

        if (bytes_read < HEADER_SIZE) {
            return error.IncompleteHeader;
        }

        self.header_skipped = true;

        // Print timing information
        const elapsed = time.nanoTimestamp() - start_time;
        reportFunctionTime("skipHeader", elapsed);
    }

    /// Read token IDs from the file (using 2-byte tokens)
    pub fn readTokenIds(self: *FinewebDataLoader, count: usize) ![]usize {
        const start_time = time.nanoTimestamp();

        const result = try self.allocator.alloc(usize, count);
        errdefer self.allocator.free(result);

        var tokens_read: usize = 0;

        // Read bytes in chunks for better performance
        const CHUNK_SIZE = count * 2; // 2 bytes per token
        var byte_buffer = try self.allocator.alloc(u8, CHUNK_SIZE);
        defer self.allocator.free(byte_buffer);

        const bytes_read = try self.buffered_reader.reader().readAll(byte_buffer);

        // Process all complete tokens in the buffer
        const complete_tokens = bytes_read / 2;

        for (0..complete_tokens) |i| {
            const token_offset = i * 2;
            const token_id = @as(usize, std.mem.bytesToValue(u16, byte_buffer[token_offset .. token_offset + 2][0..2]));
            result[tokens_read] = token_id;
            tokens_read += 1;
        }

        if (tokens_read == 0) {
            self.allocator.free(result);
            return error.EndOfFile;
        }

        // Print timing information
        const elapsed = time.nanoTimestamp() - start_time;
        reportFunctionTime("readTokenIds", elapsed);

        return result[0..tokens_read];
    }

    /// Find the next document in the token stream
    pub fn nextDocument(self: *FinewebDataLoader) !?[]usize {
        const start_time = time.nanoTimestamp();

        if (self.current_document) |doc| {
            self.allocator.free(doc);
            self.current_document = null;
        }

        if (self.reached_end) return null;

        if (!self.header_skipped) {
            try self.skipHeader();
        }

        // Create a list to store document tokens
        var document_tokens = std.ArrayList(usize).init(self.allocator);
        defer document_tokens.deinit();

        // Pre-allocate space for tokens
        try document_tokens.ensureTotalCapacity(1000);

        // Keep reading tokens until we find a separator
        var found_document = false;
        var token_count: usize = 0;
        const CHUNK_SIZE = 1;


        var byte_buffer = try self.allocator.alloc(u8, CHUNK_SIZE * 2);
        defer self.allocator.free(byte_buffer);


        // Read chunks directly into our document_tokens
        while (!found_document) {
            // Ensure we have capacity for the next chunk
            try document_tokens.ensureUnusedCapacity(CHUNK_SIZE);

            const bytes_read = try self.buffered_reader.reader().readAll(byte_buffer);
            if (bytes_read == 0) {
                if (token_count == 0) {
                    self.reached_end = true;
                    return null;
                } else {
                    found_document = true;
                    self.reached_end = true;
                    break;
                }
            }

            // Process all complete tokens in the buffer
            const complete_tokens = bytes_read / 2;

            // Process tokens and check for separator
            var i: usize = 0;
            while (i < complete_tokens) : (i += 1) {
                const token_offset = i * 2;
                const token_id = @as(usize, std.mem.bytesToValue(u16, byte_buffer[token_offset .. token_offset + 2][0..2]));

                if (token_id == SEPARATOR_TOKEN_ID) {
                    if (token_count > 0) {
                        // Found end of document
                        found_document = true;
                        break;
                    }
                    // Skip consecutive separators
                    continue;
                }

                // Add token to the document
                try document_tokens.append(token_id);
                token_count += 1;
            }
        }

        std.debug.print("Processed a single document with tokens {}\n", .{token_count});

        // Create the result
        if (document_tokens.items.len > 0) {
            const result = try document_tokens.toOwnedSlice();
            self.current_document = result;

            const elapsed = time.nanoTimestamp() - start_time;
            reportFunctionTime("nextDocument", elapsed);

            return result;
        } else {
            const elapsed = time.nanoTimestamp() - start_time;
            reportFunctionTime("nextDocument", elapsed);

            return null;
        }
    }

    /// Get the next document as a string
    pub fn nextDocumentString(self: *FinewebDataLoader) !?[]u8 {
        const start_time = time.nanoTimestamp();

        const document = try self.nextDocument();
        if (document == null) return null;
        const result = try self.documentToString(document.?);

        // Print timing information
        const elapsed = time.nanoTimestamp() - start_time;
        reportFunctionTime("nextDocumentString", elapsed);

        return result;
    }

    /// Convert a document (as token IDs) to a string
    pub fn documentToString(self: *FinewebDataLoader, document: []const usize) ![]u8 {
        const start_time = time.nanoTimestamp();

        // First pass - calculate total size needed
        var total_size: usize = 0;
        for (document) |token_id| {
            if (token_id < self.token_bytes.len and self.token_bytes[token_id] != null) {
                total_size += self.token_bytes[token_id].?.len;
            } else {
                total_size += 1;
            }
        }

        // Allocate buffer for binary data
        const buffer = try self.allocator.alloc(u8, total_size + 1024);
        errdefer self.allocator.free(buffer);

        // Second pass - fill the buffer
        var pos: usize = 0;
        for (document) |token_id| {
            if (token_id < self.token_bytes.len and self.token_bytes[token_id] != null) {
                const bytes = self.token_bytes[token_id].?;
                @memcpy(buffer[pos .. pos + bytes.len], bytes);
                pos += bytes.len;
            } else {
                // Use placeholder for unknown tokens
                buffer[pos] = '?';
                pos += 1;
            }
        }
        @memset(buffer[pos..], 0);

        // Print timing information
        const elapsed = time.nanoTimestamp() - start_time;
        reportFunctionTime("documentToString", elapsed);

        return buffer[0..pos];
    }

    /// Process documents for BPE learning
    pub fn extractSubstringsFromDocuments(self: *FinewebDataLoader, max_length: usize) !input_parser.TopStringsByLength {
        const start_time = time.nanoTimestamp();

        var result = input_parser.TopStringsByLength.init(self.allocator);
        errdefer result.deinit();

        var doc_count: usize = 0;
        var processed_count: usize = 0;

        while (true) {
            const doc_start_time = time.nanoTimestamp();

            // Get next document
            const document = try self.nextDocumentString();
            if (document == null) break;
            defer self.allocator.free(document.?);

            // Process valid documents
            if (document.?.len > 0) {
                // Extract substrings and count frequencies
                for (1..max_length + 1) |len| {
                    if (len > document.?.len) break;

                    // Process all substrings of current length in batch
                    var start: usize = 0;
                    const end = document.?.len - len + 1;

                    // Process in larger chunks to improve locality
                    const BATCH_SIZE = 65536;
                    while (start < end) {
                        const batch_end = @min(start + BATCH_SIZE, end);

                        // Process batch of substrings
                        var i = start;
                        while (i < batch_end) : (i += 1) {
                            const substring = document.?[i .. i + len];
                            try result.addString(len, substring, 1);
                        }

                        start = batch_end;
                    }
                }

                processed_count += 1;
                if (processed_count % 100 == 0) {
                    std.debug.print("Processed {d} documents\n", .{processed_count});
                }
            }

            doc_count += 1;

            // Print timing for this document iteration
            const doc_elapsed = time.nanoTimestamp() - doc_start_time;
            const function_name = try std.fmt.allocPrint(self.allocator, "extractSubstringsFromDocuments - doc #{d} (size: {d})", .{ doc_count, if (document != null) document.?.len else 0 });
            defer self.allocator.free(function_name); // Important: free the allocated string!
            reportFunctionTime(function_name, doc_elapsed);
        }

        // Sort strings by frequency for BPE
        const sort_start_time = time.nanoTimestamp();
        result.sortByFrequency();
        const sort_elapsed = time.nanoTimestamp() - sort_start_time;
        const sort_function_name = try std.fmt.allocPrint(self.allocator, "extractSubstringsFromDocuments - sorting", .{});
        defer self.allocator.free(sort_function_name);
        reportFunctionTime(sort_function_name, sort_elapsed);

        std.debug.print("Completed processing {d} valid documents out of {d} total\n", .{ processed_count, doc_count });

        // Print timing information
        const elapsed = time.nanoTimestamp() - start_time;
        const total_function_name = try std.fmt.allocPrint(self.allocator, "extractSubstringsFromDocuments - total", .{});
        defer self.allocator.free(total_function_name);
        reportFunctionTime(total_function_name, elapsed);

        return result;
    }
};

/// Example usage for FinewebDataLoader
pub fn showExample() !void {
    const allocator = std.heap.page_allocator;

    // Open the data file
    var loader = try FinewebDataLoader.init(allocator, "fineweb_train_000001.bin");
    defer loader.deinit();

    // Load the vocabulary
    try loader.loadVocabulary("vocab.json");

    // Process for BPE learning
    var substrings = try loader.extractSubstringsFromDocuments(100);
    defer substrings.deinit();

    // Display summary
    std.debug.print("Extracted {d} unique substrings\n", .{substrings.total_count});

    // Show a few examples
    const lengths = try substrings.getAllLengths();
    defer allocator.free(lengths);

    for (lengths) |length| {
        if (substrings.getStringsOfLength(length)) |strings| {
            if (strings.len > 0) {
                std.debug.print("Length {d}: {d} strings, most frequent: '{s}' ({d})\n", .{ length, strings.len, strings[0].content, strings[0].frequency });
            }
        }
    }

    // Report timing information
    try loader.reportTimings();
}
