const std = @import("std");
const input_parser = @import("input_parser.zig");
const gpt_encode = @import("gpt_encode.zig");
const time = std.time;
const fs = std.fs;
const Allocator = std.mem.Allocator;

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
    file: ?std.fs.File, // Now optional since we may be between files

    // Multi-file support
    file_paths: [][]const u8, // List of all file paths to process
    current_file_index: usize, // Current file index in file_paths
    current_file_path: []const u8, // Current file path (for easier access)

    // Buffered reader with large buffer
    buffered_reader: ?std.io.BufferedReader(2 * 1024 * 1024, std.fs.File.Reader), // Now optional

    // Direct token lookup array
    token_bytes: []?[]const u8,

    // Document state tracking
    current_document: ?[]usize,
    added_fake_separator: bool,
    reached_end: bool, // Now indicates all files are processed
    header_skipped: bool, // Now per-file state

    // Vocabulary information for rewinding
    vocab_path: ?[]const u8,

    // Main initializer - process multiple files (or a single file)
    pub fn init(allocator: Allocator, file_paths: []const []const u8) !*FinewebDataLoader {
        const start_time = time.nanoTimestamp();

        // Check if there are any files to process
        if (file_paths.len == 0) {
            std.debug.print("[ERROR] No files provided to FinewebDataLoader.init\n", .{});
            return error.NoFilesProvided;
        }

        const self = try allocator.create(FinewebDataLoader);

        // Make a copy of the file paths
        var paths_copy = try allocator.alloc([]const u8, file_paths.len);
        var i: usize = 0;
        while (i < file_paths.len) : (i += 1) {
            paths_copy[i] = try allocator.dupe(u8, file_paths[i]);
        }

        // Allocate token_bytes array
        const token_bytes = try allocator.alloc(?[]const u8, 50257);
        for (token_bytes) |*ptr| {
            ptr.* = null;
        }

        // Initialize with no open file yet
        self.* = .{
            .allocator = allocator,
            .file = null,
            .file_paths = paths_copy,
            .current_file_index = 0,
            .current_file_path = paths_copy[0], // Start with first file
            .buffered_reader = null,
            .token_bytes = token_bytes,
            .current_document = null,
            .added_fake_separator = false,
            .reached_end = false,
            .header_skipped = false,
            .vocab_path = null,
        };

        // Open the first file
        try self.openCurrentFile();

        // Print timing information
        const elapsed = time.nanoTimestamp() - start_time;
        reportFunctionTime("init", elapsed);

        return self;
    }

    // Helper function to open the current file
    fn openCurrentFile(self: *FinewebDataLoader) !void {
        const start_time = time.nanoTimestamp();

        // Close any existing file
        if (self.file) |file| {
            file.close();
            self.file = null;
            self.buffered_reader = null;
        }

        // Open the current file
        if (self.current_file_index < self.file_paths.len) {
            self.current_file_path = self.file_paths[self.current_file_index];

            const file = try std.fs.cwd().openFile(self.current_file_path, .{});
            self.file = file;

            // Initialize the buffered reader
            const buffered = std.io.bufferedReaderSize(2 * 1024 * 1024, file.reader());
            self.buffered_reader = buffered;

            // Reset per-file state
            self.header_skipped = false;

            if (self.current_file_index > 0) {
                std.debug.print("[INFO] Opened next file: {s} (file {d} of {d})\n", .{ self.current_file_path, self.current_file_index + 1, self.file_paths.len });
            }
        } else {
            self.reached_end = true;
        }

        const elapsed = time.nanoTimestamp() - start_time;
        reportFunctionTime("openCurrentFile", elapsed);
    }

    // Move to the next file
    fn moveToNextFile(self: *FinewebDataLoader) !bool {
        const start_time = time.nanoTimestamp();

        // Check if we're already at the end
        if (self.current_file_index >= self.file_paths.len - 1) {
            self.reached_end = true;
            return false;
        }

        // Move to the next file
        self.current_file_index += 1;
        try self.openCurrentFile();

        const elapsed = time.nanoTimestamp() - start_time;
        reportFunctionTime("moveToNextFile", elapsed);

        return true;
    }

    pub fn deinit(self: *FinewebDataLoader) void {
        const start_time = time.nanoTimestamp();

        // Close file if open
        if (self.file) |file| {
            file.close();
        }

        // Free all file paths
        for (self.file_paths) |path| {
            self.allocator.free(path);
        }
        self.allocator.free(self.file_paths);

        for (self.token_bytes) |maybe_str| {
            if (maybe_str) |str| {
                self.allocator.free(str);
            }
        }
        self.allocator.free(self.token_bytes);
        if (self.current_document) |doc| {
            self.allocator.free(doc);
        }

        // Free vocab path if it exists
        if (self.vocab_path) |path| {
            self.allocator.free(path);
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

        // Store the vocabulary path for potential rewinding
        if (self.vocab_path == null) {
            self.vocab_path = try self.allocator.dupe(u8, vocab_path);
        }

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

        // Print timing information
        const elapsed = time.nanoTimestamp() - start_time;
        reportFunctionTime("loadVocabulary", elapsed);
    }

    /// Rewind the data loader to start from the beginning
    pub fn rewind(self: *FinewebDataLoader) !void {
        const start_time = time.nanoTimestamp();

        // Close any open file
        if (self.file) |file| {
            file.close();
            self.file = null;
            self.buffered_reader = null;
        }

        // Reset to the first file
        self.current_file_index = 0;
        try self.openCurrentFile();

        // Reset tracking state
        self.reached_end = false;

        // Free the current document if any
        if (self.current_document) |doc| {
            self.allocator.free(doc);
            self.current_document = null;
        }

        const elapsed = time.nanoTimestamp() - start_time;
        reportFunctionTime("rewind", elapsed);
    }

    /// Skip the file header (first 1024 bytes)
    fn skipHeader(self: *FinewebDataLoader) !void {
        const start_time = time.nanoTimestamp();

        if (self.header_skipped) return;
        if (self.buffered_reader == null) return error.NoFileOpen;

        const HEADER_SIZE = 1024; // Based on our examination of the file

        // Use BufferedReader to skip bytes
        var skip_buffer: [HEADER_SIZE]u8 = undefined;
        const bytes_read = try self.buffered_reader.?.reader().readAll(&skip_buffer);

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
        //const start_time = time.nanoTimestamp();
        if (self.buffered_reader == null) return error.NoFileOpen;

        const result = try self.allocator.alloc(usize, count);
        errdefer self.allocator.free(result);

        var tokens_read: usize = 0;

        // Read bytes in chunks for better performance
        const CHUNK_SIZE = count * 2; // 2 bytes per token
        var byte_buffer = try self.allocator.alloc(u8, CHUNK_SIZE);
        defer self.allocator.free(byte_buffer);

        const bytes_read = try self.buffered_reader.?.reader().readAll(byte_buffer);

        // Process all complete tokens in the buffer
        const complete_tokens = bytes_read / 2;

        var i: usize = 0;
        while (i < complete_tokens) : (i += 1) {
            const token_offset = i * 2;
            const token_id = @as(usize, std.mem.bytesToValue(u16, byte_buffer[token_offset .. token_offset + 2][0..2]));
            result[tokens_read] = token_id;
            tokens_read += 1;
        }

        if (tokens_read == 0) {
            self.allocator.free(result);
            return error.EndOfFile;
        }

        return result[0..tokens_read];
    }

    /// Find the next document in the token stream across multiple files
    pub fn nextDocument(self: *FinewebDataLoader) !?[]usize {
        //const start_time = time.nanoTimestamp();

        if (self.current_document) |doc| {
            self.allocator.free(doc);
            self.current_document = null;
        }

        if (self.reached_end) return null;

        // Make sure a file is open
        if (self.file == null) {
            try self.openCurrentFile();
            if (self.file == null) {
                self.reached_end = true;
                return null;
            }
        }

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

        // ============================================================================
        // !!!! IMPORTANT !!!!
        //
        // This code looks brokenm but actually it's been fixed by setting
        // CHUNK_SIZE = 1.
        //
        // There's no need to fix this code. It does actually read the entirety
        // of each document.
        // ============================================================================

        // Read chunks directly into our document_tokens
        while (!found_document) {
            // Ensure we have capacity for the next chunk
            try document_tokens.ensureUnusedCapacity(CHUNK_SIZE);

            // Handle potential end of file and transition to next file
            var bytes_read: usize = 0;
            if (self.buffered_reader) |*reader| {
                bytes_read = try reader.reader().readAll(byte_buffer);
            }

            if (bytes_read == 0) {
                // End of current file reached
                if (try self.moveToNextFile()) {
                    // Successfully moved to next file, skip header and continue
                    if (!self.header_skipped) {
                        try self.skipHeader();
                    }
                    continue;
                } else if (token_count == 0) {
                    // End of all files and no document started
                    self.reached_end = true;
                    return null;
                } else {
                    // End of all files but we have a partial document
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

        // Create the result
        if (document_tokens.items.len > 0) {
            const result = try document_tokens.toOwnedSlice();
            self.current_document = result;

            return result;
        } else {
            return null;
        }
    }

    /// Get the next document as a string, handling multiple files
    pub fn nextDocumentString(self: *FinewebDataLoader) !?[]u8 {
        const document = try self.nextDocument();
        if (document == null) return null;
        const result = try self.documentToString(document.?);
        return result;
    }

    /// Get the next document as a string, handling multiple files, looping forever
    pub fn nextDocumentStringLoop(self: *FinewebDataLoader) ![]u8 {
        var document_maybe = try self.nextDocumentString();
        if (document_maybe == null) {
            try self.rewind();
        }
        document_maybe = try self.nextDocumentString();
        if (document_maybe) |document| {
            return document;
        }
        return error.YourCorpusHasNoDocuments;
    }

    /// Convert a document (as token IDs) to a string
    pub fn documentToString(self: *FinewebDataLoader, document: []const usize) ![]u8 {
        //const start_time = time.nanoTimestamp();

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
                var len: usize = 1;
                while (len <= max_length) : (len += 1) {
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

    // Get information about current file status
    pub fn getFileStatus(self: *FinewebDataLoader) struct {
        current_file_index: usize,
        total_files: usize,
        current_file_path: []const u8,
        reached_end: bool,
    } {
        return .{
            .current_file_index = self.current_file_index,
            .total_files = self.file_paths.len,
            .current_file_path = self.current_file_path,
            .reached_end = self.reached_end,
        };
    }
};
