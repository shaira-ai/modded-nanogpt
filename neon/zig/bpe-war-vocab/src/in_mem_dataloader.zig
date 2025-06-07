const std = @import("std");
const FinewebDataLoader = @import("data_loader.zig").FinewebDataLoader;
const time = std.time;
const fs = std.fs;
const Allocator = std.mem.Allocator;

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
    current_index: usize,
    reached_end: bool,

    // File information for compatibility
    file_paths: [][]const u8,

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

        self.* = .{
            .allocator = allocator,
            .documents = &[_][]const u8{},
            .current_index = 0,
            .reached_end = false,
            .file_paths = paths_copy,
        };

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

        self.allocator.destroy(self);

        const elapsed = time.nanoTimestamp() - start_time;
        reportFunctionTime("InMemoryDataLoader.deinit", elapsed);
    }

    fn loadAllDocuments(self: *InMemoryDataLoader) !void {
        const start_time = time.nanoTimestamp();

        var streaming_loader = try FinewebDataLoader.init(self.allocator, self.file_paths);
        defer streaming_loader.deinit();

        try streaming_loader.loadVocabulary("vocab.json");

        var document_list = std.ArrayList([]const u8).init(self.allocator);
        defer document_list.deinit();

        std.debug.print("[INFO] Loading all documents into memory using streaming loader...\n", .{});

        while (true) {
            const document = try streaming_loader.nextDocumentString();
            if (document == null) break;

            try document_list.append(document.?);
        }

        self.documents = try document_list.toOwnedSlice();

        const elapsed = time.nanoTimestamp() - start_time;
        reportFunctionTime("InMemoryDataLoader.loadAllDocuments", elapsed);

        std.debug.print("[INFO] Loaded {d} documents into memory\n", .{self.documents.len});
    }

    /// Get next document string
    pub fn nextDocumentString(self: *InMemoryDataLoader) !?[]u8 {
        if (self.reached_end) return null;

        const index = self.current_index;
        self.current_index += 1;

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

        const index = self.current_index;
        self.current_index += 1;
        const doc_index = index % self.documents.len;

        // Return a copy of the document
        return try self.allocator.dupe(u8, self.documents[doc_index]);
    }

    /// Rewind to start
    pub fn rewind(self: *InMemoryDataLoader) !void {
        self.current_index = 0;
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
