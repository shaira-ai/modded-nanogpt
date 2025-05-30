const std = @import("std");
const time = std.time;
const CMS_F = @import("count_min_sketch.zig").CountMinSketch;
const N_LENGTHS = @import("count_min_sketch.zig").N_LENGTHS;
const HashTable = @import("hash_table.zig").HashTable;
const HashValue = @import("hash_table.zig").HashValue;
const fs = std.fs;
const native_endian = @import("builtin").target.cpu.arch.endian();
const huge_pages_plz = @import("huge_pages_plz.zig");
const FixedBufferAllocator = std.heap.FixedBufferAllocator;
const builtin = @import("builtin");
const native_os = builtin.os.tag;
const CorpusMetadata = @import("data_loader.zig").CorpusMetadata;

// File format version for serialization
const FILE_FORMAT_VERSION: u32 = 3;
const EMPTY_USIZE_SLICE: []usize = &[_]usize{};
const EMPTY_BYTE_SLICE: []u8 = &[_]u8{};
const USE_HUGE_PAGES = native_os == .linux and false;

const SerializationHeader = extern struct {
    magic: [4]u8,
    version: u32,
    cms_width: u32,
    cms_depth: u32,
    length: u32,
    top_k: u32,
};

fn get_RHT_POW(top_k: usize) u6 {
    return std.math.log2(top_k) + 2;
}

pub fn StringFrequencyManager(
    comptime cms_width: usize,
    comptime cms_depth: usize,
    comptime MY_LEN: comptime_int,
    comptime top_k: usize,
) type {
    const CMS = CMS_F(cms_width, cms_depth, MY_LEN);
    const num_hashes = CMS.num_hashes;
    return struct {
        pub const CandidateString = struct {
            string: []u8,
            hash: u64,
            cached_bytes: u64,
            guess_count: usize,

            pub fn init(string: []u8, hash: u64, guess_count: usize) CandidateString {
                var cached_bytes: u64 = 0;
                const copy_len = @min(MY_LEN, @sizeOf(u64));
                @memcpy(@as([*]u8, @ptrCast(&cached_bytes)), string[0..copy_len]);
                if (native_endian != .big) {
                    cached_bytes = @byteSwap(cached_bytes);
                }

                return CandidateString{
                    .string = string,
                    .hash = hash,
                    .cached_bytes = cached_bytes,
                    .guess_count = guess_count,
                };
            }

            pub fn lessThan(_: void, a: CandidateString, b: CandidateString) std.math.Order {
                if (a.guess_count < b.guess_count) {
                    return .lt;
                }
                if (a.guess_count > b.guess_count) {
                    return .gt;
                }
                if (b.cached_bytes < a.cached_bytes) {
                    return .lt;
                }
                if (b.cached_bytes > a.cached_bytes) {
                    return .gt;
                }
                return std.mem.order(u8, b.string, a.string);
            }
        };

        allocator: std.mem.Allocator,
        fba: FixedBufferAllocator,
        cms: *CMS,
        length2_counters: []usize,
        length3_counters: []usize,
        length2_used_indices: std.ArrayList(u16),
        length3_used_indices: std.ArrayList(u32),
        cms_is_owned: bool = true,
        // Min heap for tracking top-K strings of my length
        heap: std.PriorityQueue(CandidateString, void, CandidateString.lessThan),
        // Map to track actual counts and positions of candidate strings
        actual_counts: HashTable(get_RHT_POW(top_k)),
        // Global position counter across all documents processed by this manager
        global_position: u64 = 0,
        corpus_metadata: ?*const CorpusMetadata = null,

        const Self = @This();

        pub fn init(allocator: std.mem.Allocator) !*Self {
            const start_time = time.nanoTimestamp();

            const self = try allocator.create(Self);
            errdefer allocator.destroy(self);

            var huge_allocator = allocator;
            var fba: FixedBufferAllocator = FixedBufferAllocator.init(EMPTY_BYTE_SLICE);
            if (USE_HUGE_PAGES) {
                const slab = try huge_pages_plz.allocateHugePages(2 * 1024 * 1024 * 1024);
                fba = FixedBufferAllocator.init(slab);
                huge_allocator = fba.allocator();
            }

            // Create the Count-Min Sketch
            const cms = try CMS.init(huge_allocator);
            errdefer cms.deinit();

            const actual_counts = try HashTable(get_RHT_POW(top_k)).init(huge_allocator);
            var heap = std.PriorityQueue(CandidateString, void, CandidateString.lessThan).init(huge_allocator, {});
            try heap.ensureTotalCapacity(top_k + 420);

            const len2_counters = if (MY_LEN == 2) try allocator.alloc(usize, 0x10000) else EMPTY_USIZE_SLICE;
            const len3_counters = if (MY_LEN == 3) try allocator.alloc(usize, 0x1000000) else EMPTY_USIZE_SLICE;
            @memset(len2_counters, 0);
            @memset(len3_counters, 0);

            const len2_used_indices = std.ArrayList(u16).init(allocator);
            const len3_used_indices = std.ArrayList(u32).init(allocator);

            self.* = .{
                .allocator = allocator,
                .fba = fba,
                .cms = cms,
                .length2_counters = len2_counters,
                .length3_counters = len3_counters,
                .length2_used_indices = len2_used_indices,
                .length3_used_indices = len3_used_indices,
                .heap = heap,
                .actual_counts = actual_counts,
                .global_position = 0,
                .corpus_metadata = null,
                .cms_is_owned = true,
            };
            if (USE_HUGE_PAGES) {
                self.heap.allocator = self.fba.allocator();
            }

            const elapsed = time.nanoTimestamp() - start_time;
            std.debug.print("[TIMING] StringFrequencyManager.init: {d:.2}ms\n", .{@as(f64, @floatFromInt(elapsed)) / time.ns_per_ms});

            return self;
        }

        pub fn deinit(self: *Self) void {
            const start_time = time.nanoTimestamp();
            if (self.cms_is_owned) {
                self.cms.deinit();
            }
            self.allocator.free(self.length2_counters);
            self.allocator.free(self.length3_counters);

            self.length2_used_indices.deinit();
            self.length3_used_indices.deinit();

            while (self.heap.count() > 0) {
                const item = self.heap.remove();
                self.allocator.free(item.string);
            }

            if (USE_HUGE_PAGES) {
                huge_pages_plz.freeHugePages(self.fba.buffer);
            } else {
                self.heap.deinit();
                self.actual_counts.deinit();
            }

            self.allocator.destroy(self);

            const elapsed = time.nanoTimestamp() - start_time;
            std.debug.print("[TIMING] StringFrequencyManager.deinit: {d:.2}ms\n", .{@as(f64, @floatFromInt(elapsed)) / time.ns_per_ms});
        }

        pub fn setCorpusMetadata(self: *Self, metadata: *const CorpusMetadata) void {
            self.corpus_metadata = metadata;
        }

        inline fn length2ToIndex(substring: []const u8) usize {
            return @as(u16, @bitCast(substring[0..2].*));
        }
        inline fn length3ToIndex(substring: []const u8) usize {
            return @as(u24, @bitCast(substring[0..3].*));
        }

        pub fn length2_token_from_index(self: *Self, index: u16) [2]u8 {
            _ = self;
            return @bitCast(index);
        }

        pub fn length3_token_from_index(self: *Self, index: u32) [3]u8 {
            _ = self;
            const index_24: u24 = @intCast(index);
            return @bitCast(index_24);
        }

        // PASS 1: Build the Count-Min Sketch
        pub fn buildCMS(self: *Self, document: []const u8) !void {
            // Process all substrings of all lengths in one pass
            for (0..document.len) |i| {
                if (MY_LEN == 2 and i + 2 <= document.len) {
                    const substring = document[i .. i + 2];
                    const index = length2ToIndex(substring);

                    // Track index only when transitioning from 0 to 1
                    if (self.length2_counters[index] == 0) {
                        try self.length2_used_indices.append(@intCast(index));
                    }
                    self.length2_counters[index] += 1;
                }

                if (MY_LEN == 3 and i + 3 <= document.len) {
                    const substring = document[i .. i + 3];
                    const index = length3ToIndex(substring);

                    // Track index only when transitioning from 0 to 1
                    if (self.length3_counters[index] == 0) {
                        try self.length3_used_indices.append(@intCast(index));
                    }
                    self.length3_counters[index] += 1;
                }

                if (MY_LEN >= 4) {
                    const max_len = @min(MY_LEN, document.len - i);
                    if (max_len >= MY_LEN) {
                        self.cms.addPrefixes(@ptrCast(&document[i]), max_len);
                    }
                }
            }

            self.cms.flush();
        }

        fn processString(
            self: *Self,
            substring: []const u8,
            hash: u64,
            guess_count: usize,
            position: u64,
        ) !void {
            var heap = &self.heap;
            var counts = &self.actual_counts;

            // Check if we're already tracking this string
            const maybe_value_ptr = counts.getPtr(substring, hash);
            if (maybe_value_ptr) |value_ptr| {
                // Apply non-overlapping logic for already tracked strings
                if (position >= value_ptr.next_valid_position) {
                    value_ptr.count += 1;
                    value_ptr.next_valid_position = position + MY_LEN;
                }
                return;
            }
            var new_item = CandidateString.init(@constCast(substring), hash, guess_count);
            if (heap.count() < top_k) {
                //std.debug.print("Adding '{s}' to heap\n", .{substring});
                // Haven't reached capacity yet, add the string
                const str_copy = try self.allocator.dupe(u8, substring);
                errdefer self.allocator.free(str_copy);
                new_item.string = str_copy;

                try heap.add(new_item);
                counts.insertKnownNotPresent(str_copy, hash, 1, position + MY_LEN);
            } else if (CandidateString.lessThan(undefined, heap.peek().?, new_item) == .lt) {
                // Current string has higher estimated count than our minimum
                const evicted = heap.remove();
                _ = counts.deleteKnownPresent(evicted.string, evicted.hash);
                // Add the new string
                const str_copy = evicted.string;
                @memcpy(str_copy, substring);
                new_item.string = str_copy;

                try heap.add(new_item);
                counts.insertKnownNotPresent(str_copy, hash, 1, position + MY_LEN);
            }
        }

        fn flushSecondPass(
            self: *Self,
            document: []const u8,
            hashes: [][num_hashes]u64,
            n_hashes: usize,
            base_position: u64,
        ) !void {
            const prefetch_ahead_amt = 60 / cms_depth;
            for (0..prefetch_ahead_amt) |i| {
                self.actual_counts.prefetch(hashes[i][0]);
                self.cms.prefetch(hashes[i]);
            }
            for (0..n_hashes -| prefetch_ahead_amt) |i| {
                const guess_count = self.cms.queryOne(hashes[i]);
                const doc_position = base_position + i;
                try self.processString(document[i .. i + MY_LEN], hashes[i][0], guess_count, doc_position);
                self.actual_counts.prefetch(hashes[i + prefetch_ahead_amt][0]);
                self.cms.prefetch(hashes[i + prefetch_ahead_amt]);
            }
            for (n_hashes -| prefetch_ahead_amt..n_hashes) |i| {
                const guess_count = self.cms.queryOne(hashes[i]);
                const doc_position = base_position + i;
                try self.processString(document[i .. i + MY_LEN], hashes[i][0], guess_count, doc_position);
            }
        }

        // PASS 2: Identify top-K strings and calculate actual counts
        pub fn processDocumentSecondPass(self: *Self, document: []const u8) !void {
            if (MY_LEN < 4) {
                return;
            }
            const hashes: [][num_hashes]u64 = &self.cms.hashes;
            var n_hashes: usize = 0;
            const max_n_hashes = 10 * 1024;

            for (0..document.len - MY_LEN + 1) |i| {
                if (n_hashes == max_n_hashes) {
                    // we have accumulated max_n_hashes worth of hashes
                    // these correspond to positions i-max_n_hashes to i-1 in the document
                    const batch_start_in_doc = i - max_n_hashes;
                    const batch_start_absolute = self.global_position + batch_start_in_doc;
                    try self.flushSecondPass(document[batch_start_in_doc..], hashes, n_hashes, batch_start_absolute);
                    n_hashes = 0;
                }
                self.cms.getHashes(&hashes[n_hashes], @ptrCast(&document[i]));
                n_hashes += 1;
            }

            if (n_hashes > 0) {
                const batch_start_in_doc = (document.len - MY_LEN + 1) - n_hashes;
                const batch_start_absolute = self.global_position + batch_start_in_doc;
                try self.flushSecondPass(document[batch_start_in_doc..], hashes, n_hashes, batch_start_absolute);
            }

            // Update global position for next document
            self.global_position += document.len;
        }

        pub fn mergeCounts(self: *Self, other: *Self) !void {
            while (other.heap.count() > 0) {
                var item = other.heap.remove();
                const others_string = item.string;
                const other_value = other.actual_counts.deleteKnownPresent(item.string, item.hash);
                const other_count = other_value.count;

                const maybe_value_ptr = self.actual_counts.getPtr(item.string, item.hash);
                if (maybe_value_ptr) |value_ptr| {
                    value_ptr.count += other_count;
                    // Reset next_valid_position since merging invalidates position tracking
                    value_ptr.next_valid_position = 0;
                } else if (self.heap.count() < top_k) {
                    const str_copy = try self.allocator.dupe(u8, item.string);
                    item.string = str_copy;
                    try self.heap.add(item);
                    // Set next_valid_position to 0 since position tracking is no longer relevant
                    self.actual_counts.insertKnownNotPresent(str_copy, item.hash, other_count, 0);
                } else if (CandidateString.lessThan(undefined, self.heap.peek().?, item) == .lt) {
                    const evicted = self.heap.remove();
                    _ = self.actual_counts.deleteKnownPresent(evicted.string, evicted.hash);
                    const str_copy = evicted.string;
                    @memcpy(str_copy, item.string);
                    item.string = str_copy;
                    try self.heap.add(item);
                    self.actual_counts.insertKnownNotPresent(str_copy, item.hash, other_count, 0);
                }
                other.allocator.free(others_string);
            }
        }

        // Save the first pass results to a file
        pub fn saveFirstPassToDisk(self: *Self, file_path: []const u8) !void {
            const start_time = time.nanoTimestamp();

            const file = try fs.cwd().createFile(file_path, .{});
            defer file.close();

            var buffered_writer = std.io.bufferedWriter(file.writer());
            var writer = buffered_writer.writer();

            const header = SerializationHeader{
                .magic = "SFMF".*,
                .version = FILE_FORMAT_VERSION,
                .cms_width = @as(u32, @intCast(cms_width)),
                .cms_depth = @as(u32, @intCast(cms_depth)),
                .length = @as(u32, @intCast(MY_LEN)),
                .top_k = @as(u32, @intCast(top_k)),
            };

            try writer.writeAll(std.mem.asBytes(&header));

            for (0..cms_depth) |i| {
                try writer.writeAll(std.mem.sliceAsBytes(self.cms.counters[i][0..]));
            }
            try writer.writeAll(std.mem.sliceAsBytes(self.length2_counters));
            try writer.writeAll(std.mem.sliceAsBytes(self.length3_counters));
            try buffered_writer.flush();

            const elapsed = time.nanoTimestamp() - start_time;
            std.debug.print("[TIMING] saveFirstPassToDisk: {d:.2}ms\n", .{@as(f64, @floatFromInt(elapsed)) / time.ns_per_ms});
        }

        pub fn loadFirstPassFromDisk(allocator: std.mem.Allocator, file_path: []const u8) !*Self {
            const start_time = time.nanoTimestamp();

            const file = try fs.cwd().openFile(file_path, .{});
            defer file.close();

            var buffered_reader = std.io.bufferedReader(file.reader());
            var reader = buffered_reader.reader();

            var header: SerializationHeader = undefined;
            const header_bytes = try reader.readAll(std.mem.asBytes(&header));

            if (header_bytes < @sizeOf(SerializationHeader)) {
                return error.InvalidFileFormat;
            }
            if (!std.mem.eql(u8, &header.magic, "SFMF")) {
                return error.InvalidMagicNumber;
            }
            if (header.version != FILE_FORMAT_VERSION) {
                return error.UnsupportedVersion;
            }

            const self = try init(allocator);
            errdefer self.deinit();

            for (0..cms_depth) |i| {
                const counters_bytes = try reader.readAll(std.mem.sliceAsBytes(self.cms.counters[i][0..]));
                if (counters_bytes < cms_width * @sizeOf(usize)) {
                    return error.InvalidFileFormat;
                }
            }
            const len2_counters_bytes = try reader.readAll(std.mem.sliceAsBytes(self.length2_counters));
            if (len2_counters_bytes < 65536 * @sizeOf(usize)) {
                return error.InvalidFileFormat;
            }
            const len3_counters_bytes = try reader.readAll(std.mem.sliceAsBytes(self.length3_counters));
            if (len3_counters_bytes < 16777216 * @sizeOf(usize)) {
                return error.InvalidFileFormat;
            }

            const elapsed = time.nanoTimestamp() - start_time;
            std.debug.print("[TIMING] loadFirstPassFromDisk: {d:.2}ms\n", .{@as(f64, @floatFromInt(elapsed)) / time.ns_per_ms});

            return self;
        }

        fn addItemToHeap(self: *Self, item: CandidateString, count: u64) !void {
            if (self.heap.count() < top_k) {
                try self.heap.add(item);
                // Initialize position for small strings to 0
                self.actual_counts.insertKnownNotPresent(item.string, item.hash, count, 0);
            } else if (CandidateString.lessThan(undefined, self.heap.peek().?, item) == .lt) {
                const evicted = self.heap.remove();
                _ = self.actual_counts.deleteKnownPresent(evicted.string, evicted.hash);
                self.allocator.free(evicted.string);
                try self.heap.add(item);
                self.actual_counts.insertKnownNotPresent(item.string, item.hash, count, 0);
            } else {
                self.allocator.free(item.string);
            }
        }

        fn computeStringHash(self: *Self, str: []const u8) u64 {
            _ = self;
            var hasher = std.hash.Wyhash.init(0);
            hasher.update(str);
            return hasher.final();
        }

        pub fn addSmallStringsToHeap(self: *Self) !void {
            // Process length 2 indices
            for (self.length2_used_indices.items) |index| {
                const count = self.length2_counters[index];
                if (count > 0) {
                    const str = try self.allocator.alloc(u8, 2);
                    const token_bytes = self.length2_token_from_index(@intCast(index));
                    @memcpy(str, &token_bytes);

                    const hash = self.computeStringHash(str);
                    const item = CandidateString.init(str, hash, count);

                    // Top-K selection
                    if (self.heap.count() < top_k) {
                        try self.heap.add(item);
                        self.actual_counts.insertKnownNotPresent(item.string, item.hash, 0, 0);
                    } else if (CandidateString.lessThan(undefined, self.heap.peek().?, item) == .lt) {
                        const evicted = self.heap.remove();
                        _ = self.actual_counts.deleteKnownPresent(evicted.string, evicted.hash);
                        self.allocator.free(evicted.string);
                        try self.heap.add(item);
                        self.actual_counts.insertKnownNotPresent(item.string, item.hash, 0, 0);
                    } else {
                        self.allocator.free(item.string);
                    }
                }
                self.length2_counters[index] = 0;
            }
            self.length2_used_indices.clearRetainingCapacity();

            // Process length 3 indices
            for (self.length3_used_indices.items) |index| {
                const count = self.length3_counters[index];
                if (count > 0) {
                    const str = try self.allocator.alloc(u8, 3);
                    const token_bytes = self.length3_token_from_index(@intCast(index));
                    @memcpy(str, &token_bytes);

                    const hash = self.computeStringHash(str);
                    const item = CandidateString.init(str, hash, count);

                    // Top-K selection
                    if (self.heap.count() < top_k) {
                        try self.heap.add(item);
                        self.actual_counts.insertKnownNotPresent(item.string, item.hash, 0, 0);
                    } else if (CandidateString.lessThan(undefined, self.heap.peek().?, item) == .lt) {
                        const evicted = self.heap.remove();
                        _ = self.actual_counts.deleteKnownPresent(evicted.string, evicted.hash);
                        self.allocator.free(evicted.string);
                        try self.heap.add(item);
                        self.actual_counts.insertKnownNotPresent(item.string, item.hash, 0, 0);
                    } else {
                        self.allocator.free(item.string);
                    }
                }
                self.length3_counters[index] = 0;
            }
            self.length3_used_indices.clearRetainingCapacity();
        }

        pub fn processDocumentSecondPassSmallStrings(self: *Self, document: []const u8) !void {
            if (MY_LEN >= 4) return;

            var lookups_attempted: usize = 0;
            var lookups_found: usize = 0;
            var overlaps_skipped: usize = 0;
            var counts_incremented: usize = 0;

            for (0..document.len) |i| {
                // Process length 2
                if (MY_LEN == 2 and i + 2 <= document.len) {
                    const substring = document[i .. i + 2];
                    const hash = self.computeStringHash(substring);
                    const position = self.global_position + i;

                    lookups_attempted += 1;

                    const maybe_value_ptr = self.actual_counts.getPtr(substring, hash);
                    if (maybe_value_ptr) |value_ptr| {
                        lookups_found += 1;

                        if (position >= value_ptr.next_valid_position) {
                            value_ptr.count += 1;
                            value_ptr.next_valid_position = position + MY_LEN;
                            counts_incremented += 1;
                        } else {
                            overlaps_skipped += 1;
                        }
                    }
                }

                if (MY_LEN == 3 and i + 3 <= document.len) {
                    const substring = document[i .. i + 3];
                    const hash = self.computeStringHash(substring);
                    const position = self.global_position + i;

                    lookups_attempted += 1;

                    const maybe_value_ptr = self.actual_counts.getPtr(substring, hash);
                    if (maybe_value_ptr) |value_ptr| {
                        lookups_found += 1;

                        if (position >= value_ptr.next_valid_position) {
                            value_ptr.count += 1;
                            value_ptr.next_valid_position = position + MY_LEN;
                            counts_incremented += 1;
                        } else {
                            overlaps_skipped += 1;
                        }
                    }
                }
            }

            const doc_num = self.global_position / 1000;
            if (lookups_found > 0 and doc_num % 10000 == 0) {
                std.debug.print("Doc {d}: attempted={d}, found={d}, incremented={d}, overlaps_skipped={d}\n", .{ doc_num, lookups_attempted, lookups_found, counts_incremented, overlaps_skipped });
            }

            // Update global position for next document
            self.global_position += document.len;
        }

        // Get results and calculate error statistics
        pub fn getResults(self: *Self) !void {
            const start_time = time.nanoTimestamp();

            std.debug.print("\n=== FREQUENCY ANALYSIS RESULTS ===\n", .{});

            var total_strings: usize = 0;

            var heap = &self.heap;
            var counts = &self.actual_counts;

            const count = heap.count();
            total_strings += count;

            std.debug.print("\nLength {d}: {d} strings\n", .{ MY_LEN, count });

            var results = std.ArrayList(CandidateString).init(self.allocator);
            defer results.deinit();

            while (heap.count() > 0) {
                try results.append(heap.remove());
            }

            std.debug.print("Top strings of length {d}:\n", .{MY_LEN});

            var total_error: f64 = 0;
            var total_error_pct: f64 = 0;

            const display_count = @min(1000000, results.items.len);

            var i: usize = results.items.len;
            var displayed: usize = 0;

            while (i > 0 and displayed < display_count) {
                i -= 1;
                const item = results.items[i];
                var actual: usize = item.guess_count;
                if (item.string.len >= 4) {
                    if (counts.getPtr(item.string, item.hash)) |value_ptr| {
                        actual = @as(usize, @intCast(value_ptr.count));
                    }
                } else {
                    if (counts.getPtr(item.string, item.hash)) |value_ptr| {
                        actual = @as(usize, @intCast(value_ptr.count));
                    }
                }

                const abs_error = if (item.guess_count > actual)
                    item.guess_count - actual
                else
                    actual - item.guess_count;

                const error_pct = if (actual > 0)
                    @as(f64, @floatFromInt(abs_error)) / @as(f64, @floatFromInt(actual)) * 100.0
                else
                    0.0;

                std.debug.print("  {d}. '{s}': est={d}, actual={d}, error={d:.2}%\n", .{ displayed + 1, item.string, item.guess_count, actual, error_pct });

                total_error += @as(f64, @floatFromInt(abs_error));
                total_error_pct += error_pct;
                displayed += 1;

                // Add back to heap to restore original state
                try heap.add(item);
            }

            if (displayed > 0) {
                const avg_error_pct = total_error_pct / @as(f64, @floatFromInt(displayed));
                std.debug.print("  Average error for length {d}: {d:.2}%\n", .{ MY_LEN, avg_error_pct });
            }

            std.debug.print("\nTotal unique strings tracked: {d}\n", .{total_strings});

            const elapsed = time.nanoTimestamp() - start_time;
            std.debug.print("[TIMING] getResults: {d:.2}ms\n", .{@as(f64, @floatFromInt(elapsed)) / time.ns_per_ms});
        }

        // Save tokens with their non-overlapping occurrence counts in interleaved format
        pub fn saveTokensToBinaryFormat(self: *Self, file_path: []const u8) !void {
            const start_time = time.nanoTimestamp();

            // Create arrays to store results by length
            var length_counts = [_]u32{0} ** 257;
            var tokens_by_length = std.ArrayList(std.ArrayList([]const u8)).init(self.allocator);
            defer tokens_by_length.deinit();
            var counts_by_length = std.ArrayList(std.ArrayList(u64)).init(self.allocator);
            defer counts_by_length.deinit();

            // Initialize arrays for each length
            for (0..257) |_| {
                try tokens_by_length.append(std.ArrayList([]const u8).init(self.allocator));
                try counts_by_length.append(std.ArrayList(u64).init(self.allocator));
            }
            defer {
                for (tokens_by_length.items) |*list| {
                    list.deinit();
                }
                for (counts_by_length.items) |*list| {
                    list.deinit();
                }
            }

            // Extract all tokens from heap and organize by length
            var heap_items = std.ArrayList(CandidateString).init(self.allocator);
            defer heap_items.deinit();

            while (self.heap.count() > 0) {
                try heap_items.append(self.heap.remove());
            }

            // Process tokens and group by length
            for (heap_items.items) |item| {
                const length = item.string.len;
                length_counts[length] += 1;

                try tokens_by_length.items[length].append(item.string);

                // Get the actual non-overlapping count
                var actual_count: u64 = @as(u64, @intCast(item.guess_count));
                if (self.actual_counts.getPtr(item.string, item.hash)) |value_ptr| {
                    actual_count = value_ptr.count;
                }
                try counts_by_length.items[length].append(actual_count);

                // Restore to heap
                try self.heap.add(item);
            }

            // Write to file in interleaved format
            const file = try fs.cwd().createFile(file_path, .{});
            defer file.close();

            var buffered_writer = std.io.bufferedWriter(file.writer());
            var writer = buffered_writer.writer();

            const format_version: u32 = 3;
            const version_bytes = std.mem.toBytes(format_version);
            try writer.writeAll(&version_bytes);

            // Write header (256 u32 values for token counts by length)
            for (0..256) |i| {
                const count_bytes = std.mem.toBytes(length_counts[i]);
                try writer.writeAll(&count_bytes);
            }

            // Write corpus metadata section
            if (self.corpus_metadata) |metadata| {
                // Write metadata
                const file_count_bytes = std.mem.toBytes(metadata.file_count);
                try writer.writeAll(&file_count_bytes);

                const timestamp_bytes = std.mem.toBytes(metadata.timestamp);
                try writer.writeAll(&timestamp_bytes);

                const hash_seed_bytes = std.mem.toBytes(metadata.hash_seed);
                try writer.writeAll(&hash_seed_bytes);

                // Write file hashes
                for (metadata.file_hashes) |hash| {
                    const hash_bytes = std.mem.toBytes(hash);
                    try writer.writeAll(&hash_bytes);
                }

                std.debug.print("[SFM] Wrote corpus metadata: {d} files, timestamp: {d}\n", .{ metadata.file_count, metadata.timestamp });
            } else {
                // Write empty metadata
                const empty_u32: u32 = 0;
                const empty_u64: u64 = 0;

                const file_count_bytes = std.mem.toBytes(empty_u32);
                try writer.writeAll(&file_count_bytes);

                const timestamp_bytes = std.mem.toBytes(empty_u64);
                try writer.writeAll(&timestamp_bytes);

                const hash_seed_bytes = std.mem.toBytes(empty_u64);
                try writer.writeAll(&hash_seed_bytes);

                std.debug.print("[SFM] No corpus metadata available, wrote empty metadata section\n", .{});
            }

            // Write interleaved data organized by length groups
            var total_written: usize = 0;
            for (1..256) |length| {
                if (length_counts[length] > 0) {
                    const tokens = &tokens_by_length.items[length];
                    const counts = &counts_by_length.items[length];

                    for (tokens.items, 0..) |token, i| {
                        // Write token bytes
                        try writer.writeAll(token);

                        // Write occurrence count (8 bytes, u64)
                        const count_bytes = std.mem.toBytes(counts.items[i]);
                        try writer.writeAll(&count_bytes);

                        total_written += 1;
                    }
                }
            }

            try buffered_writer.flush();

            const elapsed = time.nanoTimestamp() - start_time;
            std.debug.print("[TIMING] saveTokensToBinaryFormat: {d:.2}ms\n", .{@as(f64, @floatFromInt(elapsed)) / time.ns_per_ms});
            std.debug.print("Saved {d} tokens with non-overlapping counts to {s}\n", .{ heap_items.items.len, file_path });
        }
    };
}
