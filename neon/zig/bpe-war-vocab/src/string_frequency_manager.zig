const std = @import("std");
const time = std.time;
const CMS_F = @import("count_min_sketch.zig").CountMinSketch;
const N_LENGTHS = @import("count_min_sketch.zig").N_LENGTHS;
const MY_LEN = @import("count_min_sketch.zig").MY_LEN;
const HashTable = @import("hash_table.zig").HashTable;
const fs = std.fs;
const native_endian = @import("builtin").target.cpu.arch.endian();

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

// File format version for serialization
const FILE_FORMAT_VERSION: u32 = 1;

const SerializationHeader = extern struct {
    magic: [4]u8, // "SFMF" --> String Frequency Manager File
    version: u32,
    cms_width: u32,
    cms_depth: u32,
    min_length: u32,
    max_length: u32,
    top_k: u32,
};

fn get_RHT_POW(top_k: usize) u6 {
    return std.math.log2(top_k) + 2;
}

pub fn StringFrequencyManager(
    comptime cms_width: usize,
    comptime cms_depth: usize,
    comptime min_length: usize,
    comptime max_length: usize,
    comptime top_k: usize,
) type {
    const CMS = CMS_F(cms_width, cms_depth);
    const num_hashes = CMS.num_hashes;
    return struct {
        allocator: std.mem.Allocator,
        cms: *CMS,
        length2_counters: []usize,
        length3_counters: []usize,
        cms_is_owned: bool = true,
        // Min heap for tracking top-K strings of my length
        heap: std.PriorityQueue(CandidateString, void, CandidateString.lessThan),
        // Map to track actual counts of candidate strings
        actual_counts: HashTable(get_RHT_POW(top_k)),

        const Self = @This();

        pub fn init(allocator: std.mem.Allocator) !*Self {
            const start_time = time.nanoTimestamp();

            const self = try allocator.create(Self);
            errdefer allocator.destroy(self);

            // Create the Count-Min Sketch
            const cms = try CMS.init(allocator);
            errdefer cms.deinit();

            // Initialize heaps and count maps
            // const heaps = try allocator.alloc(std.PriorityQueue(CandidateString, void, CandidateString.lessThan), max_length + 1);
            // for (heaps) |*heap| {
            //     heap.* = std.PriorityQueue(CandidateString, void, CandidateString.lessThan).init(allocator, {});
            // }
            // const actual_counts = try allocator.alloc(std.StringHashMap(usize), max_length + 1);
            // for (actual_counts) |*map| {
            //     map.* = std.StringHashMap(usize).init(allocator);
            // }
            const len2_counters = try allocator.alloc(usize, 0x10000);
            const len3_counters = try allocator.alloc(usize, 0x1000000);
            @memset(len2_counters, 0);
            @memset(len3_counters, 0);

            self.* = .{
                .allocator = allocator,
                .cms = cms,
                .length2_counters = len2_counters,
                .length3_counters = len3_counters,
                .heap = std.PriorityQueue(CandidateString, void, CandidateString.lessThan).init(allocator, {}),
                .actual_counts = try HashTable(get_RHT_POW(top_k)).init(allocator),
            };

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
            while (self.heap.count() > 0) {
                const item = self.heap.remove();
                self.allocator.free(item.string);
            }
            self.heap.deinit();

            self.actual_counts.deinit();
            self.allocator.destroy(self);

            const elapsed = time.nanoTimestamp() - start_time;
            std.debug.print("[TIMING] StringFrequencyManager.deinit: {d:.2}ms\n", .{@as(f64, @floatFromInt(elapsed)) / time.ns_per_ms});
        }

        inline fn length2ToIndex(substring: []const u8) usize {
            return @as(u16, @bitCast(substring[0..2].*));
        }
        inline fn length3ToIndex(substring: []const u8) usize {
            return @as(u24, @bitCast(substring[0..3].*));
        }

        pub fn length2_token_from_index(self: *Self, index: u16) [2]u8 {
            _ = self;
            var token: [2]u8 = undefined;
            token[0] = @intCast((index >> 8) & 0xFF);
            token[1] = @intCast(index & 0xFF);
            return token;
        }

        pub fn length3_token_from_index(self: *Self, index: u32) [3]u8 {
            _ = self;
            var token: [3]u8 = undefined;
            token[0] = @intCast((index >> 16) & 0xFF);
            token[1] = @intCast((index >> 8) & 0xFF);
            token[2] = @intCast(index & 0xFF);
            return token;
        }

        // PASS 1: Build the Count-Min Sketch
        pub fn buildCMS(self: *Self, document: []const u8) !void {
            //const start_time = time.nanoTimestamp();

            // Process all substrings of all lengths in one pass
            for (0..document.len) |i| {
                if (MY_LEN == 2 and i + 2 <= document.len) {
                    const substring = document[i .. i + 2];
                    const index = length2ToIndex(substring);
                    self.length2_counters[index] += 1;
                }

                if (MY_LEN == 3 and i + 3 <= document.len) {
                    const substring = document[i .. i + 3];
                    const index = length3ToIndex(substring);
                    self.length3_counters[index] += 1;
                }

                if (MY_LEN >= 4) {
                    const max_len = @min(max_length, document.len - i);
                    if (max_len >= MY_LEN) {
                        self.cms.addPrefixes(@ptrCast(&document[i]), max_len);
                    }
                }
            }

            self.cms.flush();

            //const elapsed = time.nanoTimestamp() - start_time;
            //std.debug.print("[TIMING] buildCMS ({d} bytes): {d:.2}ms\n", .{ document.len, @as(f64, @floatFromInt(elapsed)) / time.ns_per_ms });
        }

        fn processString(
            self: *Self,
            substring: []const u8,
            hash: u64,
            guess_count: usize,
        ) !void {
            var heap = &self.heap;
            var counts = &self.actual_counts;

            // Check if we're already tracking this string
            const maybe_value_ptr = counts.getPtr(substring, hash);
            if (maybe_value_ptr) |value_ptr| {
                //std.debug.print("Already tracking '{s}'\n", .{substring});
                // Already tracking, just increment the actual count
                value_ptr.* += 1;
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
                counts.insertKnownNotPresent(str_copy, hash, 1);
            } else if (CandidateString.lessThan(undefined, heap.peek().?, new_item) == .lt) {
                // Current string has higher estimated count than our minimum
                const evicted = heap.remove();
                _ = counts.deleteKnownPresent(evicted.string, evicted.hash);
                // Add the new string
                const str_copy = evicted.string;
                @memcpy(str_copy, substring);
                new_item.string = str_copy;

                try heap.add(new_item);
                counts.insertKnownNotPresent(str_copy, hash, 1);
            }
        }

        fn flushSecondPass(
            self: *Self,
            document: []const u8,
            hashes: [][num_hashes]u64,
            n_hashes: usize,
        ) !void {
            const prefetch_ahead_amt = 60 / cms_depth;
            for (0..prefetch_ahead_amt) |i| {
                self.actual_counts.prefetch(hashes[i][0]);
                self.cms.prefetch(hashes[i]);
            }
            for (0..n_hashes -| prefetch_ahead_amt) |i| {
                const guess_count = self.cms.queryOne(hashes[i]);
                try self.processString(document[i .. i + MY_LEN], hashes[i][0], guess_count);
                self.actual_counts.prefetch(hashes[i + prefetch_ahead_amt][0]);
                self.cms.prefetch(hashes[i + prefetch_ahead_amt]);
            }
            for (n_hashes -| prefetch_ahead_amt..n_hashes) |i| {
                const guess_count = self.cms.queryOne(hashes[i]);
                try self.processString(document[i .. i + MY_LEN], hashes[i][0], guess_count);
            }
        }

        // PASS 2: Identify top-K strings and calculate actual counts
        pub fn processDocumentSecondPass(self: *Self, document: []const u8) !void {
            if (MY_LEN < 4) {
                return;
            }
            const hashes: [][num_hashes]u64 = &self.cms.hashes;
            var n_hashes: usize = 0;
            //const max_n_hashes = hashes.len;
            const max_n_hashes = 10 * 1024;
            var doc_to_pass = document;

            for (0..document.len - MY_LEN + 1) |i| {
                if (n_hashes == max_n_hashes) {
                    try self.flushSecondPass(doc_to_pass, hashes, n_hashes);
                    n_hashes = 0;
                    doc_to_pass = document[i..];
                }
                self.cms.getHashes(&hashes[n_hashes], @ptrCast(&document[i]));
                n_hashes += 1;
                //try self.processString(document[i .. i + MY_LEN]);
            }

            if (n_hashes > 0) {
                try self.flushSecondPass(doc_to_pass, hashes, n_hashes);
            }

            //const elapsed = time.nanoTimestamp() - start_time;
            //std.debug.print("[TIMING] processDocumentSecondPass ({d} bytes): {d:.2}ms\n", .{ document.len, @as(f64, @floatFromInt(elapsed)) / time.ns_per_ms });
        }

        pub fn mergeCounts(self: *Self, other: *Self) !void {
            while (other.heap.count() > 0) {
                var item = other.heap.remove();
                const others_string = item.string;
                const other_count = other.actual_counts.deleteKnownPresent(item.string, item.hash);
                const maybe_value_ptr = self.actual_counts.getPtr(item.string, item.hash);
                if (maybe_value_ptr) |value_ptr| {
                    value_ptr.* += other_count;
                } else if (self.heap.count() < top_k) {
                    const str_copy = try self.allocator.dupe(u8, item.string);
                    item.string = str_copy;
                    try self.heap.add(item);
                    self.actual_counts.insertKnownNotPresent(str_copy, item.hash, other_count);
                } else if (CandidateString.lessThan(undefined, self.heap.peek().?, item) == .lt) {
                    const evicted = self.heap.remove();
                    _ = self.actual_counts.deleteKnownPresent(evicted.string, evicted.hash);
                    const str_copy = evicted.string;
                    @memcpy(str_copy, item.string);
                    item.string = str_copy;
                    try self.heap.add(item);
                    self.actual_counts.insertKnownNotPresent(str_copy, item.hash, other_count);
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
                .min_length = @as(u32, @intCast(min_length)),
                .max_length = @as(u32, @intCast(max_length)),
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

        // Get results and calculate error statistics
        pub fn getResults(self: *Self) !void {
            const start_time = time.nanoTimestamp();

            std.debug.print("\n=== FREQUENCY ANALYSIS RESULTS ===\n", .{});

            var total_strings: usize = 0;

            // Process and display length 2 strings
            {
                const display_count = top_k;

                const ItemType2 = struct { index: u16, count: usize };
                const lessThan2 = struct {
                    pub fn lessThan(_: void, a: ItemType2, b: ItemType2) std.math.Order {
                        return std.math.order(a.count, b.count);
                    }
                }.lessThan;

                var length2_heap = std.PriorityQueue(ItemType2, void, lessThan2).init(self.allocator, {});
                defer length2_heap.deinit();

                for (self.length2_counters, 0..) |count, index| {
                    if (count == 0) continue;

                    if (length2_heap.count() < display_count) {
                        try length2_heap.add(.{ .index = @intCast(index), .count = count });
                    } else if (length2_heap.peek().?.count < count) {
                        _ = length2_heap.remove();
                        try length2_heap.add(.{ .index = @intCast(index), .count = count });
                    }
                }

                std.debug.print("\nLength 2: {d} strings\n", .{@min(display_count, length2_heap.count())});
                std.debug.print("Top strings of length 2:\n", .{});

                var results = std.ArrayList(ItemType2).init(self.allocator);
                defer results.deinit();

                while (length2_heap.count() > 0) {
                    try results.append(length2_heap.remove());
                }

                var i: usize = results.items.len;
                var displayed: usize = 0;
                var buf: [2]u8 = undefined;

                while (i > 0 and displayed < display_count) {
                    i -= 1;
                    const item = results.items[i];

                    const idx = item.index;
                    buf[0] = @intCast((idx >> 8) & 0xFF);
                    buf[1] = @intCast(idx & 0xFF);

                    std.debug.print("  {d}. '{s}': count={d}\n", .{
                        displayed + 1,
                        std.fmt.fmtSliceEscapeLower(&buf),
                        item.count,
                    });

                    displayed += 1;
                }

                total_strings += displayed;
            }

            // Process and display length 3 strings
            {
                const display_count = top_k;

                const ItemType3 = struct { index: u24, count: usize };
                const lessThan3 = struct {
                    pub fn lessThan(_: void, a: ItemType3, b: ItemType3) std.math.Order {
                        return std.math.order(a.count, b.count);
                    }
                }.lessThan;

                var length3_heap = std.PriorityQueue(ItemType3, void, lessThan3).init(self.allocator, {});
                defer length3_heap.deinit();

                // Iterate through all length 3 counters in one go (like length 2)
                for (self.length3_counters, 0..) |count, index| {
                    if (count == 0) continue;

                    if (length3_heap.count() < display_count) {
                        try length3_heap.add(.{ .index = @intCast(index), .count = count });
                    } else if (length3_heap.peek().?.count < count) {
                        _ = length3_heap.remove();
                        try length3_heap.add(.{ .index = @intCast(index), .count = count });
                    }
                }

                // Display results
                std.debug.print("\nLength 3: {d} strings\n", .{@min(display_count, length3_heap.count())});
                std.debug.print("Top strings of length 3:\n", .{});

                var results = std.ArrayList(ItemType3).init(self.allocator);
                defer results.deinit();

                while (length3_heap.count() > 0) {
                    try results.append(length3_heap.remove());
                }

                var i: usize = results.items.len;
                var displayed: usize = 0;
                var buf: [3]u8 = undefined;

                while (i > 0 and displayed < display_count) {
                    i -= 1;
                    const item = results.items[i];

                    const idx = item.index;
                    buf[0] = @intCast((idx >> 16) & 0xFF);
                    buf[1] = @intCast((idx >> 8) & 0xFF);
                    buf[2] = @intCast(idx & 0xFF);

                    std.debug.print("  {d}. '{s}': count={d}\n", .{ displayed + 1, std.fmt.fmtSliceEscapeLower(&buf), item.count });

                    displayed += 1;
                }

                total_strings += displayed;
            }

            if (MY_LEN >= 4) {
                var heap = &self.heap;
                var counts = &self.actual_counts;

                const count = heap.count();
                total_strings += count;

                std.debug.print("\nLength {d}: {d} strings\n", .{ MY_LEN, count });

                // Only print details for a few sample lengths to keep output manageable
                if (true) {
                    // Extract top strings (in reverse order)
                    var results = std.ArrayList(CandidateString).init(self.allocator);
                    defer results.deinit();

                    while (heap.count() > 0) {
                        try results.append(heap.remove());
                    }

                    // Print top strings with error stats
                    std.debug.print("Top strings of length {d}:\n", .{MY_LEN});

                    var total_error: f64 = 0;
                    var total_error_pct: f64 = 0;

                    const display_count = @min(1000, results.items.len);

                    // Display in descending order (highest frequency first)
                    var i: usize = results.items.len;
                    var displayed: usize = 0;

                    while (i > 0 and displayed < display_count) {
                        i -= 1;
                        const item = results.items[i];

                        if (counts.getPtr(item.string, item.hash)) |actual_ptr| {
                            const actual = actual_ptr.*;
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
                        }

                        // Add back to heap to restore original state
                        try heap.add(item);
                    }

                    if (displayed > 0) {
                        const avg_error_pct = total_error_pct / @as(f64, @floatFromInt(displayed));
                        std.debug.print("  Average error for length {d}: {d:.2}%\n", .{ MY_LEN, avg_error_pct });
                    }
                }
            }

            std.debug.print("\nTotal unique strings tracked: {d}\n", .{total_strings});

            const elapsed = time.nanoTimestamp() - start_time;
            std.debug.print("[TIMING] getResults: {d:.2}ms\n", .{@as(f64, @floatFromInt(elapsed)) / time.ns_per_ms});
        }
    };
}
