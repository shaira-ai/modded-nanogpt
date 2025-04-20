const std = @import("std");
const time = std.time;
const CMS = @import("count_min_sketch.zig").CountMinSketch;
const fs = std.fs;

pub const CandidateString = struct {
    string: []const u8,
    guess_count: usize,

    pub fn lessThan(_: void, a: CandidateString, b: CandidateString) std.math.Order {
        if (a.guess_count < b.guess_count) {
            return .lt;
        } else if (a.guess_count > b.guess_count) {
            return .gt;
        } else {
            return .eq;
        }
    }
};

// File format version for serialization
const FILE_FORMAT_VERSION: u32 = 1;

const SerializationHeader = struct {
    magic: [4]u8, // "SFMF" --> String Frequency Manager File
    version: u32,
    cms_width: u32,
    cms_depth: u32,
    min_length: u32,
    max_length: u32,
    top_k: u32,
};

pub fn StringFrequencyManager(
    comptime cms_width: usize,
    comptime cms_depth: usize,
) type {
    return struct {
        allocator: std.mem.Allocator,
        cms: *CMS(cms_width, cms_depth),
        min_length: usize,
        max_length: usize,
        top_k: usize,
        length2_counters: []usize,
        length3_counters: []usize,

        // Min heaps for tracking top-K strings of each length
        heaps: std.AutoHashMap(usize, std.PriorityQueue(CandidateString, void, CandidateString.lessThan)),
        // Maps to track actual counts of candidate strings
        actual_counts: std.AutoHashMap(usize, std.StringHashMap(usize)),

        const Self = @This();

        pub fn init(allocator: std.mem.Allocator, min_length: usize, max_length: usize, top_k: usize) !*Self {
            const start_time = time.nanoTimestamp();

            const self = try allocator.create(Self);
            errdefer allocator.destroy(self);

            // Create the Count-Min Sketch
            const cms = try CMS(cms_width, cms_depth).init(allocator);
            errdefer cms.deinit();

            // Initialize heaps and count maps
            const heaps = std.AutoHashMap(usize, std.PriorityQueue(CandidateString, void, CandidateString.lessThan)).init(allocator);
            const actual_counts = std.AutoHashMap(usize, std.StringHashMap(usize)).init(allocator);
            const len2_counters = try allocator.alloc(usize, 65536); // 256^2
            const len3_counters = try allocator.alloc(usize, 16777216); // 256^3
            @memset(len2_counters, 0);
            @memset(len3_counters, 0);

            self.* = .{
                .allocator = allocator,
                .cms = cms,
                .min_length = min_length,
                .max_length = max_length,
                .top_k = top_k,
                .length2_counters = len2_counters,
                .length3_counters = len3_counters,
                .heaps = heaps,
                .actual_counts = actual_counts,
            };

            const elapsed = time.nanoTimestamp() - start_time;
            std.debug.print("[TIMING] StringFrequencyManager.init: {d:.2}ms\n", .{@as(f64, @floatFromInt(elapsed)) / time.ns_per_ms});

            return self;
        }

        pub fn deinit(self: *Self) void {
            const start_time = time.nanoTimestamp();

            self.cms.deinit();
            self.allocator.free(self.length2_counters);
            self.allocator.free(self.length3_counters);
            var heap_it = self.heaps.valueIterator();
            while (heap_it.next()) |heap_ptr| {
                var heap = heap_ptr.*;
                while (heap.count() > 0) {
                    const item = heap.remove();
                    self.allocator.free(item.string);
                }
                heap.deinit();
            }
            self.heaps.deinit();

            var count_it = self.actual_counts.valueIterator();
            while (count_it.next()) |map_ptr| {
                map_ptr.*.deinit();
            }
            self.actual_counts.deinit();
            self.allocator.destroy(self);

            const elapsed = time.nanoTimestamp() - start_time;
            std.debug.print("[TIMING] StringFrequencyManager.deinit: {d:.2}ms\n", .{@as(f64, @floatFromInt(elapsed)) / time.ns_per_ms});
        }

        inline fn length2ToIndex(substring: []const u8) usize {
            return (@as(usize, substring[0]) << 8) | substring[1];
        }
        inline fn length3ToIndex(substring: []const u8) usize {
            return (@as(usize, substring[0]) << 16) | (@as(usize, substring[1]) << 8) | substring[2];
        }

        // PASS 1: Build the Count-Min Sketch
        pub fn buildCMS(self: *Self, document: []const u8) !void {
            const start_time = time.nanoTimestamp();

            // Process all substrings of all lengths in one pass
            for (0..document.len) |i| {
                if (i + 2 <= document.len) {
                    const substring = document[i .. i + 2];
                    const index = length2ToIndex(substring);
                    self.length2_counters[index] += 1;
                }

                if (i + 3 <= document.len) {
                    const substring = document[i .. i + 3];
                    const index = length3ToIndex(substring);
                    self.length3_counters[index] += 1;
                }

                const max_len = @min(self.max_length, document.len - i);
                if (max_len >= 4) {
                    self.cms.addPrefixes(@ptrCast(&document[i]), max_len);
                }
            }

            self.cms.flush();

            const elapsed = time.nanoTimestamp() - start_time;
            std.debug.print("[TIMING] buildCMS ({d} bytes): {d:.2}ms\n", .{ document.len, @as(f64, @floatFromInt(elapsed)) / time.ns_per_ms });
        }

        // PASS 2: Identify top-K strings and calculate actual counts
        pub fn processDocumentSecondPass(self: *Self, document: []const u8) !void {
            const start_time = time.nanoTimestamp();

            // Make sure we have heaps and count maps for all lengths
            for (self.min_length..self.max_length + 1) |len| {
                if (!self.heaps.contains(len)) {
                    try self.heaps.put(len, std.PriorityQueue(CandidateString, void, CandidateString.lessThan).init(self.allocator, {}));
                }

                if (!self.actual_counts.contains(len)) {
                    try self.actual_counts.put(len, std.StringHashMap(usize).init(self.allocator));
                }
            }

            for (0..document.len) |i| {
                for (self.min_length..self.max_length + 1) |len| {
                    if (i + len <= document.len) {
                        const substring = document[i .. i + len];

                        // Get estimated count - either from direct counters or CMS
                        var guess_count: usize = 0;

                        if (len == 2) {
                            guess_count = self.length2_counters[length2ToIndex(substring)];
                        } else if (len == 3) {
                            guess_count = self.length3_counters[length3ToIndex(substring)];
                        } else {
                            guess_count = try self.cms.query(substring);
                        }

                        var heap = self.heaps.getPtr(len).?;
                        var counts = self.actual_counts.getPtr(len).?;

                        // Check if we're already tracking this string
                        if (counts.contains(substring)) {
                            // Already tracking, just increment the actual count
                            counts.getPtr(substring).?.* += 1;
                        } else if (heap.count() < self.top_k) {
                            // Haven't reached capacity yet, add the string
                            const str_copy = try self.allocator.dupe(u8, substring);
                            errdefer self.allocator.free(str_copy);

                            try heap.add(.{ .string = str_copy, .guess_count = guess_count });
                            try counts.put(str_copy, 1);
                        } else if (heap.count() > 0 and heap.peek().?.guess_count < guess_count) {
                            // Current string has higher estimated count than our minimum
                            const evicted = heap.remove();
                            _ = counts.remove(evicted.string);
                            self.allocator.free(evicted.string);

                            // Add the new string
                            const str_copy = try self.allocator.dupe(u8, substring);
                            errdefer self.allocator.free(str_copy);

                            try heap.add(.{ .string = str_copy, .guess_count = guess_count });
                            try counts.put(str_copy, 1);
                        }
                        // Else: Not in top-K, ignore
                    }
                }
            }

            const elapsed = time.nanoTimestamp() - start_time;
            std.debug.print("[TIMING] processDocumentSecondPass ({d} bytes): {d:.2}ms\n", .{ document.len, @as(f64, @floatFromInt(elapsed)) / time.ns_per_ms });
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
                .min_length = @as(u32, @intCast(self.min_length)),
                .max_length = @as(u32, @intCast(self.max_length)),
                .top_k = @as(u32, @intCast(self.top_k)),
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

            const self = try init(allocator, header.min_length, header.max_length, header.top_k);
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
            var len_it = self.heaps.keyIterator();
            while (len_it.next()) |len_ptr| {
                const len = len_ptr.*;
                var heap = self.heaps.getPtr(len).?;
                var counts = self.actual_counts.getPtr(len).?;

                const count = heap.count();
                total_strings += count;

                std.debug.print("\nLength {d}: {d} strings\n", .{ len, count });

                // Only print details for a few sample lengths to keep output manageable
                if (true) {
                    // Extract top strings (in reverse order)
                    var results = std.ArrayList(CandidateString).init(self.allocator);
                    defer results.deinit();

                    while (heap.count() > 0) {
                        try results.append(heap.remove());
                    }

                    // Print top 10 strings with error stats
                    std.debug.print("Top strings of length {d}:\n", .{len});

                    var total_error: f64 = 0;
                    var total_error_pct: f64 = 0;

                    const display_count = @min(10, results.items.len);

                    // Display in descending order (highest frequency first)
                    var i: usize = results.items.len;
                    var displayed: usize = 0;

                    while (i > 0 and displayed < display_count) {
                        i -= 1;
                        const item = results.items[i];

                        if (counts.get(item.string)) |actual| {
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
                        std.debug.print("  Average error for length {d}: {d:.2}%\n", .{ len, avg_error_pct });
                    }
                }
            }

            std.debug.print("\nTotal unique strings tracked: {d}\n", .{total_strings});

            const elapsed = time.nanoTimestamp() - start_time;
            std.debug.print("[TIMING] getResults: {d:.2}ms\n", .{@as(f64, @floatFromInt(elapsed)) / time.ns_per_ms});
        }
    };
}
