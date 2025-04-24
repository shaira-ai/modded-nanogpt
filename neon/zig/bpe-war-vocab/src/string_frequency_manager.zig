const std = @import("std");
const time = std.time;
const CMS_F = @import("count_min_sketch.zig").CountMinSketch;
const N_LENGTHS = @import("count_min_sketch.zig").N_LENGTHS;
const MY_LEN = @import("count_min_sketch.zig").MY_LEN;
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

const SerializationHeader = extern struct {
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
    comptime min_length: usize,
    comptime max_length: usize,
) type {
    const CMS = CMS_F(cms_width, cms_depth);
    return struct {
        allocator: std.mem.Allocator,
        cms: *CMS,
        top_k: usize,
        length2_counters: []usize,
        length3_counters: []usize,
        cms_is_owned: bool = true,
        // Min heaps for tracking top-K strings of each length
        heaps: []std.PriorityQueue(CandidateString, void, CandidateString.lessThan),
        // Maps to track actual counts of candidate strings
        actual_counts: []std.StringHashMap(usize),

        const Self = @This();

        pub fn init(allocator: std.mem.Allocator, top_k: usize) !*Self {
            const start_time = time.nanoTimestamp();

            const self = try allocator.create(Self);
            errdefer allocator.destroy(self);

            // Create the Count-Min Sketch
            const cms = try CMS.init(allocator);
            errdefer cms.deinit();

            // Initialize heaps and count maps
            const heaps = try allocator.alloc(std.PriorityQueue(CandidateString, void, CandidateString.lessThan), max_length + 1);
            for (heaps) |*heap| {
                heap.* = std.PriorityQueue(CandidateString, void, CandidateString.lessThan).init(allocator, {});
            }
            const actual_counts = try allocator.alloc(std.StringHashMap(usize), max_length + 1);
            for (actual_counts) |*map| {
                map.* = std.StringHashMap(usize).init(allocator);
            }
            const len2_counters = try allocator.alloc(usize, 0x10000);
            const len3_counters = try allocator.alloc(usize, 0x1000000);
            @memset(len2_counters, 0);
            @memset(len3_counters, 0);

            self.* = .{
                .allocator = allocator,
                .cms = cms,
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
            if (self.cms_is_owned) {
                self.cms.deinit();
            }
            self.allocator.free(self.length2_counters);
            self.allocator.free(self.length3_counters);
            for (self.heaps) |*heap| {
                while (heap.count() > 0) {
                    const item = heap.remove();
                    self.allocator.free(item.string);
                }
                heap.deinit();
            }
            self.allocator.free(self.heaps);

            for (self.actual_counts) |*map| {
                map.deinit();
            }
            self.allocator.free(self.actual_counts);
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

        // PASS 1: Build the Count-Min Sketch
        pub fn buildCMS(self: *Self, document: []const u8) !void {
            //const start_time = time.nanoTimestamp();

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

                const max_len = @min(max_length, document.len - i);
                if (max_len >= 4) {
                    self.cms.addPrefixes(@ptrCast(&document[i]), max_len);
                }
            }

            self.cms.flush();

            //const elapsed = time.nanoTimestamp() - start_time;
            //std.debug.print("[TIMING] buildCMS ({d} bytes): {d:.2}ms\n", .{ document.len, @as(f64, @floatFromInt(elapsed)) / time.ns_per_ms });
        }

        fn processString(self: *Self, substring: []const u8, guess_count: u64) !void {
            const len = substring.len;
            var heap = &self.heaps[len];
            var counts = &self.actual_counts[len];

            // Check if we're already tracking this string
            if (counts.contains(substring)) {
                //std.debug.print("Already tracking '{s}'\n", .{substring});
                // Already tracking, just increment the actual count
                counts.getPtr(substring).?.* += 1;
            } else if (heap.count() < self.top_k) {
                //std.debug.print("Adding '{s}' to heap\n", .{substring});
                // Haven't reached capacity yet, add the string
                const str_copy = try self.allocator.dupe(u8, substring);
                errdefer self.allocator.free(str_copy);

                try heap.add(.{ .string = str_copy, .guess_count = guess_count });
                try counts.put(str_copy, 1);
            } else if (heap.count() == self.top_k and heap.peek().?.guess_count < guess_count) {
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
        }

        // PASS 2: Identify top-K strings and calculate actual counts
        pub fn processDocumentSecondPass(self: *Self, document: []const u8) !void {
            //const start_time = time.nanoTimestamp();

            for (0..document.len) |i| {
                const max_len = @min(max_length, document.len - i);
                // if (max_len < 2) {
                //     continue;
                // }

                // const guess_count_2 = self.length2_counters[length2ToIndex(document[i .. i + 2])];
                // try self.processString(document[i .. i + 2], guess_count_2);

                // if (max_len < 3) {
                //     continue;
                // }

                // const guess_count_3 = self.length3_counters[length3ToIndex(document[i .. i + 3])];
                // try self.processString(document[i .. i + 3], guess_count_3);

                if (max_len < MY_LEN) {
                    continue;
                }

                var scratch: [N_LENGTHS]u64 = undefined;
                self.cms.query(&scratch, @ptrCast(&document[i]));
                for (scratch[0..1], MY_LEN..MY_LEN + 1) |guess_count, len| {
                    try self.processString(document[i .. i + len], guess_count);
                }
            }

            //const elapsed = time.nanoTimestamp() - start_time;
            //std.debug.print("[TIMING] processDocumentSecondPass ({d} bytes): {d:.2}ms\n", .{ document.len, @as(f64, @floatFromInt(elapsed)) / time.ns_per_ms });
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

            const self = try init(allocator, header.top_k);
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
            for (self.heaps, 0..) |*heap, len| {
                var counts = &self.actual_counts[len];

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

                    const display_count = @min(1000, results.items.len);

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
