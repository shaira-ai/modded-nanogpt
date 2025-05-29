const std = @import("std");
const types = @import("types.zig");
const time = std.time;

pub const TopStringsByLength = struct {
    allocator: std.mem.Allocator,
    strings_by_length: std.AutoHashMap(usize, std.ArrayList(types.FreqString)),
    // Store string to index mapping for deduplication
    string_indexes: std.AutoHashMap(usize, std.StringHashMap(usize)),
    total_count: usize,

    pub fn init(allocator: std.mem.Allocator) TopStringsByLength {
        return .{
            .allocator = allocator,
            .strings_by_length = std.AutoHashMap(usize, std.ArrayList(types.FreqString)).init(allocator),
            .string_indexes = std.AutoHashMap(usize, std.StringHashMap(usize)).init(allocator),
            .total_count = 0,
        };
    }

    pub fn deinit(self: *TopStringsByLength) void {
        // Clean up string index maps
        var index_it = self.string_indexes.iterator();
        while (index_it.next()) |entry| {
            entry.value_ptr.*.deinit();
        }
        self.string_indexes.deinit();

        // Clean up string arrays
        var it = self.strings_by_length.iterator();
        while (it.next()) |entry| {
            for (entry.value_ptr.*.items) |str| {
                self.allocator.free(str.content);
            }
            entry.value_ptr.*.deinit();
        }
        self.strings_by_length.deinit();
    }

    pub fn getStringsOfLength(self: *TopStringsByLength, length: usize) ?[]types.FreqString {
        if (self.strings_by_length.get(length)) |array| {
            return array.items;
        }
        return null;
    }

    fn lengthLessThan(context: void, a: usize, b: usize) bool {
        _ = context;
        return a < b;
    }

    pub fn getAllLengths(self: *TopStringsByLength) ![]usize {
        var lengths = std.ArrayList(usize).init(self.allocator);
        defer lengths.deinit();

        var it = self.strings_by_length.keyIterator();
        while (it.next()) |length| {
            try lengths.append(length.*);
        }

        // Sort lengths for consistent processing
        std.sort.insertion(usize, lengths.items, {}, lengthLessThan);
        return try lengths.toOwnedSlice();
    }

    /// Simplified addString with direct deduplication
    pub fn addString(self: *TopStringsByLength, length: usize, str: []const u8, frequency: usize) !void {
        const start_time = time.nanoTimestamp();

        // Make sure we have a list for this length
        if (!self.strings_by_length.contains(length)) {
            try self.strings_by_length.put(length, std.ArrayList(types.FreqString).init(self.allocator));
            try self.string_indexes.put(length, std.StringHashMap(usize).init(self.allocator));
        }

        var strings = self.strings_by_length.getPtr(length).?;
        var indexes = self.string_indexes.getPtr(length).?;

        // Check if we've seen this string before
        if (indexes.get(str)) |index| {
            // Update existing string's frequency
            strings.items[index].frequency += frequency;
        } else {
            // New string, make a copy and add it
            const content_copy = try self.allocator.dupe(u8, str);
            errdefer self.allocator.free(content_copy);

            const new_index = strings.items.len;
            try strings.append(.{
                .content = content_copy,
                .frequency = frequency,
                .length = length,
            });

            // Remember the index for this string
            try indexes.put(content_copy, new_index);

            self.total_count += 1;
        }

        const elapsed = time.nanoTimestamp() - start_time;
        if (elapsed > 1000000) { // Only report if it took more than 1ms
            std.debug.print("[TIMING] addString for length {d}, string size {d}: {d:.2}ms\n", .{ length, str.len, @as(f64, @floatFromInt(elapsed)) / time.ns_per_ms });
        }
    }

    fn freqDescending(context: void, a: types.FreqString, b: types.FreqString) bool {
        _ = context;
        return a.frequency > b.frequency;
    }

    /// Sort all string lists by frequency (descending) to easily identify top strings
    pub fn sortByFrequency(self: *TopStringsByLength) void {
        const start_time = time.nanoTimestamp();

        var it = self.strings_by_length.valueIterator();
        while (it.next()) |array| {
            std.sort.insertion(types.FreqString, array.items, {}, freqDescending);
        }

        // After sorting, the indexes are no longer valid, so clear them
        var index_it = self.string_indexes.valueIterator();
        while (index_it.next()) |index_map| {
            index_map.clearAndFree();
        }

        const elapsed = time.nanoTimestamp() - start_time;
        std.debug.print("[TIMING] sortByFrequency: {d:.2}ms\n", .{@as(f64, @floatFromInt(elapsed)) / time.ns_per_ms});
    }
};

pub fn parseInput(allocator: std.mem.Allocator, input: []const u8) !TopStringsByLength {
    var result = TopStringsByLength.init(allocator);
    errdefer result.deinit();

    var line_count: usize = 0;
    var lines = std.mem.splitSequence(u8, input, "\n");
    while (lines.next()) |line| {
        line_count += 1;
        const trimmed = std.mem.trim(u8, line, &std.ascii.whitespace);
        if (trimmed.len == 0) continue;

        // Parse line in format: length,string,frequency
        var parts = std.mem.splitScalar(u8, trimmed, ',');

        const length_str = parts.next() orelse {
            std.debug.print("Warning: Missing length in line {d}: '{s}'\n", .{ line_count, trimmed });
            continue;
        };

        const string = parts.next() orelse {
            std.debug.print("Warning: Missing string in line {d}: '{s}'\n", .{ line_count, trimmed });
            continue;
        };

        const freq_str = parts.next() orelse {
            std.debug.print("Warning: Missing frequency in line {d}: '{s}'\n", .{ line_count, trimmed });
            continue;
        };

        // Parse numeric values
        const length = std.fmt.parseInt(usize, length_str, 10) catch |err| {
            std.debug.print("Warning: Invalid length in line {d}: '{s}' ({any})\n", .{ line_count, length_str, err });
            continue;
        };

        const frequency = std.fmt.parseInt(usize, freq_str, 10) catch |err| {
            std.debug.print("Warning: Invalid frequency in line {d}: '{s}' ({any})\n", .{ line_count, freq_str, err });
            continue;
        };

        // Verify the length matches the actual string length
        if (length != string.len) {
            std.debug.print("Warning: Declared length {d} doesn't match actual string length {d} in line {d}\n", .{ length, string.len, line_count });
            // We'll use the declared length since that's how the data is organized
        }

        try result.addString(length, string, frequency);
    }

    // Sort by frequency to easily identify top strings
    result.sortByFrequency();

    return result;
}

/// parses csv data
pub fn parseFile(allocator: std.mem.Allocator, file_path: []const u8) !TopStringsByLength {
    const file = try std.fs.cwd().openFile(file_path, .{});
    defer file.close();

    const file_size = try file.getEndPos();
    const buffer = try allocator.alloc(u8, file_size);
    defer allocator.free(buffer);
    const bytes_read = try file.readAll(buffer);

    return parseInput(allocator, buffer[0..bytes_read]);
}
