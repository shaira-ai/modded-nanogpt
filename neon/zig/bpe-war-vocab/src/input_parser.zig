const std = @import("std");
const types = @import("types.zig");

pub const TopStringsByLength = struct {
    allocator: std.mem.Allocator,
    strings_by_length: std.AutoHashMap(usize, std.ArrayList(types.FreqString)),
    total_count: usize,

    pub fn init(allocator: std.mem.Allocator) TopStringsByLength {
        return .{
            .allocator = allocator,
            .strings_by_length = std.AutoHashMap(usize, std.ArrayList(types.FreqString)).init(allocator),
            .total_count = 0,
        };
    }

    pub fn deinit(self: *TopStringsByLength) void {
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

    /// Add a string with its frequency to the appropriate length category
    pub fn addString(self: *TopStringsByLength, length: usize, str: []const u8, frequency: usize) !void {
        // Create array for this length if it doesn't exist
        if (!self.strings_by_length.contains(length)) {
            try self.strings_by_length.put(length, std.ArrayList(types.FreqString).init(self.allocator));
        }

        var strings = self.strings_by_length.getPtr(length).?;

        const content_copy = try self.allocator.dupe(u8, str);

        try strings.append(.{
            .content = content_copy,
            .frequency = frequency,
            .length = length,
        });

        self.total_count += 1;
    }

    fn freqDescending(context: void, a: types.FreqString, b: types.FreqString) bool {
        _ = context;
        return a.frequency > b.frequency;
    }

    /// Sort all string lists by frequency (descending) to easily identify top strings
    pub fn sortByFrequency(self: *TopStringsByLength) void {
        var it = self.strings_by_length.valueIterator();
        while (it.next()) |array| {
            std.sort.insertion(types.FreqString, array.items, {}, freqDescending);
        }
    }

    // pub fn printSummary(self: *TopStringsByLength) void {
    //     std.debug.print("TopStringsByLength Summary:\n", .{});
    //     std.debug.print("  Total unique strings: {d}\n", .{self.total_count});
    //     std.debug.print("  Distinct lengths: {d}\n", .{self.strings_by_length.count()});

    //     var total_weighted_frequency: usize = 0;
    //     var it = self.strings_by_length.iterator();
    //     while (it.next()) |entry| {
    //         const length = entry.key_ptr.*;
    //         const strings = entry.value_ptr.*.items;

    //         var length_frequency: usize = 0;
    //         for (strings) |str| {
    //             length_frequency += str.frequency;
    //         }

    //         total_weighted_frequency += length_frequency;
    //         std.debug.print("  Length {d}: {d} strings, cumulative frequency: {d}\n", .{ length, strings.len, length_frequency });
    //     }

    //     std.debug.print("  Total weighted frequency: {d}\n", .{total_weighted_frequency});
    // }
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

// pub fn testParser() !void {
//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     defer _ = gpa.deinit();
//     var allocator = gpa.allocator();

//     const test_input =
//         \\1,a,1000
//         \\1,b,900
//         \\1,c,800
//         \\2,aa,500
//         \\2,ab,450
//         \\2,bc,400
//         \\3,abc,300
//         \\3,def,250
//         \\3,xyz,200
//     ;

//     var strings = try parseInput(allocator, test_input);
//     defer strings.deinit();

//     // strings.printSummary();

//     if (strings.getStringsOfLength(1)) |length1_strings| {
//         std.debug.print("\nTop strings of length 1:\n", .{});
//         for (length1_strings) |str| {
//             std.debug.print("  '{s}': {d}\n", .{ str.content, str.frequency });
//         }
//     }

//     if (strings.getStringsOfLength(3)) |length3_strings| {
//         std.debug.print("\nTop strings of length 3:\n", .{});
//         for (length3_strings) |str| {
//             std.debug.print("  '{s}': {d}\n", .{ str.content, str.frequency });
//         }
//     }

//     // Get all available lengths
//     const lengths = try strings.getAllLengths();
//     defer allocator.free(lengths);

//     std.debug.print("\nAvailable lengths: ", .{});
//     for (lengths) |length| {
//         std.debug.print("{d} ", .{length});
//     }
//     std.debug.print("\n", .{});
// }
