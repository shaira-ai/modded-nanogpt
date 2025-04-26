const std = @import("std");
const Allocator = std.mem.Allocator;
const fs = std.fs;
const io = std.io;
const mem = std.mem;
const time = std.time;

const TopStringsByLength = @import("top_strings_by_length.zig").TopStringsByLength;
const types = @import("types.zig");

pub fn main() !void {
    // Initialize general purpose allocator
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Parse command line arguments
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2 or args.len > 3) {
        std.debug.print("Usage: {s} <input_file> [output_file]\n", .{args[0]});
        std.debug.print("  If output_file is not provided, output goes to stdout\n", .{});
        return;
    }

    const input_path = args[1];
    const use_stdout = args.len == 2;
    const output_path = if (use_stdout) "" else args[2];

    // Parse input file to get token data organized by length
    std.debug.print("Loading token data from {s}...\n", .{input_path});
    const start_time = time.nanoTimestamp();
    var token_data = try parseFile(allocator, input_path);
    defer token_data.deinit();
    const elapsed = time.nanoTimestamp() - start_time;
    std.debug.print("Loaded token data in {d:.2}ms\n", .{@as(f64, @floatFromInt(elapsed)) / time.ns_per_ms});

    // Output binary format
    try writeBinaryFormat(allocator, &token_data, use_stdout, output_path);
}

fn parseFile(allocator: Allocator, file_path: []const u8) !TopStringsByLength {
    const file = try std.fs.cwd().openFile(file_path, .{});
    defer file.close();

    const file_size = try file.getEndPos();
    const buffer = try allocator.alloc(u8, file_size);
    defer allocator.free(buffer);
    const bytes_read = try file.readAll(buffer);

    var result = TopStringsByLength.init(allocator);
    errdefer result.deinit();

    var line_count: usize = 0;
    var lines = std.mem.splitSequence(u8, buffer[0..bytes_read], "\n");
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

        try result.addString(length, string, frequency);
    }

    // Sort by frequency to easily identify top strings
    result.sortByFrequency();
    std.debug.print("Parsed {d} lines, found {d} unique strings\n", .{ line_count, result.total_count });

    return result;
}

fn writeBinaryFormat(allocator: Allocator, token_data: *TopStringsByLength, use_stdout: bool, output_path: []const u8) !void {
    // Prepare the output
    const file = if (use_stdout)
        std.io.getStdOut()
    else
        try std.fs.cwd().createFile(output_path, .{});

    if (!use_stdout) {
        defer file.close();
    }

    const writer = file.writer();

    // Build header: 256 u32s representing number of strings per length
    var header: [256]u32 = [_]u32{0} ** 256;

    // Get all lengths that have strings
    const lengths = try token_data.getAllLengths();
    defer allocator.free(lengths);

    // Calculate count for each length
    for (lengths) |length| {
        if (length == 0 or length > 256) {
            std.debug.print("Warning: Invalid length found: {d}, skipping\n", .{length});
            continue;
        }

        if (token_data.getStringsOfLength(length)) |strings| {
            header[length - 1] = @intCast(strings.len); // Adjust for 0-based indexing
        }
    }

    // Write header (256 u32s)
    std.debug.print("Writing header ({d} bytes)...\n", .{header.len * @sizeOf(u32)});
    try writer.writeAll(std.mem.asBytes(&header));

    // Write strings in order of length
    var total_strings: usize = 0;
    for (0..256) |i| {
        const length = i + 1; // Adjust for 0-based array indexing
        const count = header[i];

        if (count == 0) continue;

        std.debug.print("Writing {d} strings of length {d}...\n", .{ count, length });

        const strings = token_data.getStringsOfLength(length) orelse continue;

        // Write all strings of this length consecutively
        for (strings) |str| {
            try writer.writeAll(str.content);
            total_strings += 1;
        }
    }

    std.debug.print("Binary format written successfully: {d} total strings\n", .{total_strings});
    if (!use_stdout) {
        std.debug.print("Output saved to {s}\n", .{output_path});
    }
}
