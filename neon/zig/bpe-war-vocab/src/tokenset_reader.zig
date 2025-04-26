const std = @import("std");
const Allocator = std.mem.Allocator;
const fs = std.fs;
const io = std.io;
const mem = std.mem;

pub fn main() !void {
    // Initialize general purpose allocator
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Parse command line arguments
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len != 2) {
        std.debug.print("Usage: {s} <binary_token_file>\n", .{args[0]});
        return;
    }

    const input_path = args[1];
    try displayTokenSet(allocator, input_path);
}

fn displayTokenSet(allocator: Allocator, file_path: []const u8) !void {
    const file = try std.fs.cwd().openFile(file_path, .{});
    defer file.close();

    const file_size = try file.getEndPos();
    std.debug.print("Reading token set file: {s} ({d} bytes)\n", .{ file_path, file_size });

    // Ensure file is large enough for header
    const header_size = 256 * @sizeOf(u32);
    if (file_size < header_size) {
        return error.FileTooSmall;
    }

    // Read and parse header (256 u32s)
    var header: [256]u32 = undefined;
    const bytes_read = try file.readAll(std.mem.asBytes(&header));
    if (bytes_read != header_size) {
        return error.IncompleteHeader;
    }

    // Count total strings and validate header
    var total_strings: usize = 0;
    var total_bytes_in_strings: usize = 0;
    for (header, 0..) |count, i| {
        total_strings += count;
        total_bytes_in_strings += count * (i + 1); // Each string has length (i+1)
    }

    std.debug.print("Header parsed: {d} total strings, {d} bytes of string data\n", .{ total_strings, total_bytes_in_strings });

    // Validate file size
    if (file_size != header_size + total_bytes_in_strings) {
        std.debug.print("Warning: File size mismatch. Expected {d} bytes, got {d} bytes\n", .{ header_size + total_bytes_in_strings, file_size });
    }

    // Display strings from each length
    var num_strings_displayed: usize = 0;
    const max_strings_per_length: usize = 10; // Display at most this many strings per length
    const truncate_at: usize = 40; // Truncate display of strings longer than this

    for (header, 0..) |count, i| {
        const length = i + 1; // Adjust from 0-based index

        if (count == 0) continue;

        std.debug.print("\n=== Length {d}: {d} strings ===\n", .{ length, count });

        // Allocate buffer for reading strings of this length
        const buffer_size = length * @min(count, max_strings_per_length);
        const buffer = try allocator.alloc(u8, buffer_size);
        defer allocator.free(buffer);

        // Display a subset of strings
        const strings_to_display = @min(count, max_strings_per_length);
        if (strings_to_display < count) {
            std.debug.print("(Showing first {d} of {d} strings)\n", .{ strings_to_display, count });
        }

        _ = try file.reader().readAll(buffer[0 .. strings_to_display * length]);

        for (0..strings_to_display) |j| {
            const str = buffer[j * length .. (j + 1) * length];

            // Display the string (possibly truncated)
            if (length > truncate_at) {
                std.debug.print("{d}: {s}... (truncated, {d} bytes total)\n", .{ j + 1, str[0..truncate_at], length });
            } else {
                std.debug.print("{d}: {s}\n", .{ j + 1, str });
            }
        }

        // Skip the remaining strings of this length
        if (strings_to_display < count) {
            try file.seekBy(@intCast((count - strings_to_display) * length));
        }

        num_strings_displayed += strings_to_display;
    }

    std.debug.print("\nDisplayed {d} of {d} total strings\n", .{ num_strings_displayed, total_strings });
}
