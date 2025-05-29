const std = @import("std");
const fs = std.fs;
const print = std.debug.print;
const Allocator = std.mem.Allocator;

const TokenInfo = struct {
    bytes: []u8,
    count: u64,
    length: usize,
};

fn printableString(allocator: Allocator, bytes: []const u8) ![]u8 {
    var result = std.ArrayList(u8).init(allocator);
    defer result.deinit();

    for (bytes) |byte| {
        if (byte >= 32 and byte <= 126) {
            // Printable ASCII
            try result.append(byte);
        } else {
            // Non-printable - show as hex
            const hex = try std.fmt.allocPrint(allocator, "\\x{X:0>2}", .{byte});
            defer allocator.free(hex);
            try result.appendSlice(hex);
        }
    }

    return result.toOwnedSlice();
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Parse command line arguments
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        print("Usage: {s} <tokenset_file.bin> [options]\n", .{args[0]});
        print("Options:\n", .{});
        print("  --top N        Show only top N tokens by count (default: 50)\n", .{});
        print("  --length L     Show only tokens of length L\n", .{});
        print("  --min-count C  Show only tokens with count >= C\n", .{});
        print("  --stats        Show statistics summary\n", .{});
        return;
    }

    const filename = args[1];

    // Parse options
    var show_top: usize = 50;
    var filter_length: ?usize = null;
    var min_count: u64 = 0;
    var show_stats = false;

    var i: usize = 2;
    while (i < args.len) {
        if (std.mem.eql(u8, args[i], "--top") and i + 1 < args.len) {
            show_top = try std.fmt.parseUnsigned(usize, args[i + 1], 10);
            i += 2;
        } else if (std.mem.eql(u8, args[i], "--length") and i + 1 < args.len) {
            filter_length = try std.fmt.parseUnsigned(usize, args[i + 1], 10);
            i += 2;
        } else if (std.mem.eql(u8, args[i], "--min-count") and i + 1 < args.len) {
            min_count = try std.fmt.parseUnsigned(u64, args[i + 1], 10);
            i += 2;
        } else if (std.mem.eql(u8, args[i], "--stats")) {
            show_stats = true;
            i += 1;
        } else {
            print("Unknown option: {s}\n", .{args[i]});
            return;
        }
    }

    // Open and read the file
    const file = fs.cwd().openFile(filename, .{}) catch |err| {
        print("Error opening file '{s}': {s}\n", .{ filename, @errorName(err) });
        return;
    };
    defer file.close();

    var buffered_reader = std.io.bufferedReader(file.reader());
    var reader = buffered_reader.reader();

    print("Reading tokenset file: {s}\n", .{filename});
    print("=" ** 50 ++ "\n", .{});

    // Read header (256 u32 values for token counts by length)
    var header: [256]u32 = undefined;
    const header_bytes = try reader.readAll(std.mem.asBytes(&header));
    if (header_bytes != 256 * @sizeOf(u32)) {
        print("Error: Invalid header size. Expected {d} bytes, got {d}\n", .{ 256 * @sizeOf(u32), header_bytes });
        return;
    }

    // Calculate total tokens and show length distribution
    var total_tokens: usize = 0;
    var length_stats = std.ArrayList(struct { length: usize, count: u32 }).init(allocator);
    defer length_stats.deinit();

    for (header, 0..) |count, length| {
        if (count > 0) {
            total_tokens += count;
            try length_stats.append(.{ .length = length, .count = count });
        }
    }

    print("Total tokens: {d}\n", .{total_tokens});
    print("Length distribution:\n", .{});
    for (length_stats.items) |stat| {
        print("  Length {d}: {d} tokens\n", .{ stat.length, stat.count });
    }
    print("\n", .{});

    if (show_stats) {
        print("Statistics summary complete.\n", .{});
        return;
    }

    // Read tokens with their counts
    var all_tokens = std.ArrayList(TokenInfo).init(allocator);
    defer {
        for (all_tokens.items) |token| {
            allocator.free(token.bytes);
        }
        all_tokens.deinit();
    }

    // Read interleaved data organized by length groups
    for (0..256) |length| {
        const count = header[length];
        if (count == 0 or length == 0) continue; // Skip length-0 tokens

        print("Reading {d} tokens of length {d}...\n", .{ count, length });

        for (0..count) |token_idx| {
            // Read token bytes
            const token_bytes = try allocator.alloc(u8, length);
            const bytes_read = try reader.readAll(token_bytes);
            if (bytes_read != length) {
                print("Error: Expected {d} bytes for token {d}, got {d}\n", .{ length, token_idx, bytes_read });
                allocator.free(token_bytes);
                return;
            }

            // Read occurrence count (8 bytes, u64)
            var count_bytes: [8]u8 = undefined;
            const count_bytes_read = try reader.readAll(&count_bytes);
            if (count_bytes_read != 8) {
                print("Error: Expected 8 bytes for count of token {d}, got {d}\n", .{ token_idx, count_bytes_read });
                print("Token bytes: {any}\n", .{token_bytes[0..@min(token_bytes.len, 10)]});

                // Show remaining file size
                const current_pos = try file.getPos();
                const file_size = try file.getEndPos();
                print("Current file position: {d}, file size: {d}, remaining: {d}\n", .{ current_pos - count_bytes_read, file_size, file_size - (current_pos - count_bytes_read) });

                allocator.free(token_bytes);
                return;
            }
            const occurrence_count = std.mem.readInt(u64, &count_bytes, .little);

            // Apply filters
            if (filter_length != null and length != filter_length.?) {
                allocator.free(token_bytes);
                continue;
            }
            if (occurrence_count < min_count) {
                allocator.free(token_bytes);
                continue;
            }

            try all_tokens.append(TokenInfo{
                .bytes = token_bytes,
                .count = occurrence_count,
                .length = length,
            });
        }
    }

    // Sort by occurrence count (descending)
    const SortContext = struct {
        fn lessThan(context: void, a: TokenInfo, b: TokenInfo) bool {
            _ = context;
            return a.count > b.count; // Descending order
        }
    };
    std.sort.pdq(TokenInfo, all_tokens.items, {}, SortContext.lessThan);

    // Display results
    if (filter_length) |len| {
        print("Showing tokens of length {d}:\n", .{len});
    } else {
        print("Showing top {d} tokens by occurrence count:\n", .{@min(show_top, all_tokens.items.len)});
    }
    print("{s:>6} | {s:>12} | {s:>6} | {s}\n", .{ "Rank", "Count", "Length", "Token" });
    print("-" ** 70 ++ "\n", .{});

    const display_count = @min(show_top, all_tokens.items.len);
    for (all_tokens.items[0..display_count], 1..) |token, rank| {
        const printable = try printableString(allocator, token.bytes);
        defer allocator.free(printable);

        print("{d:>6} | {d:>12} | {d:>6} | '{s}'\n", .{ rank, token.count, token.length, printable });
    }

    if (all_tokens.items.len > display_count) {
        print("... and {d} more tokens\n", .{all_tokens.items.len - display_count});
    }

    print("\nSummary:\n", .{});
    print("  Total tokens matching filters: {d}\n", .{all_tokens.items.len});
    if (all_tokens.items.len > 0) {
        print("  Highest count: {d}\n", .{all_tokens.items[0].count});
        print("  Lowest count: {d}\n", .{all_tokens.items[all_tokens.items.len - 1].count});

        // Calculate total occurrences
        var total_occurrences: u64 = 0;
        for (all_tokens.items) |token| {
            total_occurrences += token.count;
        }
        print("  Total occurrences: {d}\n", .{total_occurrences});
        print("  Average occurrences per token: {d:.2}\n", .{@as(f64, @floatFromInt(total_occurrences)) / @as(f64, @floatFromInt(all_tokens.items.len))});
    }
}
