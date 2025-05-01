const std = @import("std");
const Allocator = std.mem.Allocator;
const fs = std.fs;
const io = std.io;
const mem = std.mem;
const process = std.process;
const ArrayList = std.ArrayList;
const Timer = std.time.Timer;

const MAX_TOKEN_LENGTH = 256;
const DEFAULT_OUTPUT_PATH = "/tmp/tokenset_filtered.bin";
const BUFFER_SIZE = 128 * 1024;

// Stats structure to track filtering results
const FilterStats = struct {
    total_tokens: usize,
    mixed_digits_nondigits: usize,
    letters_ending_space: usize,
    more_than_three_digits: usize,
    invalid_unicode: usize,
    latin_spaces_not_starting_space: usize,
    remaining_tokens: usize,
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var timer = try Timer.start();

    const args = try process.argsAlloc(allocator);
    defer process.argsFree(allocator, args);

    if (args.len < 2 or args.len > 4) {
        try printUsage(args[0]);
        return;
    }

    var input_path: []const u8 = undefined;
    var output_path: []const u8 = DEFAULT_OUTPUT_PATH;

    var i: usize = 1;
    while (i < args.len) {
        if (mem.eql(u8, args[i], "--output") or mem.eql(u8, args[i], "-o")) {
            if (i + 1 >= args.len) {
                std.debug.print("Error: Missing output path\n", .{});
                try printUsage(args[0]);
                return;
            }
            output_path = args[i + 1];
            i += 2;
        } else {
            input_path = args[i];
            i += 1;
        }
    }

    std.debug.print("Input file: {s}\n", .{input_path});
    std.debug.print("Output file: {s}\n", .{output_path});

    try filterTokenSets(allocator, input_path, output_path);

    const elapsed = timer.read();
    std.debug.print("Filtered token set in {d:.2}s\n", .{@as(f64, @floatFromInt(elapsed)) / std.time.ns_per_s});
}

fn printUsage(program_name: []const u8) !void {
    const stderr = std.io.getStdErr().writer();
    try stderr.print("Usage: {s} [--output OUTPUT_PATH] INPUT_FILE\n", .{program_name});
    try stderr.print("\nFilters tokens from a tokenset file based on banlist rules.\n", .{});
    try stderr.print("\nOptions:\n", .{});
    try stderr.print("  --output, -o OUTPUT_PATH  Path to write filtered output (default: {s})\n", .{DEFAULT_OUTPUT_PATH});
}

fn filterTokenSets(allocator: Allocator, input_path: []const u8, output_path: []const u8) !void {
    const input_file = try fs.cwd().openFile(input_path, .{});
    defer input_file.close();
    const output_file = try fs.cwd().createFile(output_path, .{});
    defer output_file.close();

    var input_header: [MAX_TOKEN_LENGTH]u32 = undefined;
    const header_bytes_read = try input_file.readAll(std.mem.asBytes(&input_header));
    if (header_bytes_read != MAX_TOKEN_LENGTH * @sizeOf(u32)) {
        return error.IncompleteHeader;
    }

    // Initialize output header and stats
    var output_header = [_]u32{0} ** MAX_TOKEN_LENGTH;
    var stats = FilterStats{
        .total_tokens = 0,
        .mixed_digits_nondigits = 0,
        .letters_ending_space = 0,
        .more_than_three_digits = 0,
        .invalid_unicode = 0,
        .latin_spaces_not_starting_space = 0,
        .remaining_tokens = 0,
    };

    for (input_header, 0..) |count, length_idx| {
        const length = length_idx + 1;
        if (count == 0) continue;

        std.debug.print("Processing {d} tokens of length {d}...\n", .{ count, length });
        stats.total_tokens += count;

        const tokens_per_buffer = BUFFER_SIZE / length;
        const buffer_size_in_bytes = tokens_per_buffer * length;

        var token_buffer = try allocator.alloc(u8, buffer_size_in_bytes);
        defer allocator.free(token_buffer);

        var filtered_buffer = try allocator.alloc(u8, buffer_size_in_bytes);
        defer allocator.free(filtered_buffer);

        var tokens_remaining: u32 = 0;
        var tokens_left = count;

        while (tokens_left > 0) {
            const tokens_to_read = @min(tokens_left, tokens_per_buffer);
            const bytes_to_read = tokens_to_read * length;

            const bytes_read = try input_file.readAll(token_buffer[0..bytes_to_read]);
            if (bytes_read == 0) break;

            const complete_tokens = bytes_read / length;
            if (complete_tokens == 0) break;

            var filtered_bytes: usize = 0;

            for (0..complete_tokens) |token_idx| {
                const token_start = token_idx * length;
                const token = token_buffer[token_start .. token_start + length];

                var should_filter = false;

                if (hasDigitsAndNonDigits(token)) {
                    stats.mixed_digits_nondigits += 1;
                    should_filter = true;
                } else if (hasLettersEndingInSpace(token)) {
                    stats.letters_ending_space += 1;
                    should_filter = true;
                } else if (hasMoreThanThreeDigits(token)) {
                    stats.more_than_three_digits += 1;
                    should_filter = true;
                } else if (hasLatinWithSpacesNotStartingWithSpace(token)) {
                    stats.latin_spaces_not_starting_space += 1;
                    should_filter = true;
                } else if (try hasInvalidUnicode(allocator, token)) {
                    stats.invalid_unicode += 1;
                    should_filter = true;
                }

                if (!should_filter) {
                    @memcpy(filtered_buffer[filtered_bytes .. filtered_bytes + length], token);
                    filtered_bytes += length;
                    tokens_remaining += 1;
                }
            }

            if (filtered_bytes > 0) {
                try output_file.writeAll(filtered_buffer[0..filtered_bytes]);
            }

            tokens_left -= @intCast(complete_tokens);
        }

        output_header[length_idx] = tokens_remaining;
        stats.remaining_tokens += tokens_remaining;
    }

    try output_file.seekTo(0);
    try output_file.writeAll(std.mem.asBytes(&output_header));

    std.debug.print("\nFiltering Statistics:\n", .{});
    std.debug.print("  Total tokens processed: {d}\n", .{stats.total_tokens});
    std.debug.print("  Tokens with mixed digits and non-digits: {d}\n", .{stats.mixed_digits_nondigits});
    std.debug.print("  Tokens with letters ending in space: {d}\n", .{stats.letters_ending_space});
    std.debug.print("  Tokens with more than three digits: {d}\n", .{stats.more_than_three_digits});
    std.debug.print("  Tokens with invalid Unicode: {d}\n", .{stats.invalid_unicode});
    std.debug.print("  Tokens with Latin chars and spaces not starting with space: {d}\n", .{stats.latin_spaces_not_starting_space});
    std.debug.print("  Remaining tokens: {d} ({d:.2}%)\n", .{
        stats.remaining_tokens,
        @as(f64, @floatFromInt(stats.remaining_tokens)) / @as(f64, @floatFromInt(stats.total_tokens)) * 100.0,
    });
}

fn hasDigitsAndNonDigits(token: []const u8) bool {
    var has_digit = false;
    var has_nondigit = false;

    for (token) |byte| {
        if (byte >= '0' and byte <= '9') {
            has_digit = true;
        } else {
            has_nondigit = true;
        }

        if (has_digit and has_nondigit) {
            return true;
        }
    }

    return false;
}

fn hasLettersEndingInSpace(token: []const u8) bool {
    if (token.len < 2) return false;

    const last_byte = token[token.len - 1];
    if (last_byte != ' ') return false;

    // Check if there's at least one letter in the token
    for (token[0 .. token.len - 1]) |byte| {
        if ((byte >= 'A' and byte <= 'Z') or (byte >= 'a' and byte <= 'z')) {
            return true;
        }
    }

    return false;
}

fn hasMoreThanThreeDigits(token: []const u8) bool {
    var digit_count: usize = 0;

    for (token) |byte| {
        if (byte >= '0' and byte <= '9') {
            digit_count += 1;
            if (digit_count > 3) {
                return true;
            }
        }
    }

    return false;
}

fn hasLatinWithSpacesNotStartingWithSpace(token: []const u8) bool {
    if (token.len < 2) return false;

    const first_byte = token[0];
    if (first_byte == ' ') return false;

    var has_latin = false;
    var has_space = false;

    for (token) |byte| {
        if ((byte >= 'A' and byte <= 'Z') or (byte >= 'a' and byte <= 'z')) {
            has_latin = true;
        } else if (byte == ' ') {
            has_space = true;
        }

        if (has_latin and has_space) {
            return true;
        }
    }

    return false;
}

fn hasInvalidUnicode(allocator: Allocator, token: []const u8) !bool {
    // Check if token starts with 1-3 continuation bytes (10xxxxxx pattern)
    var i: usize = 0;
    var num_continuation_bytes: usize = 0;

    while (i < token.len and i < 3) {
        const byte = token[i];
        if ((byte & 0xC0) == 0x80) {
            num_continuation_bytes += 1;
            i += 1;
        } else {
            break;
        }
    }

    if (num_continuation_bytes == 0 or i >= token.len) {
        return false;
    }

    // Now check the exception case: if all of s, s+"\x80", s+"\x80\x80", s+"\x80\x80\x80"
    // are not valid unicode strings
    const remaining = token[i..];

    // Test s itself
    if (!isValidUtf8(remaining)) {
        // Create test strings by appending continuation bytes
        var test1 = ArrayList(u8).init(allocator);
        defer test1.deinit();
        try test1.appendSlice(remaining);
        try test1.append(0x80);

        var test2 = ArrayList(u8).init(allocator);
        defer test2.deinit();
        try test2.appendSlice(remaining);
        try test2.append(0x80);
        try test2.append(0x80);

        var test3 = ArrayList(u8).init(allocator);
        defer test3.deinit();
        try test3.appendSlice(remaining);
        try test3.append(0x80);
        try test3.append(0x80);
        try test3.append(0x80);

        if (!isValidUtf8(test1.items) and !isValidUtf8(test2.items) and !isValidUtf8(test3.items)) {
            return false;
        }
    }

    // If we get here, the token starts with 1-3 continuation bytes followed by at least one byte,
    // and at least one of the tests produced valid UTF-8, so it matches our ban criteria
    return true;
}

fn isValidUtf8(bytes: []const u8) bool {
    var i: usize = 0;
    while (i < bytes.len) {
        const byte = bytes[i];

        // Single byte character (0xxxxxxx)
        if ((byte & 0x80) == 0) {
            i += 1;
            continue;
        }

        // Determine expected sequence length
        var seq_len: usize = 0;
        if ((byte & 0xE0) == 0xC0) {
            seq_len = 2; // 110xxxxx
        } else if ((byte & 0xF0) == 0xE0) {
            seq_len = 3; // 1110xxxx
        } else if ((byte & 0xF8) == 0xF0) {
            seq_len = 4; // 11110xxx
        } else {
            // Invalid leading byte
            return false;
        }

        if (i + seq_len > bytes.len) {
            return false;
        }

        for (1..seq_len) |j| {
            if ((bytes[i + j] & 0xC0) != 0x80) {
                return false;
            }
        }
        i += seq_len;
    }

    return true;
}
