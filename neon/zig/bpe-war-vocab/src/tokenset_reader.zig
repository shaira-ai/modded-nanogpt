const std = @import("std");
const print = std.debug.print;
const Allocator = std.mem.Allocator;

// Corpus metadata from tokenset files
const CorpusMetadata = struct {
    file_count: u32,
    timestamp: u64,
    hash_seed: u64,
    file_hashes: []u64,

    pub fn deinit(self: *CorpusMetadata, allocator: Allocator) void {
        allocator.free(self.file_hashes);
    }
};

// Token data extracted from tokenset files
const TokenData = struct {
    bytes: []u8,
    count: u64,
    length: usize,

    pub fn deinit(self: *TokenData, allocator: Allocator) void {
        allocator.free(self.bytes);
    }
};

/// Read tokenset file in new format (version 3)
fn readTokensetFile(allocator: Allocator, file_path: []const u8) !struct {
    tokens: []TokenData,
    metadata: CorpusMetadata,
} {
    const file = try std.fs.cwd().openFile(file_path, .{});
    defer file.close();

    var reader = file.reader();

    // Read format version
    const format_version = try reader.readInt(u32, .little);
    if (format_version != 3) {
        print("Error: Expected format version 3, got {d}\n", .{format_version});
        return error.UnsupportedFormat;
    }

    print("Format version: {d}\n", .{format_version});

    // Read header (token counts by length)
    var header: [256]u32 = undefined;
    for (&header) |*count| {
        count.* = try reader.readInt(u32, .little);
    }

    // Read corpus metadata
    var metadata: CorpusMetadata = undefined;
    metadata.file_count = try reader.readInt(u32, .little);
    metadata.timestamp = try reader.readInt(u64, .little);
    metadata.hash_seed = try reader.readInt(u64, .little);

    // Read file hashes
    metadata.file_hashes = try allocator.alloc(u64, metadata.file_count);
    for (metadata.file_hashes) |*hash| {
        hash.* = try reader.readInt(u64, .little);
    }

    // Count total tokens
    var total_tokens: usize = 0;
    for (header) |count| {
        total_tokens += count;
    }

    print("Total tokens: {d}\n", .{total_tokens});

    // Read token data
    var tokens = try allocator.alloc(TokenData, total_tokens);
    var token_idx: usize = 0;

    for (header, 0..) |count, length| {
        if (count == 0) continue;

        for (0..count) |_| {
            // Read token bytes
            const token_bytes = try allocator.alloc(u8, length);
            const bytes_read = try reader.readAll(token_bytes);
            if (bytes_read != length) {
                return error.IncompleteTokenData;
            }

            // Read occurrence count
            const occurrence_count = try reader.readInt(u64, .little);

            tokens[token_idx] = TokenData{
                .bytes = token_bytes,
                .count = occurrence_count,
                .length = length,
            };
            token_idx += 1;
        }
    }

    return .{ .tokens = tokens, .metadata = metadata };
}

/// Print token statistics
fn printTokenStats(tokens: []const TokenData) void {
    print("\n=== TOKEN STATISTICS ===\n");

    // Count by length
    var length_counts: [16]u32 = [_]u32{0} ** 16;
    var total_occurrences: u64 = 0;

    for (tokens) |token| {
        if (token.length < 16) {
            length_counts[token.length] += 1;
        }
        total_occurrences += token.count;
    }

    for (length_counts, 0..) |count, length| {
        if (count > 0) {
            print("Length {d}: {d} tokens\n", .{ length, count });
        }
    }

    print("Total occurrences: {d}\n", .{total_occurrences});
}

/// Print top tokens by occurrence count
fn printTopTokens(tokens: []const TokenData, allocator: Allocator, limit: usize) !void {
    print("\n=== TOP {d} TOKENS BY OCCURRENCE ===\n", .{limit});

    // Create array of indices for sorting
    var indices = try allocator.alloc(usize, tokens.len);
    defer allocator.free(indices);

    for (indices, 0..) |*idx, i| {
        idx.* = i;
    }

    // Sort indices by token occurrence count (descending)
    std.sort.pdq(usize, indices, tokens, struct {
        fn compare(token_list: []const TokenData, a: usize, b: usize) bool {
            return token_list[a].count > token_list[b].count;
        }
    }.compare);

    for (indices[0..@min(limit, indices.len)]) |idx| {
        const token = tokens[idx];
        print("Token {d}: count={d}, length={d}, bytes=", .{ idx, token.count, token.length });

        // Print bytes as hex and try to print as string if printable
        print("[");
        for (token.bytes, 0..) |byte, i| {
            if (i > 0) print(" ");
            print("{:02x}", .{byte});
        }
        print("]");

        // Try to print as string if all bytes are printable
        var all_printable = true;
        for (token.bytes) |byte| {
            if (byte < 32 or byte > 126) {
                all_printable = false;
                break;
            }
        }

        if (all_printable) {
            print(" \"{}\"", .{std.zig.fmtEscapes(token.bytes)});
        }

        print("\n");
    }
}

/// Print usage information
fn printUsage(program_name: []const u8) void {
    print("Usage: {s} [options] <tokenset_file>\n", .{program_name});
    print("\n", .{});
    print("Options:\n", .{});
    print("  --metadata              Show only metadata (default)\n", .{});
    print("  --stats                 Show token statistics\n", .{});
    print("  --top <N>               Show top N tokens by occurrence\n", .{});
    print("  --all                   Show metadata, stats, and top 20 tokens\n", .{});
    print("  --help                  Show this help message\n", .{});
    print("\n", .{});
    print("Examples:\n", .{});
    print("  {s} tokenset_4.bin --metadata\n", .{program_name});
    print("  {s} tokenset_4.bin --stats\n", .{program_name});
    print("  {s} tokenset_4.bin --top 10\n", .{program_name});
    print("  {s} tokenset_4.bin --all\n", .{program_name});
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        printUsage(args[0]);
        return;
    }

    // Parse arguments
    var tokenset_file: ?[]const u8 = null;
    var show_metadata = true;
    var show_stats = false;
    var show_top: ?usize = null;

    var i: usize = 1;
    while (i < args.len) {
        const arg = args[i];

        if (std.mem.eql(u8, arg, "--help")) {
            printUsage(args[0]);
            return;
        } else if (std.mem.eql(u8, arg, "--metadata")) {
            show_metadata = true;
        } else if (std.mem.eql(u8, arg, "--stats")) {
            show_stats = true;
        } else if (std.mem.eql(u8, arg, "--top")) {
            if (i + 1 >= args.len) {
                print("Error: --top requires a number\n", .{});
                return;
            }
            i += 1;
            show_top = std.fmt.parseInt(usize, args[i], 10) catch |err| {
                print("Error: Invalid number for --top: {s} ({any})\n", .{ args[i], err });
                return;
            };
        } else if (std.mem.eql(u8, arg, "--all")) {
            show_metadata = true;
            show_stats = true;
            show_top = 20;
        } else if (std.mem.startsWith(u8, arg, "--")) {
            print("Unknown option: {s}\n", .{arg});
            return;
        } else {
            // It's the tokenset file
            if (tokenset_file != null) {
                print("Error: Multiple tokenset files specified\n", .{});
                return;
            }
            tokenset_file = arg;
        }

        i += 1;
    }

    if (tokenset_file == null) {
        print("Error: No tokenset file specified\n", .{});
        printUsage(args[0]);
        return;
    }

    // Read the tokenset file
    print("Reading tokenset file: {s}\n", .{tokenset_file.?});
    var result = readTokensetFile(allocator, tokenset_file.?) catch |err| {
        print("Error reading tokenset file: {any}\n", .{err});
        return;
    };
    defer {
        for (result.tokens) |*token| {
            token.deinit(allocator);
        }
        allocator.free(result.tokens);
        result.metadata.deinit(allocator);
    }

    // Show metadata
    if (show_metadata) {
        print("\n=== METADATA ===\n");
        print("File count: {d}\n", .{result.metadata.file_count});
        print("Timestamp: {d}\n", .{result.metadata.timestamp});
        print("Hash seed: 0x{x}\n", .{result.metadata.hash_seed});

        print("File hashes:\n");
        for (result.metadata.file_hashes, 0..) |hash, idx| {
            print("  File {d}: 0x{x}\n", .{ idx, hash });
        }
    }

    // Show stats
    if (show_stats) {
        printTokenStats(result.tokens);
    }

    // Show top tokens
    if (show_top) |limit| {
        try printTopTokens(result.tokens, allocator, limit);
    }
}
