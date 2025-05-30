const std = @import("std");
const print = std.debug.print;

const TokenInfo = struct {
    bytes: []const u8,
    count: u64,
    length: usize,
};

const CorpusMetadata = struct {
    file_count: u32,
    timestamp: u64,
    hash_seed: u64,
    file_hashes: []u64,

    pub fn deinit(self: *CorpusMetadata, allocator: std.mem.Allocator) void {
        allocator.free(self.file_hashes);
    }

    pub fn print_summary(self: *const CorpusMetadata) void {
        print("=== Corpus Metadata ===\n", .{});
        print("Files processed: {d}\n", .{self.file_count});
        print("Timestamp: {d}\n", .{self.timestamp});
        print("Hash seed: 0x{x}\n", .{self.hash_seed});

        if (self.file_hashes.len > 0) {
            print("File hashes:\n", .{});
            const max_to_show = @min(10, self.file_hashes.len);
            for (self.file_hashes[0..max_to_show], 0..) |hash, i| {
                print("  File {d}: 0x{x}\n", .{ i, hash });
            }
            if (self.file_hashes.len > max_to_show) {
                print("  ... and {d} more files\n", .{self.file_hashes.len - max_to_show});
            }
        }
        print("\n", .{});
    }
};

const ReadOptions = struct {
    show_stats: bool = false,
    show_metadata: bool = false,
    show_tokens: bool = true,
    top_n: usize = 50,
    min_count: u64 = 0,
    filter_length: ?usize = null,
};

fn readTokensetFile(allocator: std.mem.Allocator, file_path: []const u8, options: ReadOptions) !void {
    const file = try std.fs.cwd().openFile(file_path, .{});
    defer file.close();

    var reader = file.reader();

    // Read format version (new files have version 3+)
    const format_version = reader.readInt(u32, .little) catch |err| {
        if (err == error.EndOfStream) {
            print("Error: File appears to be empty or corrupted\n", .{});
            return;
        }
        return err;
    };

    var has_metadata = false;
    var corpus_metadata: CorpusMetadata = undefined;
    var header: [256]u32 = undefined;

    if (format_version >= 3) {
        // New format with metadata
        has_metadata = true;

        // Read header
        for (&header) |*count| {
            count.* = try reader.readInt(u32, .little);
        }

        // Read corpus metadata
        corpus_metadata.file_count = try reader.readInt(u32, .little);
        corpus_metadata.timestamp = try reader.readInt(u64, .little);
        corpus_metadata.hash_seed = try reader.readInt(u64, .little);

        // Read file hashes
        if (corpus_metadata.file_count > 0) {
            corpus_metadata.file_hashes = try allocator.alloc(u64, corpus_metadata.file_count);
            for (corpus_metadata.file_hashes) |*hash| {
                hash.* = try reader.readInt(u64, .little);
            }
        } else {
            corpus_metadata.file_hashes = try allocator.alloc(u64, 0);
        }
    } else {
        // Old format - treat format_version as first header entry
        has_metadata = false;
        header[0] = format_version;

        // Read rest of header
        for (header[1..]) |*count| {
            count.* = try reader.readInt(u32, .little);
        }
    }

    defer if (has_metadata) corpus_metadata.deinit(allocator);

    // Count total tokens and calculate length distribution
    var total_tokens: usize = 0;
    var length_distribution: [256]u32 = [_]u32{0} ** 256;

    for (header, 0..) |count, length| {
        if (count > 0) {
            total_tokens += count;
            length_distribution[length] = count;
        }
    }

    print("Reading tokenset file: {s}\n", .{file_path});
    if (has_metadata) {
        print("Format version: {d} (with corpus metadata)\n", .{format_version});
    } else {
        print("Format version: legacy (no metadata)\n", .{});
    }
    print("==================================================\n", .{});

    // Show corpus metadata if available and requested
    if (has_metadata and (options.show_metadata or options.show_stats)) {
        corpus_metadata.print_summary();
    }

    print("Total tokens: {d}\n", .{total_tokens});

    if (options.show_stats) {
        print("Length distribution:\n", .{});
        for (length_distribution, 0..) |count, length| {
            if (count > 0) {
                print("  Length {d}: {d} tokens\n", .{ length, count });
            }
        }
        print("\n", .{});
        return; // Just show stats, don't read tokens
    }

    // Filter by length if specified
    if (options.filter_length) |target_length| {
        if (length_distribution[target_length] == 0) {
            print("No tokens found with length {d}\n", .{target_length});
            return;
        }
        print("Filtering to show only length {d} tokens ({d} total)\n", .{ target_length, length_distribution[target_length] });
    } else {
        print("Length distribution:\n", .{});
        for (length_distribution, 0..) |count, length| {
            if (count > 0) {
                print("  Length {d}: {d} tokens\n", .{ length, count });
            }
        }
    }
    print("\n", .{});

    if (!options.show_tokens) {
        return;
    }

    // Read and collect tokens
    var all_tokens = std.ArrayList(TokenInfo).init(allocator);
    defer {
        for (all_tokens.items) |token| {
            allocator.free(token.bytes);
        }
        all_tokens.deinit();
    }

    for (0..256) |length| {
        const count = header[length];
        if (count == 0) continue;

        // Skip if filtering by length
        if (options.filter_length) |target_length| {
            if (length != target_length) {
                // Skip tokens of this length but still need to read past them
                for (0..count) |_| {
                    var buffer: [256]u8 = undefined;
                    const bytes_read = try reader.readAll(buffer[0..length]);
                    if (bytes_read < length) {
                        print("Error: Expected {d} bytes for token, got {d}\n", .{ length, bytes_read });
                        return;
                    }
                    _ = try reader.readInt(u64, .little); // Skip count
                }
                continue;
            }
        }

        for (0..count) |_| {
            // Read token bytes
            const token_bytes = try allocator.alloc(u8, length);
            const bytes_read = try reader.readAll(token_bytes);
            if (bytes_read < length) {
                print("Error: Expected {d} bytes for token, got {d}\n", .{ length, bytes_read });
                allocator.free(token_bytes);
                return;
            }

            // Read occurrence count
            const occurrence_count = try reader.readInt(u64, .little);

            // Apply count filter
            if (occurrence_count >= options.min_count) {
                try all_tokens.append(TokenInfo{
                    .bytes = token_bytes,
                    .count = occurrence_count,
                    .length = length,
                });
            } else {
                allocator.free(token_bytes);
            }
        }
    }

    // Sort tokens by count (descending)
    std.sort.heap(TokenInfo, all_tokens.items, {}, tokenCountCompare);

    // Display results
    const tokens_to_show = @min(options.top_n, all_tokens.items.len);

    if (options.filter_length) |target_length| {
        print("Top {d} tokens of length {d} by occurrence count:\n", .{ tokens_to_show, target_length });
    } else {
        print("Top {d} tokens by occurrence count:\n", .{tokens_to_show});
    }

    if (options.min_count > 0) {
        print("(showing only tokens with count >= {d})\n", .{options.min_count});
    }

    print("  Rank |        Count | Length | Token\n", .{});
    print("----------------------------------------------------------------------\n", .{});

    for (all_tokens.items[0..tokens_to_show], 0..) |token, i| {
        print("  {d:>4} | {d:>11} | {d:>6} | ", .{ i + 1, token.count, token.length });
        printToken(token.bytes);
        print("\n", .{});
    }

    if (all_tokens.items.len > tokens_to_show) {
        print("... and {d} more tokens\n", .{all_tokens.items.len - tokens_to_show});
    }

    // Summary statistics
    if (all_tokens.items.len > 0) {
        const highest_count = all_tokens.items[0].count;
        const lowest_count = all_tokens.items[all_tokens.items.len - 1].count;
        var total_occurrences: u64 = 0;
        for (all_tokens.items) |token| {
            total_occurrences += token.count;
        }
        const avg_occurrences = @as(f64, @floatFromInt(total_occurrences)) / @as(f64, @floatFromInt(all_tokens.items.len));

        print("\nSummary:\n", .{});
        print("  Total tokens matching filters: {d}\n", .{all_tokens.items.len});
        print("  Highest count: {d}\n", .{highest_count});
        print("  Lowest count: {d}\n", .{lowest_count});
        print("  Total occurrences: {d}\n", .{total_occurrences});
        print("  Average occurrences per token: {d:.2}\n", .{avg_occurrences});
    }
}

fn tokenCountCompare(context: void, a: TokenInfo, b: TokenInfo) bool {
    _ = context;
    return a.count > b.count;
}

fn printToken(bytes: []const u8) void {
    print("'", .{});
    for (bytes) |byte| {
        if (std.ascii.isPrint(byte) and byte != '\'') {
            print("{c}", .{byte});
        } else {
            print("\\x{X:0>2}", .{byte});
        }
    }
    print("'", .{});
}

fn printUsage() void {
    print("Usage: tokenset_reader <tokenset_file> [options]\n", .{});
    print("\n", .{});
    print("Options:\n", .{});
    print("  --stats              Show file statistics only\n", .{});
    print("  --metadata           Show corpus metadata only\n", .{});
    print("  --top <N>            Show top N tokens (default: 50)\n", .{});
    print("  --length <L>         Show only tokens of length L\n", .{});
    print("  --min-count <C>      Show only tokens with count >= C\n", .{});
    print("\n", .{});
    print("Examples:\n", .{});
    print("  tokenset_reader tokenset_4.bin\n", .{});
    print("  tokenset_reader tokenset_4.bin --stats\n", .{});
    print("  tokenset_reader tokenset_4.bin --metadata\n", .{});
    print("  tokenset_reader tokenset_4.bin --top 20 --length 4\n", .{});
    print("  tokenset_reader tokenset_4.bin --min-count 1000\n", .{});
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        printUsage();
        return;
    }

    const file_path = args[1];
    var options = ReadOptions{};

    // Parse command line options
    var i: usize = 2;
    while (i < args.len) {
        const arg = args[i];

        if (std.mem.eql(u8, arg, "--stats")) {
            options.show_stats = true;
            options.show_tokens = false;
        } else if (std.mem.eql(u8, arg, "--metadata")) {
            options.show_metadata = true;
            options.show_tokens = false;
        } else if (std.mem.eql(u8, arg, "--top")) {
            if (i + 1 >= args.len) {
                print("Error: --top requires a number\n", .{});
                return;
            }
            i += 1;
            options.top_n = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--length")) {
            if (i + 1 >= args.len) {
                print("Error: --length requires a number\n", .{});
                return;
            }
            i += 1;
            options.filter_length = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--min-count")) {
            if (i + 1 >= args.len) {
                print("Error: --min-count requires a number\n", .{});
                return;
            }
            i += 1;
            options.min_count = try std.fmt.parseInt(u64, args[i], 10);
        } else {
            print("Unknown option: {s}\n", .{arg});
            printUsage();
            return;
        }

        i += 1;
    }

    readTokensetFile(allocator, file_path, options) catch |err| {
        print("Error reading file: {}\n", .{err});
    };
}
