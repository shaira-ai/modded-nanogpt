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

// Stats header format expected by vocab_learner
const STATS_MAGIC = "TOKSTAT".*;
const StatsHeader = extern struct {
    magic: [7]u8,
    pad_a: [1]u8,
    vocab_size: u32,
    n_token_ids: u32,
    file_count: u32,
    timestamp: i64,
    hash_seed: u64,
    reserved: [20]u8,

    comptime {
        if (@sizeOf(StatsHeader) != 64) {
            @compileError("StatsHeader must be 64 bytes");
        }
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

/// Generate both tokenset and corpus stats files with consistent token ordering
fn generateOutputFiles(allocator: Allocator, all_tokens: []const []const TokenData, metadata: *const CorpusMetadata, output_tokenset: []const u8, output_stats: []const u8) !void {
    print("Generating output files with consistent token ordering...\n", .{});

    // Count tokens by length and organize them
    var length_counts: [256]u32 = [_]u32{0} ** 256;
    var tokens_by_length: [256]std.ArrayList(TokenData) = undefined;
    for (0..256) |i| {
        tokens_by_length[i] = std.ArrayList(TokenData).init(allocator);
    }
    defer {
        for (&tokens_by_length) |*list| {
            list.deinit();
        }
    }

    // Organize tokens by length
    for (all_tokens) |tokens| {
        for (tokens) |token| {
            if (token.length > 0 and token.length < 256) {
                try tokens_by_length[token.length].append(token);
                length_counts[token.length - 1] += 1; // vocab_learner indexing: header[i] = count of length-(i+1) tokens
            }
        }
    }

    // Add missing single-byte tokens
    var all_byte_values: [256]bool = [_]bool{false} ** 256;
    for (tokens_by_length[1].items) |token| {
        if (token.bytes.len == 1) {
            all_byte_values[token.bytes[0]] = true;
        }
    }

    var missing_tokens = std.ArrayList(TokenData).init(allocator);
    defer {
        for (missing_tokens.items) |token| {
            allocator.free(token.bytes);
        }
        missing_tokens.deinit();
    }

    for (0..256) |byte_val| {
        if (!all_byte_values[byte_val]) {
            const missing_byte = try allocator.alloc(u8, 1);
            missing_byte[0] = @intCast(byte_val);
            const missing_token = TokenData{
                .bytes = missing_byte,
                .count = 0, // Missing tokens have zero count
                .length = 1,
            };
            try tokens_by_length[1].append(missing_token);
            try missing_tokens.append(missing_token);
            length_counts[0] += 1; // length-1 tokens in header[0]
            print("Warning: Added missing single-byte token: 0x{x}\n", .{byte_val});
        }
    }

    // Calculate final token count
    var final_token_count: u32 = 0;
    for (length_counts) |count| {
        final_token_count += count;
    }

    print("Final token count: {d}\n", .{final_token_count});

    // Generate tokenset file
    {
        const file = try std.fs.cwd().createFile(output_tokenset, .{});
        defer file.close();
        var writer = file.writer();

        // Write header
        for (length_counts) |count| {
            try writer.writeInt(u32, count, .little);
        }

        // Write tokens in order by length
        for (1..256) |length| {
            for (tokens_by_length[length].items) |token| {
                try writer.writeAll(token.bytes);
            }
        }

        print("Generated tokenset with {d} tokens\n", .{final_token_count});
    }

    // Generate corpus stats file
    {
        const file = try std.fs.cwd().createFile(output_stats, .{});
        defer file.close();
        var writer = file.writer();

        // Write header
        const header = StatsHeader{
            .magic = STATS_MAGIC,
            .pad_a = [_]u8{0},
            .vocab_size = 0,
            .n_token_ids = final_token_count,
            .file_count = metadata.file_count,
            .timestamp = @intCast(metadata.timestamp),
            .hash_seed = metadata.hash_seed,
            .reserved = [_]u8{0} ** 20,
        };
        try writer.writeAll(std.mem.asBytes(&header));

        // Write file hashes in original order (don't sort!)
        for (metadata.file_hashes) |hash| {
            try writer.writeInt(u64, hash, .little);
        }

        // Write token stats in same order as tokenset file
        var token_id: u32 = 0;
        for (1..256) |length| {
            for (tokens_by_length[length].items) |token| {
                // Write token ID
                try writer.writeInt(u32, token_id, .little);

                // Write occurrence count
                try writer.writeInt(u64, token.count, .little);

                // Write estimated savings
                const est_savings = @as(f64, @floatFromInt(token.count * (@max(1, token.length) - 1)));
                const savings_bits: u64 = @bitCast(est_savings);
                try writer.writeInt(u64, savings_bits, .little);

                // Write sampled data (zeros)
                try writer.writeInt(u64, 0, .little); // sampled_occurrences
                try writer.writeInt(u64, 0, .little); // sampled_savings
                try writer.writeInt(u64, 0, .little); // sampled_step

                token_id += 1;
            }
        }

        print("Generated corpus stats for {d} tokens\n", .{final_token_count});
    }
}

/// Print usage information
fn printUsage(program_name: []const u8) void {
    print("Usage: {s} [options] tokenset_2.bin tokenset_3.bin tokenset_4.bin ...\n", .{program_name});
    print("\n", .{});
    print("Options:\n", .{});
    print("  --output-tokenset <path>    Output path for combined tokenset (default: tokenset_combined.bin)\n", .{});
    print("  --output-stats <path>       Output path for corpus stats (default: corpus_stats.bin)\n", .{});
    print("  --help                      Show this help message\n", .{});
    print("\n", .{});
    print("Example:\n", .{});
    print("  {s} tokenset_2.bin tokenset_3.bin tokenset_4.bin\n", .{program_name});
    print("\n", .{});
    print("This tool converts new-format tokenset files (with metadata and occurrence counts)\n", .{});
    print("into the old format expected by vocab_learner.zig, plus a corpus_stats.bin file.\n", .{});
    print("\n", .{});
    print("Note: File hashes are now compatible between top-k and vocab_learner (no conversion needed).\n", .{});
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
    var tokenset_files = std.ArrayList([]const u8).init(allocator);
    defer tokenset_files.deinit();

    var output_tokenset: []const u8 = "tokenset_combined.bin";
    var output_stats: []const u8 = "corpus_stats.bin";

    var i: usize = 1;
    while (i < args.len) {
        const arg = args[i];

        if (std.mem.eql(u8, arg, "--help")) {
            printUsage(args[0]);
            return;
        } else if (std.mem.eql(u8, arg, "--output-tokenset")) {
            if (i + 1 >= args.len) {
                print("Error: --output-tokenset requires a value\n", .{});
                return;
            }
            i += 1;
            output_tokenset = args[i];
        } else if (std.mem.eql(u8, arg, "--output-stats")) {
            if (i + 1 >= args.len) {
                print("Error: --output-stats requires a value\n", .{});
                return;
            }
            i += 1;
            output_stats = args[i];
        } else if (std.mem.startsWith(u8, arg, "--")) {
            print("Unknown option: {s}\n", .{arg});
            return;
        } else {
            // It's a tokenset file
            try tokenset_files.append(arg);
        }

        i += 1;
    }

    if (tokenset_files.items.len == 0) {
        print("Error: No tokenset files specified\n", .{});
        printUsage(args[0]);
        return;
    }

    print("Converting {d} tokenset files to vocab_learner format...\n", .{tokenset_files.items.len});

    // Read all tokenset files
    var all_tokens = try allocator.alloc([]TokenData, tokenset_files.items.len);
    defer {
        for (all_tokens) |tokens| {
            for (tokens) |*token| {
                token.deinit(allocator);
            }
            allocator.free(tokens);
        }
        allocator.free(all_tokens);
    }

    var metadata: ?CorpusMetadata = null;
    defer if (metadata) |*m| m.deinit(allocator);

    for (tokenset_files.items, 0..) |file_path, idx| {
        print("Reading {s}...\n", .{file_path});

        var result = try readTokensetFile(allocator, file_path);
        all_tokens[idx] = result.tokens;

        if (metadata == null) {
            metadata = result.metadata;
        } else {
            // Verify metadata matches
            if (metadata.?.file_count != result.metadata.file_count or
                metadata.?.hash_seed != result.metadata.hash_seed)
            {
                print("Error: Metadata mismatch between tokenset files\n", .{});
                return;
            }
            result.metadata.deinit(allocator);
        }

        print("  Loaded {d} tokens\n", .{result.tokens.len});
    }

    // Generate output files with consistent ordering
    try generateOutputFiles(allocator, all_tokens, &metadata.?, output_tokenset, output_stats);

    print("\nConversion complete!\n", .{});
    print("  Output tokenset: {s}\n", .{output_tokenset});
    print("  Output stats: {s}\n", .{output_stats});
    print("\nYou can now use these files with vocab_learner.zig:\n", .{});
    print("  ./vocab_learner {s} <corpus_path> --stats-path {s}\n", .{ output_tokenset, output_stats });
    print("\nFile hashes are now compatible between top-k and vocab_learner!\n", .{});
}
