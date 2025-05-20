const std = @import("std");

// Constants from vocab_learner.zig
const VOCAB_MAGIC = "VOCA".*;
const VOCAB_VERSION: u32 = 1;
const HEADER_SIZE = 32;

const VocabHeader = struct {
    magic: [4]u8,
    version: u32,
    vocab_size: u32,
    reserved: [20]u8,
};

const InspectorError = error{
    IncompleteHeader,
    InvalidMagicNumber,
    UnsupportedVersion,
    BufferTooSmall,
    IncompleteToken,
    InvalidTokenSequence,
};

fn printToken(token: []const u8) void {
    std.debug.print("\"", .{});
    for (token) |byte| {
        if (byte >= 32 and byte < 127) {
            std.debug.print("{c}", .{byte});
        } else {
            std.debug.print("\\x{x:0>2}", .{byte});
        }
    }
    std.debug.print("\"", .{});
}

fn countTokensByLength(tokens: []const []const u8) void {
    var counts = std.AutoHashMap(usize, usize).init(std.heap.page_allocator);
    defer counts.deinit();

    // Count tokens by length
    for (tokens) |token| {
        const result = counts.getOrPut(token.len) catch unreachable;
        if (result.found_existing) {
            result.value_ptr.* += 1;
        } else {
            result.value_ptr.* = 1;
        }
    }

    std.debug.print("\nToken length distribution:\n", .{});
    var it = counts.iterator();
    while (it.next()) |entry| {
        const percentage = @as(f64, @floatFromInt(entry.value_ptr.*)) /
            @as(f64, @floatFromInt(tokens.len)) * 100.0;
        std.debug.print("  Length {d}: {d} tokens ({d:.1}%)\n", .{ entry.key_ptr.*, entry.value_ptr.*, percentage });
    }
}

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    // Parse command line arguments
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        std.debug.print("Usage: {s} <vocab_file>\n", .{args[0]});
        return;
    }

    const vocab_path = args[1];

    std.debug.print("Inspecting vocabulary file: {s}\n", .{vocab_path});
    std.debug.print("=====================================\n\n", .{});

    // Open file
    const file = std.fs.cwd().openFile(vocab_path, .{}) catch |err| {
        std.debug.print("Error opening file: {}\n", .{err});
        return;
    };
    defer file.close();

    // Read and validate header
    var header: VocabHeader = undefined;
    const bytes_read = file.readAll(std.mem.asBytes(&header)) catch |err| {
        std.debug.print("Error reading header: {}\n", .{err});
        return;
    };

    if (bytes_read != @sizeOf(VocabHeader)) {
        std.debug.print("Incomplete header: read {d} bytes, expected {d}\n", .{ bytes_read, @sizeOf(VocabHeader) });
        return;
    }

    // Print header info
    std.debug.print("Header Information:\n", .{});
    std.debug.print("  Magic: ", .{});
    for (header.magic) |byte| std.debug.print("{c}", .{byte});
    std.debug.print("\n", .{});
    std.debug.print("  Version: {d}\n", .{header.version});

    // Validate magic number
    if (!std.mem.eql(u8, &header.magic, &VOCAB_MAGIC)) {
        std.debug.print("\nWARNING: Invalid magic number! Expected 'VOCA'\n", .{});
    }

    // Check version compatibility
    if (header.version != VOCAB_VERSION) {
        std.debug.print("\nWARNING: Unsupported version! Expected {d}, found {d}\n", .{ VOCAB_VERSION, header.version });
    }

    // Allocate space for all tokens
    var tokens = std.ArrayList([]const u8).init(allocator);
    defer {
        for (tokens.items) |token| {
            allocator.free(token);
        }
        tokens.deinit();
    }

    // Read and display tokens
    std.debug.print("\nTokens:\n", .{});
    var token_id: u32 = 0;
    while (token_id < header.vocab_size) {
        var id_bytes: [4]u8 = undefined;
        var len_bytes: [4]u8 = undefined;

        // Read token ID and length
        const id_read = file.readAll(&id_bytes) catch |err| {
            std.debug.print("Error reading token ID: {}\n", .{err});
            return;
        };
        const len_read = file.readAll(&len_bytes) catch |err| {
            std.debug.print("Error reading token length: {}\n", .{err});
            return;
        };

        if (id_read != 4 or len_read != 4) {
            std.debug.print("Incomplete token header at token {d}\n", .{token_id});
            return;
        }

        const read_token_id = std.mem.readInt(u32, &id_bytes, .little);
        const token_length = std.mem.readInt(u32, &len_bytes, .little);

        // Validate token ID is sequential
        if (read_token_id != token_id) {
            std.debug.print("Invalid token sequence: expected {d}, found {d}\n", .{ token_id, read_token_id });
            return;
        }

        // Read token content
        const token = allocator.alloc(u8, token_length) catch |err| {
            std.debug.print("Error allocating token memory: {}\n", .{err});
            return;
        };
        errdefer allocator.free(token);

        const token_bytes_read = file.readAll(token) catch |err| {
            std.debug.print("Error reading token content: {}\n", .{err});
            return;
        };

        if (token_bytes_read != token_length) {
            std.debug.print("Incomplete token content at token {d}\n", .{token_id});
            allocator.free(token);
            return;
        }

        // Print token info
        std.debug.print("  ID {d:3}: ", .{token_id});
        printToken(token);
        std.debug.print(" (len: {d})\n", .{token_length});

        // Store token for statistics
        tokens.append(token) catch {
            allocator.free(token);
            continue;
        };

        token_id += 1;
    }
    std.debug.print("\n\nVocabulary Size: {d}\n", .{header.vocab_size});
    // Print statistics
    countTokensByLength(tokens.items);

    // Check if file has extra data
    var dummy: [1]u8 = undefined;
    const extra_bytes = file.readAll(&dummy) catch 0;
    if (extra_bytes > 0) {
        std.debug.print("\nWARNING: File contains extra data after vocabulary!\n", .{});
    }

    std.debug.print("\nFile inspection complete.\n", .{});
}
