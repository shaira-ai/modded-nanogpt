const std = @import("std");
const Allocator = std.mem.Allocator;
const fs = std.fs;
const io = std.io;
const mem = std.mem;
const process = std.process;
const ArrayList = std.ArrayList;
const Timer = std.time.Timer;

const MAX_TOKEN_LENGTH = 256;
const DEFAULT_OUTPUT_PATH = "/tmp/tokenset_combined.bin";
const BUFFER_SIZE = 64 * 1024; // 64KB buffer for file operations

pub fn main() !void {
    // Initialize general purpose allocator
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Start timer for performance tracking
    var timer = try Timer.start();

    // Parse command line arguments
    const args = try process.argsAlloc(allocator);
    defer process.argsFree(allocator, args);

    if (args.len < 2 or args.len > 4) {
        try printUsage(args[0]);
        return;
    }

    var input_dir_path: []const u8 = undefined;
    var output_path: []const u8 = DEFAULT_OUTPUT_PATH;

    // Parse args
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
            input_dir_path = args[i];
            i += 1;
        }
    }

    std.debug.print("Input directory: {s}\n", .{input_dir_path});
    std.debug.print("Output file: {s}\n", .{output_path});

    try combineTokenSets(allocator, input_dir_path, output_path);

    const elapsed = timer.read();
    std.debug.print("Combined token sets in {d:.2}s\n", .{@as(f64, @floatFromInt(elapsed)) / std.time.ns_per_s});
}

fn printUsage(program_name: []const u8) !void {
    const stderr = std.io.getStdErr().writer();
    try stderr.print("Usage: {s} [--output OUTPUT_PATH] INPUT_DIRECTORY\n", .{program_name});
    try stderr.print("\nCombines multiple tokenset files into a single file.\n", .{});
    try stderr.print("\nOptions:\n", .{});
    try stderr.print("  --output, -o OUTPUT_PATH  Path to write combined output (default: {s})\n", .{DEFAULT_OUTPUT_PATH});
}

fn combineTokenSets(allocator: Allocator, input_dir_path: []const u8, output_path: []const u8) !void {
    // Open directory
    var dir = try fs.cwd().openDir(input_dir_path, .{ .iterate = true });
    defer dir.close();

    // Collect all tokenset files
    var tokenset_files = ArrayList([]const u8).init(allocator);
    defer {
        for (tokenset_files.items) |path| {
            allocator.free(path);
        }
        tokenset_files.deinit();
    }

    // Iterate through directory
    var dir_it = dir.iterate();
    while (try dir_it.next()) |entry| {
        if (entry.kind != .file) continue;

        // Check if the file is a tokenset file (ends with .bin.zst)
        if (std.mem.endsWith(u8, entry.name, ".bin.zst") and std.mem.startsWith(u8, entry.name, "tokenset_")) {
            const duped_name = try allocator.dupe(u8, entry.name);
            try tokenset_files.append(duped_name);
        }
    }

    // Sort files by tokenset length (extracted from filename)
    std.mem.sort([]const u8, tokenset_files.items, {}, comptime struct {
        fn lessThan(context: void, a: []const u8, b: []const u8) bool {
            _ = context; // Unused
            // Extract length from filename (tokenset_X.bin.zst)
            const a_start = std.mem.indexOf(u8, a, "_") orelse 0;
            const a_end = std.mem.indexOf(u8, a, ".") orelse a.len;
            const b_start = std.mem.indexOf(u8, b, "_") orelse 0;
            const b_end = std.mem.indexOf(u8, b, ".") orelse b.len;

            const a_len_str = a[a_start + 1 .. a_end];
            const b_len_str = b[b_start + 1 .. b_end];

            const a_len = std.fmt.parseInt(usize, a_len_str, 10) catch 0;
            const b_len = std.fmt.parseInt(usize, b_len_str, 10) catch 0;

            return a_len < b_len;
        }
    }.lessThan);

    std.debug.print("Found {d} tokenset files to combine\n", .{tokenset_files.items.len});

    // Create combined header with zeros
    var combined_header = [_]u32{0} ** MAX_TOKEN_LENGTH;

    // First pass: read headers and calculate combined totals
    for (tokenset_files.items) |file_name| {
        std.debug.print("Processing file: {s}\n", .{file_name});

        // Decompress file to temporary location
        const temp_path = try decompressFile(allocator, dir, file_name);
        defer {
            // Clean up temp file
            fs.cwd().deleteFile(temp_path) catch {};
            allocator.free(temp_path);
        }

        // Open decompressed file
        const file = try fs.cwd().openFile(temp_path, .{});
        defer file.close();

        // Read header
        var header: [MAX_TOKEN_LENGTH]u32 = undefined;
        const bytes_read = try file.readAll(std.mem.asBytes(&header));
        if (bytes_read != MAX_TOKEN_LENGTH * @sizeOf(u32)) {
            return error.IncompleteHeader;
        }

        // Add to combined header
        for (header, 0..) |count, i| {
            combined_header[i] += count;
        }
    }

    // Calculate total strings and bytes
    var total_strings: usize = 0;
    var total_bytes_in_strings: usize = 0;
    for (combined_header, 0..) |count, i| {
        total_strings += count;
        total_bytes_in_strings += count * (i + 1); // Each string has length (i+1)
    }

    std.debug.print("Combined totals: {d} strings, {d} bytes of data\n", .{ total_strings, total_bytes_in_strings });

    // Create output file
    const output_file = try fs.cwd().createFile(output_path, .{});
    defer output_file.close();

    // Write combined header
    try output_file.writeAll(std.mem.asBytes(&combined_header));

    // Second pass: copy token data by length
    var buffer = try allocator.alloc(u8, BUFFER_SIZE);
    defer allocator.free(buffer);

    for (combined_header, 0..) |total_count, length_idx| {
        const length = length_idx + 1;
        if (total_count == 0) continue;

        std.debug.print("Processing tokens of length {d} ({d} tokens)...\n", .{ length, total_count });

        // Process each file for this length
        for (tokenset_files.items) |file_name| {
            // Extract token length from filename
            const start = std.mem.indexOf(u8, file_name, "_") orelse 0;
            const end = std.mem.indexOf(u8, file_name, ".") orelse file_name.len;
            const len_str = file_name[start + 1 .. end];
            const file_len = std.fmt.parseInt(usize, len_str, 10) catch 0;

            // Skip if file doesn't contain tokens of current length
            if (file_len != length) continue;

            // Decompress file
            const temp_path = try decompressFile(allocator, dir, file_name);
            defer {
                fs.cwd().deleteFile(temp_path) catch {};
                allocator.free(temp_path);
            }

            // Open decompressed file
            const file = try fs.cwd().openFile(temp_path, .{});
            defer file.close();

            // Read header to get token count
            var header: [MAX_TOKEN_LENGTH]u32 = undefined;
            _ = try file.readAll(std.mem.asBytes(&header));
            const count = header[length_idx];

            if (count == 0) continue;

            // Seek to token data position
            try file.seekTo(@as(usize, MAX_TOKEN_LENGTH) * @sizeOf(u32));

            // Copy tokens to output file
            const total_bytes = count * length;
            var bytes_remaining = total_bytes;

            while (bytes_remaining > 0) {
                const bytes_to_read = @min(bytes_remaining, buffer.len);
                const bytes_read = try file.readAll(buffer[0..bytes_to_read]);
                if (bytes_read == 0) break;

                try output_file.writeAll(buffer[0..bytes_read]);
                bytes_remaining -= bytes_read;
            }
        }
    }

    std.debug.print("Successfully combined {d} tokenset files into {s}\n", .{ tokenset_files.items.len, output_path });
}

fn decompressFile(allocator: Allocator, dir: fs.Dir, file_name: []const u8) ![]const u8 {
    // Create temporary filename for decompressed file
    const temp_filename = try std.fmt.allocPrint(allocator, "/tmp/{s}.decompressed", .{file_name});

    // Get directory path as string
    var pathbuf: [std.fs.max_path_bytes]u8 = undefined;
    const dir_path = try dir.realpath(".", &pathbuf);

    // Full path to input file
    const full_path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ dir_path, file_name });
    defer allocator.free(full_path);

    // Create child process
    var child = std.process.Child.init(&[_][]const u8{
        "zstd", "-d", "-f", "-o", temp_filename, full_path,
    }, allocator);

    // Spawn and wait
    try child.spawn();
    const term = try child.wait();

    // Check exit status
    switch (term) {
        .Exited => |code| {
            if (code != 0) {
                std.debug.print("zstd decompression failed with code {d}\n", .{code});
                return error.DecompressionFailed;
            }
        },
        else => return error.DecompressionFailed,
    }

    return temp_filename;
}
