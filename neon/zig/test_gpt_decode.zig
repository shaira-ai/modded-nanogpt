const std = @import("std");
const io = std.io;
const gpt_encode = @import("gpt_encode.zig");

var in_buffer: [100 * 1024 * 1024]u8 = undefined;
var out_buffer: [100 * 1024 * 1024]u8 = undefined;

pub fn main() !void {
    const stdin = io.getStdIn();
    const stdout = io.getStdOut();
    
    // Read all input
    const bytes_read = try stdin.reader().readAll(&in_buffer);
    if (bytes_read == 0) return;
    
    // Decode and write to stdout
    const encoded = try gpt_encode.encode(&out_buffer, in_buffer[0..bytes_read]);
    try stdout.writer().writeAll(encoded);
}
