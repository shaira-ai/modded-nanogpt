const std = @import("std");
const bpe = @import("bpe.zig");

pub fn main() !void {
    try bpe.testBPEWithWAR();
}
