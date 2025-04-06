const std = @import("std");

pub const Token = struct {
    bytes: []const u8,
    id: usize,
};

pub const FreqString = struct {
    content: []const u8,
    frequency: usize,
    length: usize,
};

pub const Vocabulary = struct {
    tokens: std.ArrayList(Token),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Vocabulary {
        return .{
            .tokens = std.ArrayList(Token).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Vocabulary) void {
        self.tokens.deinit();
    }
};

pub const TokenCount = struct {
    string_index: usize,
    count: usize,
};

pub const TokenPair = struct {
    first: usize,
    second: usize,
    frequency: usize,

    pub fn lessThan(context: void, a: TokenPair, b: TokenPair) bool {
        _ = context;
        return a.frequency > b.frequency;
    }
};
