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

pub const StringInfo = struct {
    segmentation: []usize,
    freq_string: *const FreqString,
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
