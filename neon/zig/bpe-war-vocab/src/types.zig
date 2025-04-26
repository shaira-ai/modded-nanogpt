const std = @import("std");

pub const Token = struct {
    bytes: []const u8,
    id: usize,
};

pub const FreqString = struct {
    content: []const u8,
    frequency: usize,
    length: usize,

    pub fn compare(context: void, a: FreqString, b: FreqString) std.math.Order {
        _ = context;
        return std.math.order(b.frequency, a.frequency);
    }
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

pub const TokenCandidate = struct {
    token_id: usize,
    lower_bound_war: f64,
    upper_bound_war: f64,
    doc_contributions: std.AutoHashMap(usize, f64),
    extra_data: std.StringHashMap(f64), // Changed to StringHashMap

    pub fn init(allocator: std.mem.Allocator, token_id: usize) !*TokenCandidate {
        const self = try allocator.create(TokenCandidate);
        self.* = .{
            .token_id = token_id,
            .lower_bound_war = 0,
            .upper_bound_war = 0,
            .doc_contributions = std.AutoHashMap(usize, f64).init(allocator),
            .extra_data = std.StringHashMap(f64).init(allocator), // Initialize as StringHashMap
        };
        return self;
    }

    pub fn deinit(self: *TokenCandidate, allocator: std.mem.Allocator) void {
        self.doc_contributions.deinit();
        self.extra_data.deinit();
        allocator.destroy(self);
    }
};
