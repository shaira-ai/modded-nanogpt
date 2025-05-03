const std = @import("std");
const Queue = @import("queue.zig").Queue;
const Allocator = std.mem.Allocator;

const StateInfo = struct {
    blue: u32,
    green: u32,
    depth: u32,
    token_id: u32,
};

pub const BakaCorasick = struct {
    transitions: [*][256]u32,
    info: [*]StateInfo,
    len: u64,
    capacity: u64,
    allocator: Allocator,

    pub const NO_TOKEN = ~@as(u32, 0);
    const N_MATCH_ARRS = 10;
    const Self = @This();

    pub fn init(allocator: Allocator) !Self {
        const capacity = 1024;
        const transitions = try allocator.alloc([256]u32, capacity);
        const info = try allocator.alloc(StateInfo, capacity);
        transitions[0] = .{0} ** 256;
        info[0] = .{
            .blue = 0,
            .green = 0,
            .depth = 0,
            .token_id = NO_TOKEN,
        };
        return BakaCorasick{
            .transitions = transitions.ptr,
            .info = info.ptr,
            .len = 1,
            .capacity = capacity,
            .allocator = allocator,
        };
    }

    // Allocate a new state
    fn allocState(self: *Self) !u32 {
        // We need to allocate a new state
        if (self.len >= self.capacity) {
            if (self.len >= 0x100000000) {
                return error.TooManyStates;
            }
            // Grow the arrays
            const new_capacity = self.capacity * 2;
            const new_transitions = try self.allocator.alloc([256]u32, new_capacity);
            const new_info = try self.allocator.alloc(StateInfo, new_capacity);

            // Copy existing data
            @memcpy(new_transitions[0..self.capacity], self.transitions[0..self.capacity]);
            @memcpy(new_info[0..self.capacity], self.info[0..self.capacity]);

            // Free old arrays
            self.allocator.free(self.transitions[0..self.capacity]);
            self.allocator.free(self.info[0..self.capacity]);

            // Update pointers and capacity
            self.transitions = new_transitions.ptr;
            self.info = new_info.ptr;
            self.capacity = new_capacity;
        }
        const state_id = @as(u32, @intCast(self.len));
        self.len += 1;
        return state_id;
    }

    pub fn copyFrom(self: *Self, other: *Self) !void {
        if (self.capacity < other.capacity) {
            self.allocator.free(self.transitions[0..self.capacity]);
            self.allocator.free(self.info[0..self.capacity]);
            self.capacity = other.capacity;
            self.transitions = (try self.allocator.alloc([256]u32, self.capacity)).ptr;
            self.info = (try self.allocator.alloc(StateInfo, self.capacity)).ptr;
        }
        self.len = other.len;
        @memcpy(self.transitions[0..self.len], other.transitions[0..self.len]);
        @memcpy(self.info[0..self.len], other.info[0..self.len]);
    }

    // Insert a new token/word (only builds the trie structure without suffix links)
    pub fn insert(self: *Self, token_str: []const u8, token_id: u32) !void {
        var current_state: u32 = 0; // Start at root

        // Build the trie path for this token
        for (token_str, 0..) |c, i| {
            const my_depth = self.info[current_state].depth;
            const old_next = self.transitions[current_state][c];
            const old_next_depth = self.info[old_next].depth;
            if (old_next_depth <= my_depth) {
                // Create a new state
                const next_state = try self.allocState();
                self.transitions[current_state][c] = next_state;

                // Initialize with basic info (without suffix links)
                self.info[next_state] = .{
                    .blue = 0, // Will be computed later with computeSuffixLinks
                    .green = 0, // Will be computed later
                    .depth = @intCast(i + 1),
                    .token_id = NO_TOKEN,
                };
                self.transitions[next_state] = .{0} ** 256;
            }

            current_state = self.transitions[current_state][c];
        }

        // Mark the final state as the end of the token
        self.info[current_state].token_id = token_id;
    }

    // Compute all suffix links (blue and green arcs) using BFS
    pub fn computeSuffixLinks(self: *Self) !void {
        var queue = Queue(u32){};
        defer queue.deinit(self.allocator);

        // Initialize BFS with the root's children
        // The root's suffix link is already set to 0 (itself)
        for (0..256) |i| {
            const c: u8 = @intCast(i);
            const child = self.transitions[0][c];
            const my_depth = self.info[0].depth;
            if (child != 0) {
                const child_depth = self.info[child].depth;
                if (child_depth > my_depth) {
                    try queue.push(self.allocator, child);
                }
            }
        }

        // BFS traversal to compute suffix links
        while (queue.pop()) |current| {
            var dict_suffix = self.info[current].blue;
            // Follow blue arcs until finding a final state
            while (dict_suffix != 0 and self.info[dict_suffix].token_id == NO_TOKEN) {
                dict_suffix = self.info[dict_suffix].blue;
            }
            self.info[current].green = dict_suffix;
            const current_suffix = self.info[current].blue;
            const my_depth = self.info[current].depth;
            // Add all children to the queue and compute their suffix links
            for (0..256) |i| {
                const c: u8 = @intCast(i);
                const child = self.transitions[current][c];

                if (child != 0) {
                    const child_depth = self.info[child].depth;
                    if (child_depth > my_depth) {
                        // Start from the suffix link of the current node
                        var suffix_node = current_suffix;

                        // Follow suffix links until finding a node that has the current character
                        // or until reaching the root
                        while (suffix_node != 0 and self.transitions[suffix_node][c] == 0) {
                            suffix_node = self.info[suffix_node].blue;
                        }

                        // Set the suffix link for the child
                        self.info[child].blue = self.transitions[suffix_node][c];

                        // Add child to the queue
                        try queue.push(self.allocator, child);
                    }
                }
            }
        }

        // densify the transitions
        for (0..256) |i| {
            const c: u8 = @intCast(i);
            const child = self.transitions[0][c];
            const my_depth = self.info[0].depth;
            if (child != 0) {
                const child_depth = self.info[child].depth;
                if (child_depth > my_depth) {
                    try queue.push(self.allocator, child);
                }
            }
        }
        while (queue.pop()) |current| {
            const current_suffix = self.info[current].blue;
            const my_depth = self.info[current].depth;
            for (0..256) |i| {
                const c: u8 = @intCast(i);
                const child = self.transitions[current][c];
                if (self.info[child].depth > my_depth) {
                    try queue.push(self.allocator, child);
                } else {
                    var suffix_node = current_suffix;
                    while (suffix_node != 0 and self.transitions[suffix_node][c] == 0) {
                        suffix_node = self.info[suffix_node].blue;
                    }
                    self.transitions[current][c] = self.transitions[suffix_node][c];
                }
            }
        }
    }
};
