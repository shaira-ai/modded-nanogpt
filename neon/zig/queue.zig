const std = @import("std");
const Allocator = std.mem.Allocator;

pub fn Queue(comptime T: type) type {
    return struct {
        const Self = @This();
        items: []T = &([_]T{}),
        head: usize = 0,
        len: usize = 0,
        pub fn deinit(self: *Self, allocator: Allocator) void {
            allocator.free(self.items);
        }
        pub fn push(self: *Self, allocator: Allocator, item: T) !void {
            if (self.len == self.items.len) {
                if (self.items.len == 0) {
                    self.items = try allocator.alloc(T, 1024);
                } else {
                    const new_items = try allocator.alloc(T, self.items.len * 2);
                    const first_chunk_len = self.items.len - self.head;
                    @memcpy(new_items[0..first_chunk_len], self.items[self.head..]);
                    @memcpy(new_items[first_chunk_len..self.items.len], self.items[0..self.head]);
                    self.head = 0;
                    allocator.free(self.items);
                    self.items = new_items;
                }
            }
            const tail = (self.head + self.len) & (self.items.len - 1);
            self.items[tail] = item;
            self.len += 1;
        }
        pub fn pop(self: *Self) ?T {
            if (self.is_empty()) return null;
            const value = self.items[self.head];
            self.head = (self.head + 1) & (self.items.len - 1);
            self.len -= 1;
            return value;
        }
        pub fn is_empty(self: Self) bool {
            return self.len == 0;
        }
        pub fn peek(self: Self) ?T {
            if (self.is_empty()) return null;
            return self.items[self.head];
        }
    };
}
