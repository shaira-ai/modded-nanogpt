const std = @import("std");
const EMPTY_SLICE: []const u8 = "";

pub fn HashTable(comptime RHT_POW: u6) type {
    const RHT_LEN: usize = @as(usize, 1) << RHT_POW;
    const RHT_LEN_EXTENDED: usize = RHT_LEN * 2;
    const RHT_MASK: usize = RHT_LEN - 1;

    const Entry = struct {
        key: []const u8,
        hash: u64,
        value: u64,
    };

    return struct {
        const Self = @This();
        xs: [*]Entry,
        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator) !Self {
            const xs = try allocator.alloc(Entry, RHT_LEN_EXTENDED);
            for (xs) |*ptr| {
                ptr.* = .{
                    .key = EMPTY_SLICE,
                    .hash = 0,
                    .value = 0,
                };
            }
            return .{
                .xs = xs.ptr,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.xs[0..RHT_LEN_EXTENDED]);
        }

        pub inline fn prefetch(self: *Self, hash: u64) void {
            @prefetch(&self.xs[hash & RHT_MASK], .{});
        }

        pub fn getPtr(self: *Self, key: []const u8, hash: u64) ?*u64 {
            var bucknum: u64 = hash & RHT_MASK;
            const xs = self.xs;
            while (true) {
                if (xs[bucknum].hash == hash and std.mem.eql(u8, xs[bucknum].key, key)) {
                    return &xs[bucknum].value;
                }
                if (xs[bucknum].key.len == 0) {
                    return null;
                }
                bucknum +%= 1;
            }
        }

        inline fn insertKnownNotPresentInner(
            noalias xs: [*]Entry,
            noalias key_: []const u8,
            hash_: u64,
            value_: u64,
        ) void {
            var key = key_;
            var hash = hash_;
            var value = value_;
            var home_bucknum = hash & RHT_MASK;
            var bucknum = home_bucknum;
            while (true) {
                if (xs[bucknum].key.len == 0) {
                    xs[bucknum].key = key;
                    xs[bucknum].hash = hash;
                    xs[bucknum].value = value;
                    return;
                }
                const this_guys_home_bucknum = xs[bucknum].hash & RHT_MASK;
                if (this_guys_home_bucknum > home_bucknum) {
                    const tmp = xs[bucknum];
                    xs[bucknum] = .{
                        .key = key,
                        .hash = hash,
                        .value = value,
                    };
                    key = tmp.key;
                    hash = tmp.hash;
                    value = tmp.value;
                    home_bucknum = this_guys_home_bucknum;
                }
                bucknum +%= 1;
            }
        }

        pub fn insertKnownNotPresent(self: *Self, key: []const u8, hash: u64, value: u64) void {
            return insertKnownNotPresentInner(self.xs, key, hash, value);
        }

        inline fn deleteKnownPresentInner(
            noalias xs: [*]Entry,
            noalias key: []const u8,
            hash: u64,
        ) u64 {
            var bucknum = hash & RHT_MASK;
            var ret: u64 = 0;
            while (true) {
                if (xs[bucknum].hash == hash and std.mem.eql(u8, xs[bucknum].key, key)) {
                    ret = xs[bucknum].value;
                    break;
                }
                bucknum +%= 1;
            }
            while (true) {
                const next_bucknum = bucknum +% 1;
                const next_hash = xs[next_bucknum].hash;
                const next_len = xs[next_bucknum].key.len;
                const next_home_bucknum = next_hash & RHT_MASK;
                if (next_len == 0 or next_home_bucknum == next_bucknum) {
                    break;
                }
                xs[bucknum] = xs[next_bucknum];
                bucknum = next_bucknum;
            }
            xs[bucknum] = .{
                .key = EMPTY_SLICE,
                .hash = 0,
                .value = 0,
            };
            return ret;
        }

        pub fn deleteKnownPresent(self: *Self, key: []const u8, hash: u64) u64 {
            return deleteKnownPresentInner(self.xs, key, hash);
        }
    };
}
