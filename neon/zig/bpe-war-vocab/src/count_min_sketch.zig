const std = @import("std");
const time = std.time;

pub const N_LENGTHS = 253;
pub const MY_LEN = 10;

/// Count-Min Sketch implementation for efficiently approximating string frequencies
/// Uses xxhash for fast hashing and implements conservative updating
pub fn CountMinSketch(
    comptime width: usize,
    comptime depth: usize,
) type {
    return struct {
        const seeds: [12]u64 = .{
            0xafe2fa70575ec43d,
            0x7fca6b0ddd27e948,
            0xd858097e82fb9342,
            0xa53c3de53f1daec2,
            0x97a5f9a012f82955,
            0x9af2ffee17c43b83,
            0x7020e7c9fcd727f1,
            0x18469637a034de86,
            0x36a42e718aff893d,
            0x17ba09373d5503c0,
            0x2b11dd16d215db7c,
            0x6c9a257217884407,
        };


        allocator: std.mem.Allocator,
        hash_seeds: [num_hashes]u64, // Seeds for the hash functions
        counters: *[depth][width]u64, // 2D array of counters
        hashes: [4 * 1024 * 1024][num_hashes]u64, // Pre-allocated hash buffer
        hash_idx: usize = 0, // Index into the hash buffer

        pub const num_hashes = (depth * @ctz(width) + 63) / 64;
        const width_mask: usize = width - 1; // Mask for efficient modulo (width - 1)
        const Self = @This();
        const FakeXxHash = @import("xxhash.zig").XxHash3(MY_LEN, MY_LEN, num_hashes);

        /// Initialize a new Count-Min Sketch with the given parameters
        pub fn init(allocator: std.mem.Allocator) !*Self {
            //const start_time = time.nanoTimestamp();

            // Ensure width is a power of 2 for efficient modulo
            if (width & (width - 1) != 0) {
                return error.WidthNotPowerOf2;
            }

            const self = try allocator.create(Self);
            errdefer allocator.destroy(self);
            self.counters = try allocator.create([depth][width]u64);
            errdefer allocator.destroy(self.counters);

            var i: usize = 0;
            while (i < depth) : (i += 1) {
                @memset(&self.counters[i], 0); // Initialize counters to 0
            }

            // Generate seeds for hash functions

            // Use different seeds for each hash function
            if (num_hashes > seeds.len) {
                return error.TooManyHashes;
            }
            for (0..num_hashes) |j| {
                self.hash_seeds[j] = seeds[j];
            }

            self.allocator = allocator;
            self.hash_idx = 0;

            //const elapsed = time.nanoTimestamp() - start_time;
            //_ = elapsed;
            //std.debug.print("[TIMING] CountMinSketch.init: {d:.2}ms\n", .{@as(f64, @floatFromInt(elapsed)) / time.ns_per_ms});
            return self;
        }

        /// Free all allocated memory for the Count-Min Sketch
        pub fn deinit(self: *Self) void {
            //const start_time = time.nanoTimestamp();

            // Free counters array
            self.allocator.destroy(self);

            //const elapsed = time.nanoTimestamp() - start_time;
            //_ = elapsed;
            //std.debug.print("[TIMING] CountMinSketch.deinit: {d:.2}ms\n", .{@as(f64, @floatFromInt(elapsed)) / time.ns_per_ms});
        }

        inline fn readHashFromHashes(hashes: [num_hashes]u64, comptime i: usize) u64 {
            const SmallHashType = std.meta.Int(.unsigned, @ctz(width));
            const WholeHashesType = std.meta.Int(.unsigned, num_hashes * 64);
            const whole_hashes: WholeHashesType = @bitCast(hashes);
            const shift = @ctz(width) * i;
            return @as(SmallHashType, @truncate(whole_hashes >> shift));
        }

        pub inline fn prefetch(self: *Self, hashes: [num_hashes]u64) void {
            inline for (0..depth) |i| {
                @prefetch(&self.counters[i][readHashFromHashes(hashes, i)], .{});
            }
        }

        inline fn addHashes(self: *Self, hashes: [num_hashes]u64) void {
            var min_value: u64 = std.math.maxInt(u64);
            var big_hashes: [depth]u64 = undefined;
            inline for (0..depth) |i| {
                big_hashes[i] = readHashFromHashes(hashes, i);
            }
            inline for (0..depth) |i| {
                min_value = @min(min_value, self.counters[i][big_hashes[i]]);
            }
            min_value += 1;

            // // Conservative update: only increment counters that are equal to the minimum
            inline for (0..depth) |i| {
                self.counters[i][big_hashes[i]] = @max(self.counters[i][big_hashes[i]], min_value);
            }
        }

        /// Add all prefixes of a string with conservative updating
        pub noinline fn addPrefixes(self: *Self, string: [*]const u8, len: usize) void {
            if (len < MY_LEN) {
                return;
            }
            const num_hashes_i_will_add = 1;
            if (self.hash_idx + num_hashes_i_will_add >= self.hashes.len) {
                self.flush();
            }
            FakeXxHash.hash(&self.hashes[self.hash_idx], self.hash_seeds, @as(*const [256]u8, @ptrCast(string)));
            self.hash_idx += 1;
        }

        pub inline fn getHashes(self: *Self, dst: [*]u64, string: [*]const u8) void {
            FakeXxHash.hash(dst, self.hash_seeds, @ptrCast(string));
        }

        /// Flush all remaining strings
        pub noinline fn flush(self: *Self) void {
            @setEvalBranchQuota(1_000_000);
            const guess_prefetch_amt = 60 / depth;
            inline for (guess_prefetch_amt..guess_prefetch_amt+1) |prefetch_ahead_amt| {
                //const start_time = time.nanoTimestamp();
                for (0..prefetch_ahead_amt) |i| {
                    self.prefetch(self.hashes[i]);
                }
                for (0..self.hash_idx -| prefetch_ahead_amt) |i| {
                    self.addHashes(self.hashes[i]);
                    self.prefetch(self.hashes[i + prefetch_ahead_amt]);
                }
                for (self.hash_idx -| prefetch_ahead_amt..self.hash_idx) |i| {
                    self.addHashes(self.hashes[i]);
                }
                //const elapsed = time.nanoTimestamp() - start_time;
                //std.debug.print("[TIMING] Added {} hashes with prefetch_ahead_amt={}: {d:.2}ms\n", .{ self.hash_idx, prefetch_ahead_amt, @as(f64, @floatFromInt(elapsed)) / time.ns_per_ms });
            }
            self.hash_idx = 0;
        }

        pub inline fn queryOne(self: *Self, hashes: [num_hashes]u64) u64 {
            var min_value: u64 = std.math.maxInt(u64);
            var big_hashes: [depth]u64 = undefined;
            inline for (0..depth) |i| {
                big_hashes[i] = readHashFromHashes(hashes, i);
            }
            inline for (0..depth) |i| {
                min_value = @min(min_value, self.counters[i][big_hashes[i]]);
            }
            return min_value;
        }

        /// Query the approximate frequency of a string
        pub noinline fn _query(self: *Self, dst: [*]u64, string: [*]const u8) void {
            var hashes: [N_LENGTHS][num_hashes]u64 = undefined;
            FakeXxHash.hash(@ptrCast(&hashes), self.hash_seeds, @as(*const [256]u8, @ptrCast(string)));
            const prefetch_ahead_amt = 16 * 5 / depth;
            for (0..prefetch_ahead_amt) |i| {
                self.prefetch(hashes[i]);
            }
            for (0..N_LENGTHS -| prefetch_ahead_amt) |i| {
                dst[i] = self.queryOne(hashes[i]);
                self.prefetch(hashes[i + prefetch_ahead_amt]);
            }
            for (N_LENGTHS -| prefetch_ahead_amt..N_LENGTHS) |i| {
                dst[i] = self.queryOne(hashes[i]);
            }
        }
        
        /// Query the approximate frequency of a string
        pub noinline fn query(self: *Self, dst: [*]u64, string: [*]const u8) void {
            var hashes: [N_LENGTHS][num_hashes]u64 = undefined;
            FakeXxHash.hash(@ptrCast(&hashes), self.hash_seeds, @as(*const [256]u8, @ptrCast(string)));
            dst[0] = self.queryOne(hashes[0]);
        }

        /// Compute hash indices for a string using xxhash with different seeds
        inline fn computeHashIndices(self: *Self, string: []const u8) [num_hashes]u64 {
            var ret: [num_hashes]u64 = undefined;
            // Use xxHash with different seeds as specified
            for (0..num_hashes) |i| {
                const hash = std.hash.XxHash3.hash(self.hash_seeds[i], string);
                ret[i] = @truncate(hash);
            }
            return ret;
        }

        /// Reset all counters to zero
        pub fn reset(self: *Self) void {
            for (self.counters) |row| {
                @memset(row, 0);
            }
        }

        /// Merge another Count-Min Sketch into this one
        pub fn merge(self: *Self, other: *Self) !void {
            for (0..depth) |i| {
                for (0..width) |j| {
                    self.counters[i][j] += other.counters[i][j];
                }
            }
        }

        pub fn copyFrom(self: *Self, other: *Self) !void {
            for (0..depth) |i| {
                for (0..width) |j| {
                    self.counters[i][j] = other.counters[i][j];
                }
            }
        }
    };
}
