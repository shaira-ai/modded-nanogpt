const std = @import("std");
const time = std.time;

/// Count-Min Sketch implementation for efficiently approximating string frequencies
/// Uses xxhash for fast hashing and implements conservative updating
pub fn CountMinSketch(
    comptime _width: usize,
    comptime _depth: usize,
) type {
    return struct {
        allocator: std.mem.Allocator,
        hash_seeds: [num_hashes]u64, // Seeds for the hash functions
        counters: [depth][width]usize, // 2D array of counters
        hashes: [100*1024*1024][num_hashes]u64, // Pre-allocated hash buffer
        hash_idx: usize = 0, // Index into the hash buffer

        const depth = _depth;
        const width = _width;
        const num_hashes = (depth * @ctz(width) + 63) / 64;
        const width_mask: usize = width - 1; // Mask for efficient modulo (width - 1)
        const Self = @This();

        /// Initialize a new Count-Min Sketch with the given parameters
        pub fn init(allocator: std.mem.Allocator) !*Self {
            const start_time = time.nanoTimestamp();

            // Ensure width is a power of 2 for efficient modulo
            if (width & (width - 1) != 0) {
                return error.WidthNotPowerOf2;
            }

            const self = try allocator.create(Self);
            errdefer allocator.destroy(self);

            var i: usize = 0;
            while (i < depth) : (i += 1) {
                @memset(&self.counters[i], 0); // Initialize counters to 0
            }

            // Generate seeds for hash functions

            // Use different seeds for each hash function
            const base_seed: u64 = 0xdeadbeef;
            for (0..num_hashes) |j| {
                self.hash_seeds[j] = base_seed ^ @as(u64, @intCast(j * 0x9e3779b9));
            }

            self.allocator = allocator;

            const elapsed = time.nanoTimestamp() - start_time;
            std.debug.print("[TIMING] CountMinSketch.init: {d:.2}ms\n", .{@as(f64, @floatFromInt(elapsed)) / time.ns_per_ms});
            return self;
        }

        /// Free all allocated memory for the Count-Min Sketch
        pub fn deinit(self: *Self) void {
            const start_time = time.nanoTimestamp();

            // Free counters array
            self.allocator.destroy(self);

            const elapsed = time.nanoTimestamp() - start_time;
            std.debug.print("[TIMING] CountMinSketch.deinit: {d:.2}ms\n", .{@as(f64, @floatFromInt(elapsed)) / time.ns_per_ms});
        }

        inline fn readHashFromHashes(hashes: [num_hashes]u64, comptime i: usize) u64 {
            const stoopid: [*]const u8 = @ptrCast(&hashes);
            if (@ctz(width) != 24) {
                @compileError("Width must be a power of 2");
            }
            var ret: u24 = undefined;
            @memcpy(@as(*[3]u8, @ptrCast(&ret)), stoopid+i*3);
            return ret;
        }

        inline fn prefetch(self: *Self, hashes: [num_hashes]u64) void {
            inline for (0..depth) |i| {
                @prefetch(&self.counters[i][readHashFromHashes(hashes, i)], .{});
            }
        }

        inline fn addHashes(self: *Self, hashes: [num_hashes]u64) void {
            var min_value: usize = std.math.maxInt(usize);
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
                self.counters[i][big_hashes[i]] = @min(self.counters[i][big_hashes[i]], min_value);
            }
        }

        /// Add a string with conservative updating
        pub fn conservativeAdd(self: *Self, string: []const u8) !void {
            if (self.hash_idx >= self.hashes.len) {
                @setEvalBranchQuota(1_000_000);
                inline for (16..17) |prefetch_ahead_amt| {
                    const start_time = time.nanoTimestamp();
                    for (0..prefetch_ahead_amt) |i| {
                        self.prefetch(self.hashes[self.hash_idx - prefetch_ahead_amt + i]); 
                    }
                    for (0..self.hash_idx-prefetch_ahead_amt) |i| {
                        self.addHashes(self.hashes[i]); 
                        self.prefetch(self.hashes[i+prefetch_ahead_amt]);
                    }
                    for (self.hash_idx-prefetch_ahead_amt..self.hash_idx) |i| {
                        self.addHashes(self.hashes[i]); 
                    }
                    const elapsed = time.nanoTimestamp() - start_time;
                    std.debug.print("[TIMING] Added {} hashes with prefetch_ahead_amt={}: {d:.2}ms\n", .{ self.hash_idx, prefetch_ahead_amt, @as(f64, @floatFromInt(elapsed)) / time.ns_per_ms });
                }
                self.hash_idx = 0;
            }
            const hash_indices = self.computeHashIndices(string);
            self.hashes[self.hash_idx] = hash_indices;
            self.hash_idx += 1;
        }

        /// Query the approximate frequency of a string
        pub fn query(self: *Self, string: []const u8) !usize {
            const hash_indices = self.computeHashIndices(string);

            // Return the minimum counter value as the frequency estimate
            var min_value: usize = std.math.maxInt(usize);
            for (0..num_hashes) |i| {
                min_value = @min(min_value, self.counters[i][hash_indices[i]]);
            }

            return min_value;
        }

        /// Compute hash indices for a string using xxhash with different seeds
        inline fn computeHashIndices(self: *Self, string: []const u8) [num_hashes]usize {
            var indices: [num_hashes]usize = undefined;

            // Use xxHash with different seeds as specified
            for (0..num_hashes) |i| {
                const hash = std.hash.XxHash3.hash(self.hash_seeds[i], string);

                // Truncate hash to fit the width using mask
                indices[i] = @as(usize, @intCast(hash)) & width_mask;
            }

            return indices;
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
    };
}