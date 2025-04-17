const std = @import("std");
const time = std.time;

/// Count-Min Sketch implementation for efficiently approximating string frequencies
/// Uses xxhash for fast hashing and implements conservative updating
pub const CountMinSketch = struct {
    allocator: std.mem.Allocator,
    width: usize, // Number of counters per hash function (should be power of 2)
    depth: usize, // Number of hash functions
    width_mask: usize, // Mask for efficient modulo (width - 1)
    counters: [][]usize, // 2D array of counters
    hash_seeds: []u64, // Seeds for the hash functions

    /// Initialize a new Count-Min Sketch with the given parameters
    pub fn init(allocator: std.mem.Allocator, width: usize, depth: usize) !*CountMinSketch {
        const start_time = time.nanoTimestamp();

        // Ensure width is a power of 2 for efficient modulo
        if (width & (width - 1) != 0) {
            return error.WidthNotPowerOf2;
        }

        const self = try allocator.create(CountMinSketch);
        errdefer allocator.destroy(self);

        // Allocate counters array (2D)
        const counters = try allocator.alloc([]usize, depth);
        errdefer allocator.free(counters);

        // Initialize each row of counters
        var i: usize = 0;
        errdefer {
            while (i > 0) {
                i -= 1;
                allocator.free(counters[i]);
            }
        }

        while (i < depth) : (i += 1) {
            counters[i] = try allocator.alloc(usize, width);
            @memset(counters[i], 0); // Initialize counters to 0
        }

        // Generate seeds for hash functions
        const hash_seeds = try allocator.alloc(u64, depth);
        errdefer allocator.free(hash_seeds);

        // Use different seeds for each hash function
        const base_seed: u64 = 0xdeadbeef;
        for (0..depth) |j| {
            hash_seeds[j] = base_seed ^ @as(u64, @intCast(j * 0x9e3779b9));
        }

        self.* = .{
            .allocator = allocator,
            .width = width,
            .depth = depth,
            .width_mask = width - 1, // Mask for efficient modulo
            .counters = counters,
            .hash_seeds = hash_seeds,
        };

        const elapsed = time.nanoTimestamp() - start_time;
        std.debug.print("[TIMING] CountMinSketch.init: {d:.2}ms\n", .{@as(f64, @floatFromInt(elapsed)) / time.ns_per_ms});

        return self;
    }

    /// Free all allocated memory for the Count-Min Sketch
    pub fn deinit(self: *CountMinSketch) void {
        const start_time = time.nanoTimestamp();

        // Free counters array
        for (self.counters) |row| {
            self.allocator.free(row);
        }
        self.allocator.free(self.counters);
        self.allocator.free(self.hash_seeds);
        self.allocator.destroy(self);

        const elapsed = time.nanoTimestamp() - start_time;
        std.debug.print("[TIMING] CountMinSketch.deinit: {d:.2}ms\n", .{@as(f64, @floatFromInt(elapsed)) / time.ns_per_ms});
    }

    /// Add a string with conservative updating
    pub fn conservativeAdd(self: *CountMinSketch, string: []const u8, count: usize) !void {
        const hash_indices = try self.computeHashIndices(string);
        defer self.allocator.free(hash_indices);

        // Find the minimum counter value among all hash positions
        var min_value: usize = std.math.maxInt(usize);
        for (0..self.depth) |i| {
            min_value = @min(min_value, self.counters[i][hash_indices[i]]);
        }

        // Conservative update: only increment counters that are equal to the minimum
        for (0..self.depth) |i| {
            if (self.counters[i][hash_indices[i]] == min_value) {
                self.counters[i][hash_indices[i]] += count;
            }
        }
    }

    /// Query the approximate frequency of a string
    pub fn query(self: *CountMinSketch, string: []const u8) !usize {
        const hash_indices = try self.computeHashIndices(string);
        defer self.allocator.free(hash_indices);

        // Return the minimum counter value as the frequency estimate
        var min_value: usize = std.math.maxInt(usize);
        for (0..self.depth) |i| {
            min_value = @min(min_value, self.counters[i][hash_indices[i]]);
        }

        return min_value;
    }

    /// Compute hash indices for a string using xxhash with different seeds
    fn computeHashIndices(self: *CountMinSketch, string: []const u8) ![]usize {
        const indices = try self.allocator.alloc(usize, self.depth);
        errdefer self.allocator.free(indices);

        // Use xxHash with different seeds as specified
        for (0..self.depth) |i| {
            const hash = std.hash.XxHash64.hash(self.hash_seeds[i], string);

            // Truncate hash to fit the width using mask
            indices[i] = @as(usize, @intCast(hash)) & self.width_mask;
        }

        return indices;
    }

    /// Reset all counters to zero
    pub fn reset(self: *CountMinSketch) void {
        for (self.counters) |row| {
            @memset(row, 0);
        }
    }

    /// Merge another Count-Min Sketch into this one
    pub fn merge(self: *CountMinSketch, other: *CountMinSketch) !void {
        if (self.width != other.width or self.depth != other.depth) {
            return error.IncompatibleSketches;
        }

        for (0..self.depth) |i| {
            for (0..self.width) |j| {
                self.counters[i][j] += other.counters[i][j];
            }
        }
    }
};
