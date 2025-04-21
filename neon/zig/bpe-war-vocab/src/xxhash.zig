const std = @import("std");
const builtin = @import("builtin");
const native_endian = builtin.cpu.arch.endian();
const rotl = std.math.rotl;



pub const XxHash64 = struct {
    const prime_1: u64 = 0x9E3779B185EBCA87; // 0b1001111000110111011110011011000110000101111010111100101010000111
    const prime_2: u64 = 0xC2B2AE3D27D4EB4F; // 0b1100001010110010101011100011110100100111110101001110101101001111
    const prime_3: u64 = 0x165667B19E3779F9; // 0b0001011001010110011001111011000110011110001101110111100111111001
    const prime_4: u64 = 0x85EBCA77C2B2AE63; // 0b1000010111101011110010100111011111000010101100101010111001100011
    const prime_5: u64 = 0x27D4EB2F165667C5; // 0b0010011111010100111010110010111100010110010101100110011111000101
};


pub const XxHash32 = struct {
    const prime_1: u32 = 0x9E3779B1; // 0b10011110001101110111100110110001
    const prime_2: u32 = 0x85EBCA77; // 0b10000101111010111100101001110111
    const prime_3: u32 = 0xC2B2AE3D; // 0b11000010101100101010111000111101
    const prime_4: u32 = 0x27D4EB2F; // 0b00100111110101001110101100101111
    const prime_5: u32 = 0x165667B1; // 0b00010110010101100110011110110001
};

pub fn XxHash3(
    comptime min_length: usize,
    comptime max_length: usize,
    comptime VEC_WIDTH: comptime_int
) type {
    if (min_length < 4) {
        @compileError("min_length must be >= 4");
    }
    if (max_length > 256) {
        @compileError("max_length must be <= 256");
    }
    if (max_length < min_length) {
        @compileError("max_length must be >= min_length");
    }
    return struct {
        const V = @Vector(VEC_WIDTH, u64);
        const V32 = @Vector(VEC_WIDTH, u32);
        const Block = @Vector(8, u64);
        const default_secret: [192]u8 = .{
            0xb8, 0xfe, 0x6c, 0x39, 0x23, 0xa4, 0x4b, 0xbe, 0x7c, 0x01, 0x81, 0x2c, 0xf7, 0x21, 0xad, 0x1c,
            0xde, 0xd4, 0x6d, 0xe9, 0x83, 0x90, 0x97, 0xdb, 0x72, 0x40, 0xa4, 0xa4, 0xb7, 0xb3, 0x67, 0x1f,
            0xcb, 0x79, 0xe6, 0x4e, 0xcc, 0xc0, 0xe5, 0x78, 0x82, 0x5a, 0xd0, 0x7d, 0xcc, 0xff, 0x72, 0x21,
            0xb8, 0x08, 0x46, 0x74, 0xf7, 0x43, 0x24, 0x8e, 0xe0, 0x35, 0x90, 0xe6, 0x81, 0x3a, 0x26, 0x4c,
            0x3c, 0x28, 0x52, 0xbb, 0x91, 0xc3, 0x00, 0xcb, 0x88, 0xd0, 0x65, 0x8b, 0x1b, 0x53, 0x2e, 0xa3,
            0x71, 0x64, 0x48, 0x97, 0xa2, 0x0d, 0xf9, 0x4e, 0x38, 0x19, 0xef, 0x46, 0xa9, 0xde, 0xac, 0xd8,
            0xa8, 0xfa, 0x76, 0x3f, 0xe3, 0x9c, 0x34, 0x3f, 0xf9, 0xdc, 0xbb, 0xc7, 0xc7, 0x0b, 0x4f, 0x1d,
            0x8a, 0x51, 0xe0, 0x4b, 0xcd, 0xb4, 0x59, 0x31, 0xc8, 0x9f, 0x7e, 0xc9, 0xd9, 0x78, 0x73, 0x64,
            0xea, 0xc5, 0xac, 0x83, 0x34, 0xd3, 0xeb, 0xc3, 0xc5, 0x81, 0xa0, 0xff, 0xfa, 0x13, 0x63, 0xeb,
            0x17, 0x0d, 0xdd, 0x51, 0xb7, 0xf0, 0xda, 0x49, 0xd3, 0x16, 0x55, 0x26, 0x29, 0xd4, 0x68, 0x9e,
            0x2b, 0x16, 0xbe, 0x58, 0x7d, 0x47, 0xa1, 0xfc, 0x8f, 0xf8, 0xb8, 0xd1, 0x7a, 0xd0, 0x31, 0xce,
            0x45, 0xcb, 0x3a, 0x8f, 0x95, 0x16, 0x04, 0x28, 0xaf, 0xd7, 0xfb, 0xca, 0xbb, 0x4b, 0x40, 0x7e,
        };

        const prime_mx1 = 0x165667919E3779F9;
        const prime_mx2 = 0x9FB21C651E98DF25;

        const Accumulator = extern struct {
            consumed: usize = 0,
            seed: u64,
            secret: [192]u8 = undefined,
            state: Block = Block{
                XxHash32.prime_3,
                XxHash64.prime_1,
                XxHash64.prime_2,
                XxHash64.prime_3,
                XxHash64.prime_4,
                XxHash32.prime_2,
                XxHash64.prime_5,
                XxHash32.prime_1,
            },

            inline fn init(seed: u64) Accumulator {
                var self = Accumulator{ .seed = seed };
                for (
                    std.mem.bytesAsSlice(Block, &self.secret),
                    std.mem.bytesAsSlice(Block, &default_secret),
                ) |*dst, src| {
                    dst.* = swap(swap(src) +% Block{
                        seed, @as(u64, 0) -% seed,
                        seed, @as(u64, 0) -% seed,
                        seed, @as(u64, 0) -% seed,
                        seed, @as(u64, 0) -% seed,
                    });
                }
                return self;
            }

            inline fn round(
                noalias state: *Block,
                noalias input_block: *align(1) const Block,
                noalias secret_block: *align(1) const Block,
            ) void {
                const data = swap(input_block.*);
                const mixed = data ^ swap(secret_block.*);
                state.* +%= (mixed & @as(Block, @splat(0xffffffff))) *% (mixed >> @splat(32));
                state.* +%= @shuffle(u64, data, undefined, [_]i32{ 1, 0, 3, 2, 5, 4, 7, 6 });
            }

            inline fn accumulate(noalias self: *Accumulator, blocks: []align(1) const Block) void {
                const secret = std.mem.bytesAsSlice(u64, self.secret[self.consumed * 8 ..]);
                for (blocks, secret[0..blocks.len]) |*input_block, *secret_block| {
                    @prefetch(@as([*]const u8, @ptrCast(input_block)) + 320, .{});
                    round(&self.state, input_block, @ptrCast(secret_block));
                }
            }

            inline fn scramble(self: *Accumulator) void {
                const secret_block: Block = @bitCast(self.secret[192 - @sizeOf(Block) .. 192].*);
                self.state ^= self.state >> @splat(47);
                self.state ^= swap(secret_block);
                self.state *%= @as(Block, @splat(XxHash32.prime_1));
            }

            inline fn consume(noalias self: *Accumulator, input_blocks: []align(1) const Block) void {
                const blocks_per_scramble = 1024 / @sizeOf(Block);
                std.debug.assert(self.consumed <= blocks_per_scramble);

                var blocks = input_blocks;
                var blocks_until_scramble = blocks_per_scramble - self.consumed;
                while (blocks.len >= blocks_until_scramble) {
                    self.accumulate(blocks[0..blocks_until_scramble]);
                    self.scramble();

                    self.consumed = 0;
                    blocks = blocks[blocks_until_scramble..];
                    blocks_until_scramble = blocks_per_scramble;
                }

                self.accumulate(blocks);
                self.consumed += blocks.len;
            }

            inline fn digest(noalias self: *Accumulator, total_len: u64, noalias last_block: *align(1) const Block) u64 {
                const secret_block = self.secret[192 - @sizeOf(Block) - 7 ..][0..@sizeOf(Block)];
                round(&self.state, last_block, @ptrCast(secret_block));

                const merge_block: Block = @bitCast(self.secret[11 .. 11 + @sizeOf(Block)].*);
                self.state ^= swap(merge_block);

                var result = XxHash64.prime_1 *% total_len;
                inline for (0..4) |i| {
                    result +%= fold(self.state[i * 2], self.state[i * 2 + 1]);
                }
                return avalanche(.h3, result);
            }
        };

        inline fn avalanche(comptime mode: union(enum) { h3, h64, rrmxmx: u64 }, x0: u64) u64 {
            switch (mode) {
                .h3 => {
                    const x1 = (x0 ^ (x0 >> 37)) *% prime_mx1;
                    return x1 ^ (x1 >> 32);
                },
                .h64 => {
                    const x1 = (x0 ^ (x0 >> 33)) *% XxHash64.prime_2;
                    const x2 = (x1 ^ (x1 >> 29)) *% XxHash64.prime_3;
                    return x2 ^ (x2 >> 32);
                },
                .rrmxmx => |len| {
                    const x1 = (x0 ^ rotl(u64, x0, 49) ^ rotl(u64, x0, 24)) *% prime_mx2;
                    const x2 = (x1 ^ ((x1 >> 35) +% len)) *% prime_mx2;
                    return x2 ^ (x2 >> 28);
                },
            }
        }

        inline fn fold(a: u64, b: u64) u64 {
            const wide: [2]u64 = @bitCast(@as(u128, a) *% b);
            return wide[0] ^ wide[1];
        }

        inline fn swap(x: anytype) @TypeOf(x) {
            return if (native_endian == .big) @byteSwap(x) else x;
        }

        inline fn disableAutoVectorization(x: anytype) void {
            if (!@inComptime()) asm volatile (""
                :
                : [x] "r" (x),
            );
        }

        inline fn mix16(seed: u64, input: []const u8, secret: []const u8) u64 {
            const blk: [4]u64 = @bitCast([_][16]u8{ input[0..16].*, secret[0..16].* });
            disableAutoVectorization(seed);

            return fold(
                swap(blk[0]) ^ (swap(blk[2]) +% seed),
                swap(blk[1]) ^ (swap(blk[3]) -% seed),
            );
        }

        // Public API - Oneshot

        pub noinline fn hash(noalias dst_: [*]u64, seed: [VEC_WIDTH]u64, noalias input: *const [256]u8) void {
            if (min_length == max_length) {
                return hashSingle(dst_, seed, input);
            }
            @setEvalBranchQuota(1_000_000);
            const secret = &default_secret;
            var dst = dst_;
            if (min_length < 9) {
                // hash8 for 4<=len<9
                const flip: [2]u64 = @bitCast(secret[8..24].*);
                var blk: [5]u32 = undefined;
                inline for (0..5) |i| {
                    blk[i] = @bitCast(input[i..i+4].*);
                }
                var mixed: [VEC_WIDTH]u64 = undefined;
                inline for (0..VEC_WIDTH) |j| {
                    mixed[j] = seed[j] ^ (@as(u64, @byteSwap(@as(u32, @truncate(seed[j])))) << 32);
                }
                var key: [VEC_WIDTH]u64 = undefined;
                inline for (0..VEC_WIDTH) |j| {
                    key[j] = (swap(flip[0]) ^ swap(flip[1])) -% mixed[j];
                }
                var combined: [5]u64 = undefined;
                inline for (0..5) |i| {
                    combined[i] = (@as(u64, swap(blk[0])) << 32) +% swap(blk[i]);
                }
                var ret: [5*VEC_WIDTH]u64 = undefined;
                inline for (0..5) |i| {
                    inline for (0..VEC_WIDTH) |j| {
                        ret[i*VEC_WIDTH+j] = avalanche(.{ .rrmxmx = i+4 }, key[j] ^ combined[i]);
                    }
                }
                inline for (0..5) |i| {
                    const this_len = i + 4;
                    if (min_length <= this_len and this_len <= max_length) {
                        inline for (0..VEC_WIDTH) |j| {
                            dst[j] = ret[i*VEC_WIDTH+j];
                        }
                        dst += VEC_WIDTH;
                    }
                }
            }
            if (min_length < 17 and max_length > 8) {
                // hash16 for 9<=len<17
                const flip: [4]u64 = @bitCast(secret[24..56].*);
                var blk: [9]u64 = undefined;
                inline for (0..9) |i| {
                    blk[i] = @bitCast(input[i..i+8].*);
                }
                const sf0_xor_sf1 = (swap(flip[0]) ^ swap(flip[1]));
                var lo_2nd_part: [VEC_WIDTH]u64 = undefined;
                inline for (0..VEC_WIDTH) |j| {
                    lo_2nd_part[j] = sf0_xor_sf1 +% seed[j];
                }
                var lo: [VEC_WIDTH]u64 = undefined;
                inline for (0..VEC_WIDTH) |j| {
                    lo[j] = swap(blk[0]) ^ lo_2nd_part[j];
                }
                var bslo: [VEC_WIDTH]u64 = undefined;
                inline for (0..VEC_WIDTH) |j| {
                    bslo[j] = @byteSwap(lo[j]);
                }
                const sf2_xor_sf3 = (swap(flip[2]) ^ swap(flip[3]));
                var hi_2nd_part: [VEC_WIDTH]u64 = undefined;
                inline for (0..VEC_WIDTH) |j| {
                    hi_2nd_part[j] = sf2_xor_sf3 -% seed[j];
                }
                var hi: [8][VEC_WIDTH]u64 = undefined;
                inline for (0..8) |i| {
                    inline for (0..VEC_WIDTH) |j| {
                        hi[i][j] = swap(blk[i+1]) ^ hi_2nd_part[j];
                    }
                }
                var combined: [8][VEC_WIDTH]u64 = undefined;
                inline for (0..8) |i| {
                    inline for (0..VEC_WIDTH) |j| {
                        combined[i][j] = @as(u64, i+9) +% bslo[j] +% hi[i][j] +% fold(lo[j], hi[i][j]);
                    }
                }
                var ret: [8*VEC_WIDTH]u64 = undefined;
                inline for (0..8) |i| {
                    inline for (0..VEC_WIDTH) |j| {
                        ret[i*VEC_WIDTH+j] = avalanche(.h3, combined[i][j]);
                    }
                }
                inline for (0..8) |i| {
                    const this_len = i + 9;
                    if (min_length <= this_len and this_len <= max_length) {
                        inline for (0..VEC_WIDTH) |j| {
                            dst[j] = ret[i*VEC_WIDTH+j];
                        }
                        dst += VEC_WIDTH;
                    }
                }
            }
            if (min_length < 129 and max_length > 16) {
                // hash128 for 17<=len<129
                var acc: [112][VEC_WIDTH]u64 = undefined;
                var prime1_acc: u64 = XxHash64.prime_1 *% 17;
                for (0..112) |i| {
                    inline for (0..VEC_WIDTH) |j| {
                        acc[i][j] = prime1_acc;
                    }
                    prime1_acc +%= XxHash64.prime_1;
                }
                inline for (0..4) |ii| {
                    const in_offset = 48 - (ii * 16);
                    const scrt_offset = 96 - (ii * 32);
                    var first_mix: [VEC_WIDTH]u64 = undefined;
                    inline for (0..VEC_WIDTH) |j| {
                        first_mix[j] = mix16(seed[j], input[in_offset..], secret[scrt_offset..]);
                    }
                    for (scrt_offset-|16..112) |i| {
                        inline for (0..VEC_WIDTH) |j| {
                            acc[i][j] +%= first_mix[j];
                            acc[i][j] +%= mix16(seed[j], input[(i+17) - (in_offset + 16) ..], secret[scrt_offset + 16 ..]);
                        }
                    }
                }
                for (0..112) |i| {
                    const this_len = i + 17;
                    if (min_length <= this_len and this_len <= max_length) {
                        inline for (0..VEC_WIDTH) |j| {
                            dst[j] = avalanche(.h3, acc[i][j]);
                        }
                        dst += VEC_WIDTH;
                    }
                }
            }
            if (min_length < 241 and max_length > 128) {
                // hash240 for 129<=len<241
                var acc: [VEC_WIDTH]u64 = .{0}**VEC_WIDTH;
                inline for (0..8) |ii| {
                    inline for (0..VEC_WIDTH) |j| {
                        acc[j] +%= mix16(seed[j], input[ii*16..], secret[ii*16..]);
                    }
                }
                var acc_end_acc: [VEC_WIDTH]u64 = .{0}**VEC_WIDTH;
                var prime_acc: u64 = XxHash64.prime_1 *% 129;
                for (0..15) |i| {
                    const this_len = i + 129;
                    if (min_length <= this_len and this_len <= max_length) {
                        var acc_endi: [VEC_WIDTH]u64 = undefined;
                        var acci: [VEC_WIDTH]u64 = undefined;
                        inline for (0..VEC_WIDTH) |j| {
                            acci[j] = prime_acc +% acc[j];
                        }
                        inline for (0..VEC_WIDTH) |j| {
                            acc_endi[j] = mix16(seed[j], input[(129+i) - 16 ..], secret[136 - 17 ..]);
                        }
                        inline for (0..VEC_WIDTH) |j| {
                            acci[j] = avalanche(.h3, acci[j]) +% acc_endi[j];
                        }
                        inline for (0..VEC_WIDTH) |j| {
                            dst[j] = avalanche(.h3, acci[j]);
                        }
                        dst += VEC_WIDTH;
                    }
                    prime_acc +%= XxHash64.prime_1;
                }
                for (8..14) |ii| {
                    inline for (0..VEC_WIDTH) |j| {
                        acc_end_acc[j] +%= mix16(seed[j], input[(ii*16)..], secret[((ii-8)*16)+3..]);
                    }
                    for (0..16) |i| {
                        const this_len = 144 + (ii-8) * 16 + i;
                        if (min_length <= this_len and this_len <= max_length) {
                            var acc_endi: [VEC_WIDTH]u64 = undefined;
                            var acci: [VEC_WIDTH]u64 = undefined;
                            inline for (0..VEC_WIDTH) |j| {
                                acci[j] = prime_acc +% acc[j];
                            }
                            inline for (0..VEC_WIDTH) |j| {
                                acc_endi[j] = mix16(seed[j], input[(16 * ii + i) ..], secret[136 - 17 ..]);
                            }
                            inline for (0..VEC_WIDTH) |j| {
                                acci[j] = avalanche(.h3, acci[j]) +% acc_endi[j] +% acc_end_acc[j];
                            }
                            inline for (0..VEC_WIDTH) |j| {
                                dst[j] = avalanche(.h3, acci[j]);
                            }
                            dst += VEC_WIDTH;
                        }
                        prime_acc +%= XxHash64.prime_1;
                    }
                }
                // length exactly 240
                if (min_length <= 240 and max_length >= 240) {
                    inline for (0..VEC_WIDTH) |j| {
                        acc_end_acc[j] +%= mix16(seed[j], input[224 ..], secret[((14-8)*16)+3..]);
                    }
                    var acc_endi: [VEC_WIDTH]u64 = undefined;
                    var acci: [VEC_WIDTH]u64 = undefined;
                    inline for (0..VEC_WIDTH) |j| {
                        acci[j] = prime_acc +% acc[j];
                    }
                    inline for (0..VEC_WIDTH) |j| {
                        acc_endi[j] = mix16(seed[j], input[224 ..], secret[136 - 17 ..]);
                    }
                    inline for (0..VEC_WIDTH) |j| {
                        acci[j] = avalanche(.h3, acci[j]) +% acc_endi[j] +% acc_end_acc[j];
                    }
                    inline for (0..VEC_WIDTH) |j| {
                        dst[j] = avalanche(.h3, acci[j]);
                    }
                    dst += VEC_WIDTH;
                }
            }
            if (max_length > 240) {
                // hashLong for 241<=len<=256
                var base_accs: [VEC_WIDTH]Accumulator = undefined;
                inline for (0..VEC_WIDTH) |j| {
                    base_accs[j] = Accumulator.init(seed[j]);
                }
                inline for (0..VEC_WIDTH) |j| {
                    base_accs[j].consume(std.mem.bytesAsSlice(Block, input[0..192]));
                }
                var base_states: [VEC_WIDTH]Block = undefined;
                inline for (0..VEC_WIDTH) |j| {
                    base_states[j] = base_accs[j].state;
                }
                for (241..257) |i| {
                    if (min_length <= i and i <= max_length) {
                        inline for (0..VEC_WIDTH) |j| {
                            dst[j] = base_accs[j].digest(i, @ptrCast(&input[i-64]));
                        }
                        dst += VEC_WIDTH;
                        inline for (0..VEC_WIDTH) |j| {
                            base_accs[j].state = base_states[j];
                        }
                    }
                }
            }
        }

        pub inline fn hashSingle(noalias dst_: [*]u64, seed: [VEC_WIDTH]u64, noalias input: *const [256]u8) void {
            @setEvalBranchQuota(1_000_000);
            const secret = &default_secret;
            var dst = dst_;
            if (min_length < 9) {
                // hash8 for 4<=len<9
                const flip: [2]u64 = @bitCast(secret[8..24].*);
                var blk: [5]u32 = undefined;
                inline for (0..5) |i| {
                    blk[i] = @bitCast(input[i..i+4].*);
                }
                var mixed: [VEC_WIDTH]u64 = undefined;
                inline for (0..VEC_WIDTH) |j| {
                    mixed[j] = seed[j] ^ (@as(u64, @byteSwap(@as(u32, @truncate(seed[j])))) << 32);
                }
                var key: [VEC_WIDTH]u64 = undefined;
                inline for (0..VEC_WIDTH) |j| {
                    key[j] = (swap(flip[0]) ^ swap(flip[1])) -% mixed[j];
                }
                var combined: [5]u64 = undefined;
                inline for (0..5) |i| {
                    combined[i] = (@as(u64, swap(blk[0])) << 32) +% swap(blk[i]);
                }
                var ret: [5*VEC_WIDTH]u64 = undefined;
                inline for (0..5) |i| {
                    inline for (0..VEC_WIDTH) |j| {
                        ret[i*VEC_WIDTH+j] = avalanche(.{ .rrmxmx = i+4 }, key[j] ^ combined[i]);
                    }
                }
                inline for (0..5) |i| {
                    const this_len = i + 4;
                    if (min_length <= this_len and this_len <= max_length) {
                        inline for (0..VEC_WIDTH) |j| {
                            dst[j] = ret[i*VEC_WIDTH+j];
                        }
                        dst += VEC_WIDTH;
                    }
                }
            }
            if (min_length < 17 and max_length > 8) {
                // hash16 for 9<=len<17
                const flip: [4]u64 = @bitCast(secret[24..56].*);
                var blk: [9]u64 = undefined;
                inline for (0..9) |i| {
                    blk[i] = @bitCast(input[i..i+8].*);
                }
                const sf0_xor_sf1 = (swap(flip[0]) ^ swap(flip[1]));
                var lo_2nd_part: [VEC_WIDTH]u64 = undefined;
                inline for (0..VEC_WIDTH) |j| {
                    lo_2nd_part[j] = sf0_xor_sf1 +% seed[j];
                }
                var lo: [VEC_WIDTH]u64 = undefined;
                inline for (0..VEC_WIDTH) |j| {
                    lo[j] = swap(blk[0]) ^ lo_2nd_part[j];
                }
                var bslo: [VEC_WIDTH]u64 = undefined;
                inline for (0..VEC_WIDTH) |j| {
                    bslo[j] = @byteSwap(lo[j]);
                }
                const sf2_xor_sf3 = (swap(flip[2]) ^ swap(flip[3]));
                var hi_2nd_part: [VEC_WIDTH]u64 = undefined;
                inline for (0..VEC_WIDTH) |j| {
                    hi_2nd_part[j] = sf2_xor_sf3 -% seed[j];
                }
                var hi: [8][VEC_WIDTH]u64 = undefined;
                inline for (0..8) |i| {
                    inline for (0..VEC_WIDTH) |j| {
                        hi[i][j] = swap(blk[i+1]) ^ hi_2nd_part[j];
                    }
                }
                var combined: [8][VEC_WIDTH]u64 = undefined;
                inline for (0..8) |i| {
                    inline for (0..VEC_WIDTH) |j| {
                        combined[i][j] = @as(u64, i+9) +% bslo[j] +% hi[i][j] +% fold(lo[j], hi[i][j]);
                    }
                }
                var ret: [8*VEC_WIDTH]u64 = undefined;
                inline for (0..8) |i| {
                    inline for (0..VEC_WIDTH) |j| {
                        ret[i*VEC_WIDTH+j] = avalanche(.h3, combined[i][j]);
                    }
                }
                inline for (0..8) |i| {
                    const this_len = i + 9;
                    if (min_length <= this_len and this_len <= max_length) {
                        inline for (0..VEC_WIDTH) |j| {
                            dst[j] = ret[i*VEC_WIDTH+j];
                        }
                        dst += VEC_WIDTH;
                    }
                }
            }
            if (min_length < 129 and max_length > 16) {
                // hash128 for 17<=len<129
                var acc: [112][VEC_WIDTH]u64 = undefined;
                var prime1_acc: u64 = XxHash64.prime_1 *% 17;
                inline for (0..112) |i| {
                    inline for (0..VEC_WIDTH) |j| {
                        acc[i][j] = prime1_acc;
                    }
                    prime1_acc +%= XxHash64.prime_1;
                }
                inline for (0..4) |ii| {
                    const in_offset = 48 - (ii * 16);
                    const scrt_offset = 96 - (ii * 32);
                    var first_mix: [VEC_WIDTH]u64 = undefined;
                    inline for (0..VEC_WIDTH) |j| {
                        first_mix[j] = mix16(seed[j], input[in_offset..], secret[scrt_offset..]);
                    }
                    inline for (scrt_offset-|16..112) |i| {
                        inline for (0..VEC_WIDTH) |j| {
                            acc[i][j] +%= first_mix[j];
                            acc[i][j] +%= mix16(seed[j], input[(i+17) - (in_offset + 16) ..], secret[scrt_offset + 16 ..]);
                        }
                    }
                }
                inline for (0..112) |i| {
                    const this_len = i + 17;
                    if (min_length <= this_len and this_len <= max_length) {
                        inline for (0..VEC_WIDTH) |j| {
                            dst[j] = avalanche(.h3, acc[i][j]);
                        }
                        dst += VEC_WIDTH;
                    }
                }
            }
            if (min_length < 241 and max_length > 128) {
                // hash240 for 129<=len<241
                var acc: [VEC_WIDTH]u64 = .{0}**VEC_WIDTH;
                inline for (0..8) |ii| {
                    inline for (0..VEC_WIDTH) |j| {
                        acc[j] +%= mix16(seed[j], input[ii*16..], secret[ii*16..]);
                    }
                }
                var acc_end_acc: [VEC_WIDTH]u64 = .{0}**VEC_WIDTH;
                var prime_acc: u64 = XxHash64.prime_1 *% 129;
                inline for (0..15) |i| {
                    const this_len = i + 129;
                    if (min_length <= this_len and this_len <= max_length) {
                        var acc_endi: [VEC_WIDTH]u64 = undefined;
                        var acci: [VEC_WIDTH]u64 = undefined;
                        inline for (0..VEC_WIDTH) |j| {
                            acci[j] = prime_acc +% acc[j];
                        }
                        inline for (0..VEC_WIDTH) |j| {
                            acc_endi[j] = mix16(seed[j], input[(129+i) - 16 ..], secret[136 - 17 ..]);
                        }
                        inline for (0..VEC_WIDTH) |j| {
                            acci[j] = avalanche(.h3, acci[j]) +% acc_endi[j];
                        }
                        inline for (0..VEC_WIDTH) |j| {
                            dst[j] = avalanche(.h3, acci[j]);
                        }
                        dst += VEC_WIDTH;
                    }
                    prime_acc +%= XxHash64.prime_1;
                }
                inline for (8..14) |ii| {
                    inline for (0..VEC_WIDTH) |j| {
                        acc_end_acc[j] +%= mix16(seed[j], input[(ii*16)..], secret[((ii-8)*16)+3..]);
                    }
                    inline for (0..16) |i| {
                        const this_len = 144 + (ii-8) * 16 + i;
                        if (min_length <= this_len and this_len <= max_length) {
                            var acc_endi: [VEC_WIDTH]u64 = undefined;
                            var acci: [VEC_WIDTH]u64 = undefined;
                            inline for (0..VEC_WIDTH) |j| {
                                acci[j] = prime_acc +% acc[j];
                            }
                            inline for (0..VEC_WIDTH) |j| {
                                acc_endi[j] = mix16(seed[j], input[(16 * ii + i) ..], secret[136 - 17 ..]);
                            }
                            inline for (0..VEC_WIDTH) |j| {
                                acci[j] = avalanche(.h3, acci[j]) +% acc_endi[j] +% acc_end_acc[j];
                            }
                            inline for (0..VEC_WIDTH) |j| {
                                dst[j] = avalanche(.h3, acci[j]);
                            }
                            dst += VEC_WIDTH;
                        }
                        prime_acc +%= XxHash64.prime_1;
                    }
                }
                // length exactly 240
                if (min_length <= 240 and max_length >= 240) {
                    inline for (0..VEC_WIDTH) |j| {
                        acc_end_acc[j] +%= mix16(seed[j], input[224 ..], secret[((14-8)*16)+3..]);
                    }
                    var acc_endi: [VEC_WIDTH]u64 = undefined;
                    var acci: [VEC_WIDTH]u64 = undefined;
                    inline for (0..VEC_WIDTH) |j| {
                        acci[j] = prime_acc +% acc[j];
                    }
                    inline for (0..VEC_WIDTH) |j| {
                        acc_endi[j] = mix16(seed[j], input[224 ..], secret[136 - 17 ..]);
                    }
                    inline for (0..VEC_WIDTH) |j| {
                        acci[j] = avalanche(.h3, acci[j]) +% acc_endi[j] +% acc_end_acc[j];
                    }
                    inline for (0..VEC_WIDTH) |j| {
                        dst[j] = avalanche(.h3, acci[j]);
                    }
                    dst += VEC_WIDTH;
                }
            }
            if (max_length > 240) {
                // hashLong for 241<=len<=256
                var base_accs: [VEC_WIDTH]Accumulator = undefined;
                inline for (0..VEC_WIDTH) |j| {
                    base_accs[j] = Accumulator.init(seed[j]);
                }
                inline for (0..VEC_WIDTH) |j| {
                    base_accs[j].consume(std.mem.bytesAsSlice(Block, input[0..192]));
                }
                var base_states: [VEC_WIDTH]Block = undefined;
                inline for (0..VEC_WIDTH) |j| {
                    base_states[j] = base_accs[j].state;
                }
                inline for (241..257) |i| {
                    if (min_length <= i and i <= max_length) {
                        inline for (0..VEC_WIDTH) |j| {
                            dst[j] = base_accs[j].digest(i, @ptrCast(&input[i-64]));
                        }
                        dst += VEC_WIDTH;
                        inline for (0..VEC_WIDTH) |j| {
                            base_accs[j].state = base_states[j];
                        }
                    }
                }
            }
        }
    };
}