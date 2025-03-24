const std = @import("std");
const BakaCorasick = @import("baka_corasick.zig").BakaCorasick;
const ArrayList = std.ArrayList;
const Queue = @import("queue.zig").Queue;

const allocator = std.heap.c_allocator;

// [*:0]const u8 : a sequence of constant bytes that ends with a zero byte
// [*]const [*:0]const u8 : a sequence of constant strings
// result_count: pointer to a location to write the number of strings processed
// the return type is a pointer to an array of strings
export fn convertToUC(input_arr: [*]const [*:0]const u8, input_count: usize, result_count: *usize) [*]const [*:0]u8 {

    // allocate memory for result
    var result = allocator.alloc([*:0]u8, input_count) catch {
        result_count.* = 0;
        return undefined;
    };

    var count: usize = 0;
    for (0..input_count) |i| {
        const input = std.mem.span(input_arr[i]);
        //allocate memory for uppercase version with null terminator
        var upper = allocator.allocSentinel(u8, input.len, 0) catch continue;

        //convert to uppercase
        for (input, 0..) |c, j| {
            upper[j] = std.ascii.toUpper(c);
        }

        result[i] = upper.ptr;
        count += 1;
    }

    result_count.* = count;
    return result.ptr;
}

export fn makeBC(input_arr: [*]const [*]const u8, lengths: [*]const usize, token_ids: [*]const u32, input_count: usize) usize {
    var bc = allocator.create(BakaCorasick) catch @panic("Failed to allocate memory for BakaCorasick");
    bc.* = BakaCorasick.init(allocator) catch @panic("Failed to initialize BakaCorasick");
    const print_ids = [_]u32{309, 289, 339, 365, 255, 10662};
    for (0..input_count) |i| {
        for (print_ids) |id| {
            if (token_ids[i] == id) {
                //std.log.err("token {d} is \"{s}\" with length {d}", .{token_ids[i], input_arr[i][0..lengths[i]], lengths[i]});
            }
        }
        const input = input_arr[i][0..lengths[i]];
        bc.insert(input, token_ids[i]) catch @panic("Failed to insert token");
    }
    for (print_ids) |id| {
        var str: []const u8 = undefined;
        for (0..input_count) |i| {
            if (token_ids[i] == id) {
                str = input_arr[i][0..lengths[i]];
                var state: u32 = 0;
                var idx: u32 = 0;
                //std.log.err("Hello, idx={} and state={} and depth={} and token_id={}", .{idx, state, bc.info[state].depth, bc.info[state].token_id});
                while (idx < str.len) {
                    state = bc.transitions[state][str[idx]];
                    idx += 1;
                    //std.log.err("Hello, idx={} and state={} and depth={} and token_id={}", .{idx, state, bc.info[state].depth, bc.info[state].token_id});
                }
            }
        }
    }
    bc.computeSuffixLinks() catch @panic("Failed to compute suffix links");
    //std.log.err("I will return {d}", .{@intFromPtr(bc)});
    return @intFromPtr(bc);
}

export fn getAllMatches(
    bc_: usize,
    input: [*]const u8,
    input_len: usize,
    result_count: *usize,
) [*]const u32 {
    return getAllMatchesInternal(false, bc_, input, input_len, result_count);
}

export fn getAllMatchesIncludingPastEnd(
    bc_: usize,
    input: [*]const u8,
    input_len: usize,
    result_count: *usize,
) [*]const u32 {
    return getAllMatchesInternal(true, bc_, input, input_len, result_count);
}

fn getAllMatchesInternal(
    comptime GET_MATCHES_PAST_END: bool,
    bc_: usize,
    input: [*]const u8,
    input_len: usize,
    result_count: *usize,
) [*]const u32 {
    //std.log.err("I got a pointer to BakaCorasick at {d}", .{bc_});
    const bc: *BakaCorasick = @ptrFromInt(bc_);
    var state: u32 = 0;
    var ret: ArrayList(u32) = ArrayList(u32).init(allocator);
    defer ret.deinit();
    var idx: u32 = 0;
    while (idx < input_len) {
        //std.log.err("Hello, idx={} and state={}", .{idx, state});
        state = bc.transitions[state][input[idx]];
        idx += 1;
        var green_state = state;
        if (bc.info[green_state].token_id == BakaCorasick.NO_TOKEN) {
            green_state = bc.info[green_state].green;
        }
        while(green_state != 0) {
            const token_id = bc.info[green_state].token_id;
            const depth = bc.info[green_state].depth;
            const start_idx = idx - depth;
            ret.ensureUnusedCapacity(3) catch @panic("Failed to ensure unused capacity");
            ret.appendAssumeCapacity(start_idx);
            ret.appendAssumeCapacity(idx);
            ret.appendAssumeCapacity(token_id);
            green_state = bc.info[green_state].green;
        }
    }
    if (GET_MATCHES_PAST_END) {
        const BFSState = struct {
            bc_state: u32,
            idx: u32,
        };
        var queue = Queue(BFSState){};
        for (0..256) |c| {
            queue.push(allocator, .{ .bc_state = bc.transitions[state][c], .idx = idx + 1 }) catch @panic("Failed to push to queue");
        }
        while (queue.pop()) |st| {
            state = st.bc_state;
            idx = st.idx;
            var depth = bc.info[state].depth;
            var start_idx = idx - depth;
            if (start_idx >= input_len) {
                continue;
            }
            var green_state = state;
            if (bc.info[green_state].token_id == BakaCorasick.NO_TOKEN) {
                green_state = bc.info[green_state].green;
            }
            while (green_state != 0) {
                depth = bc.info[green_state].depth;
                start_idx = idx - depth;
                if (start_idx >= input_len) {
                    break;
                }
                const token_id = bc.info[green_state].token_id;
                ret.ensureUnusedCapacity(3) catch @panic("Failed to ensure unused capacity");
                ret.appendAssumeCapacity(start_idx);
                ret.appendAssumeCapacity(idx);
                ret.appendAssumeCapacity(token_id);
                green_state = bc.info[green_state].green;
            }
            for (0..256) |c| {
                queue.push(allocator, .{ .bc_state = bc.transitions[state][c], .idx = idx + 1 }) catch @panic("Failed to push to queue");
            }
        }
    }
    const ret_slice = allocator.alloc(u32, ret.items.len) catch @panic("Failed to allocate memory for return slice");
    for (ret.items, 0..) |item, i| {
        ret_slice[i] = item;
    }
    result_count.* = ret_slice.len;
    return ret_slice.ptr;
}

export fn freeSlice(slice: [*]const u32, count: usize) void {
    allocator.free(slice[0..count]);
}

export fn getBCMemoryUsage(bc_: usize) usize {
    const bc: *BakaCorasick = @ptrFromInt(bc_);
    return bc.capacity * (@sizeOf(@TypeOf(bc.transitions[0])) + @sizeOf(@TypeOf(bc.info[0])));
}

export fn free_uc_strings(strings: [*]const [*:0]u8, count: usize) void {
    for (0..count) |i| {
        const str = std.mem.span(strings[i]);
        allocator.free(str);
    }

    const array = strings[0..count];
    allocator.free(array);
}
