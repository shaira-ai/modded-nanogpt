const std = @import("std");

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const allocator = gpa.allocator();

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

export fn free_uc_strings(strings: [*]const [*:0]u8, count: usize) void {
    for (0..count) |i| {
        const str = std.mem.span(strings[i]);
        allocator.free(str);
    }

    const array = strings[0..count];
    allocator.free(array);
}
