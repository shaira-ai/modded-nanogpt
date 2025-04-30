const std = @import("std");
const builtin = @import("builtin");
const mem = std.mem;
const maxInt = std.math.maxInt;
const assert = std.debug.assert;
const native_os = builtin.os.tag;
const linux = std.os.linux;
const posix = std.posix;

pub fn allocateHugePages(size: usize) ![]u8 {
    if (native_os != .linux) {
        return error.UnsupportedOS;
    }

    const requested_size = size;
    // First attempt: 1GB huge pages
    {
        const page_size: usize = 1024 * 1024 * 1024; // 1GB
        const aligned_len = mem.alignForward(usize, requested_size, page_size);
        const hint = null;

        // Define MAP flags for 1GB huge pages
        const HUGETLB_FLAG_ENCODE_SHIFT = 26;
        const huge_page_flavor = @as(u32, 30) << HUGETLB_FLAG_ENCODE_SHIFT; // 1GB = 2^30

        const base_flags = linux.MAP{
            .TYPE = .PRIVATE,
            .ANONYMOUS = true,
            .HUGETLB = true,
        };

        const map_flags: linux.MAP = @bitCast(huge_page_flavor | @as(u32, @bitCast(base_flags)));

        const slice_or_err = posix.mmap(
            hint,
            aligned_len,
            posix.PROT.READ | posix.PROT.WRITE,
            map_flags,
            -1,
            0,
        );
        if (slice_or_err) |slice| {
            std.log.info("Successfully allocated {} bytes using 1GB huge pages", .{aligned_len});
            return slice;
        } else |err| {
            std.log.debug("Failed to allocate 1GB huge pages: {s}", .{@errorName(err)});
            // Continue to next attempt
        }
    }

    // Second attempt: 2MB huge pages
    {
        const page_size: usize = 2 * 1024 * 1024; // 2MB
        const aligned_len = mem.alignForward(usize, requested_size, page_size);
        const hint = null;

        // Define MAP flags for 2MB huge pages
        const HUGETLB_FLAG_ENCODE_SHIFT = 26;
        const huge_page_flavor = @as(u32, 21) << HUGETLB_FLAG_ENCODE_SHIFT; // 2MB = 2^21

        const base_flags = linux.MAP{
            .TYPE = .PRIVATE,
            .ANONYMOUS = true,
            .HUGETLB = true,
        };

        const map_flags: linux.MAP = @bitCast(huge_page_flavor | @as(u32, @bitCast(base_flags)));

        const slice_or_err = posix.mmap(
            hint,
            aligned_len,
            posix.PROT.READ | posix.PROT.WRITE,
            map_flags,
            -1,
            0,
        );
        if (slice_or_err) |slice| {
            std.log.info("Successfully allocated {} bytes using 2MB huge pages", .{aligned_len});
            return slice;
        } else |err| {
            std.log.debug("Failed to allocate 2MB huge pages: {s}", .{@errorName(err)});
            // Continue to next attempt
        }
    }

    // Final attempt: normal pages
    {
        const page_size = 4096;
        const aligned_len = mem.alignForward(usize, requested_size, page_size);
        const hint = null;

        const slice = posix.mmap(
            hint,
            aligned_len,
            posix.PROT.READ | posix.PROT.WRITE,
            .{ .TYPE = .PRIVATE, .ANONYMOUS = true },
            -1,
            0,
        ) catch |err| {
            std.log.err("Failed to allocate normal pages: {s}", .{@errorName(err)});
            return err;
        };

        std.log.info("Successfully allocated {} bytes using normal pages", .{aligned_len});

        return slice;
    }
}

pub fn freeHugePages(slice: []u8) void {
    if (native_os != .linux) {
        return;
    }

    // We don't need to know what page size was used for allocation
    // munmap works regardless of the page size
    posix.munmap(@alignCast(slice));
}