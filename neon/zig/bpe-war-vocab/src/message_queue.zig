const std = @import("std");
const spsc = @import("spsc.zig");
const message = @import("message.zig");
const Allocator = std.mem.Allocator;

/// SPSC queue capacity - must be a power of 2
pub const QUEUE_CAPACITY = 32;

/// Queue for coordinator messages
pub const CoordinatorMessageQueue = struct {
    queue: spsc.BoundedQueue(message.CoordinatorMessage, QUEUE_CAPACITY),
    allocator: Allocator,

    /// Initialize a new queue
    pub fn init(allocator: Allocator) !*CoordinatorMessageQueue {
        const self = try allocator.create(CoordinatorMessageQueue);
        errdefer allocator.destroy(self);

        self.queue = try spsc.BoundedQueue(message.CoordinatorMessage, QUEUE_CAPACITY).init(allocator);
        self.allocator = allocator;

        return self;
    }

    /// Clean up resources
    pub fn deinit(self: *CoordinatorMessageQueue) void {
        self.queue.deinit(self.allocator);
        self.allocator.destroy(self);
    }

    /// Push a message to the queue
    pub fn push(self: *CoordinatorMessageQueue, msg: message.CoordinatorMessage) bool {
        return self.queue.push(msg);
    }

    /// Pop a message from the queue
    pub fn pop(self: *CoordinatorMessageQueue) ?message.CoordinatorMessage {
        return self.queue.pop();
    }

    /// Get number of messages in the queue
    pub fn count(self: *CoordinatorMessageQueue) usize {
        return self.queue.count();
    }
};

/// Queue for worker messages
pub const WorkerMessageQueue = struct {
    queue: spsc.BoundedQueue(message.WorkerMessage, QUEUE_CAPACITY),
    allocator: Allocator,

    /// Initialize a new queue
    pub fn init(allocator: Allocator) !*WorkerMessageQueue {
        const self = try allocator.create(WorkerMessageQueue);
        errdefer allocator.destroy(self);

        self.queue = try spsc.BoundedQueue(message.WorkerMessage, QUEUE_CAPACITY).init(allocator);
        self.allocator = allocator;

        return self;
    }

    /// Clean up resources
    pub fn deinit(self: *WorkerMessageQueue) void {
        // Free any error messages still in the queue
        while (self.queue.pop()) |msg| {
            message.freeWorkerMessage(self.allocator, &msg);
        }

        self.queue.deinit(self.allocator);
        self.allocator.destroy(self);
    }

    /// Push a message to the queue
    pub fn push(self: *WorkerMessageQueue, msg: message.WorkerMessage) bool {
        return self.queue.push(msg);
    }

    /// Pop a message from the queue
    pub fn pop(self: *WorkerMessageQueue) ?message.WorkerMessage {
        return self.queue.pop();
    }

    /// Get number of messages in the queue
    pub fn count(self: *WorkerMessageQueue) usize {
        return self.queue.count();
    }
};
