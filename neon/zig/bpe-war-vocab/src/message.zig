const std = @import("std");
const Allocator = std.mem.Allocator;

/// Message types that can be sent from coordinator to workers
pub const CoordinatorMessageType = enum {
    /// Process a document (first or second pass)
    ProcessDocument,
    /// Find the top K strings based on current CMS data
    FindTopK,
    /// Request worker's CMS data (replaces MergeCMS)
    RequestCMS,
    /// Dump the worker's state to a file
    DumpState,
    /// Shutdown the worker thread
    Shutdown,
};

/// Message types that can be sent from workers to coordinator
pub const WorkerMessageType = enum {
    /// Document has been processed
    DocumentProcessed,
    /// Top K strings have been found
    TopKComplete,
    /// Worker is providing its CMS for merging (replaces CMSMergeComplete)
    ProvideCMS,
    /// State has been dumped to a file
    StateDumped,
    /// Error occurred during processing
    Error,
};

/// Fixed-size error message buffer to avoid heap allocations
pub const MAX_ERROR_MSG_LEN = 10;

/// Message sent from coordinator to a worker - converted to tagged union
pub const CoordinatorMessage = union(CoordinatorMessageType) {
    ProcessDocument: struct {
        worker_id: usize,
        document: []const u8,
        pass: u8,
    },
    FindTopK: struct {
        worker_id: usize,
    },
    RequestCMS: struct {
        worker_id: usize,
    },
    DumpState: struct {
        worker_id: usize,
        dump_path: []const u8,
    },
    Shutdown: struct {
        worker_id: usize,
    },
};

/// Message sent from a worker to the coordinator - converted to tagged union
pub const WorkerMessage = union(WorkerMessageType) {
    DocumentProcessed: struct {
        worker_id: usize, // Keep worker_id for consistency
        document: []const u8,
        pass: u8,
    },
    TopKComplete: struct {
        worker_id: usize, // Keep worker_id for consistency
    },
    ProvideCMS: struct {
        worker_id: usize, // Keep worker_id for consistency
        worker_cms: *anyopaque,
    },
    StateDumped: struct {
        worker_id: usize, // Keep worker_id for consistency
    },
    Error: struct {
        worker_id: usize, // Worker ID is especially important for error reporting
        error_buffer: [MAX_ERROR_MSG_LEN]u8,
        error_len: usize,

        /// Get error message as a slice
        pub fn getErrorMessage(self: *const @This()) ?[]const u8 {
            if (self.error_len == 0) {
                return null;
            }
            return self.error_buffer[0..self.error_len];
        }
    },
};

/// Create a new CoordinatorMessage for processing a document
pub fn createProcessDocumentMessage(worker_id: usize, document: []const u8, pass: u8) CoordinatorMessage {
    return CoordinatorMessage{
        .ProcessDocument = .{
            .worker_id = worker_id,
            .document = document,
            .pass = pass,
        },
    };
}

/// Create a new CoordinatorMessage for finding top K strings
pub fn createFindTopKMessage(worker_id: usize) CoordinatorMessage {
    return CoordinatorMessage{
        .FindTopK = .{
            .worker_id = worker_id,
        },
    };
}

/// Create a new CoordinatorMessage for requesting CMS data
pub fn createRequestCMSMessage(worker_id: usize) CoordinatorMessage {
    return CoordinatorMessage{
        .RequestCMS = .{
            .worker_id = worker_id,
        },
    };
}

/// Create a new CoordinatorMessage for dumping state
pub fn createDumpStateMessage(worker_id: usize, dump_path: []const u8) CoordinatorMessage {
    return CoordinatorMessage{
        .DumpState = .{
            .worker_id = worker_id,
            .dump_path = dump_path,
        },
    };
}

/// Create a new CoordinatorMessage for shutting down
pub fn createShutdownMessage(worker_id: usize) CoordinatorMessage {
    return CoordinatorMessage{
        .Shutdown = .{
            .worker_id = worker_id,
        },
    };
}

/// Create a new WorkerMessage for document processed
pub fn createDocumentProcessedMessage(worker_id: usize, document: []const u8, pass: u8) WorkerMessage {
    return WorkerMessage{
        .DocumentProcessed = .{
            .worker_id = worker_id,
            .document = document,
            .pass = pass,
        },
    };
}

/// Create a new WorkerMessage for top K complete
pub fn createTopKCompleteMessage(worker_id: usize) WorkerMessage {
    return WorkerMessage{
        .TopKComplete = .{
            .worker_id = worker_id,
        },
    };
}

/// Create a new WorkerMessage for providing CMS data
pub fn createProvideCMSMessage(worker_id: usize, worker_cms: *anyopaque) WorkerMessage {
    return WorkerMessage{
        .ProvideCMS = .{
            .worker_id = worker_id,
            .worker_cms = worker_cms,
        },
    };
}

/// Create a new WorkerMessage for state dumped
pub fn createStateDumpedMessage(worker_id: usize) WorkerMessage {
    return WorkerMessage{
        .StateDumped = .{
            .worker_id = worker_id,
        },
    };
}

/// Create a new WorkerMessage for error
pub fn createErrorMessage(worker_id: usize, error_msg: []const u8) WorkerMessage {
    const msg = WorkerMessage{
        .Error = .{
            .worker_id = worker_id,
            .error_buffer = undefined,
            .error_len = 0,
        },
    };

    // Still keeping this as a comment for compatibility with existing code
    _ = error_msg;

    // Copy error message to fixed buffer, truncating if necessary
    //const copy_len = @min(error_msg.len, MAX_ERROR_MSG_LEN);
    //@memcpy(msg.Error.error_buffer[0..copy_len], error_msg[0..copy_len]);
    //msg.Error.error_len = copy_len;

    return msg;
}

/// No memory to free in WorkerMessage anymore - kept for API compatibility
pub fn freeWorkerMessage(allocator: Allocator, message: *const WorkerMessage) void {
    _ = allocator;
    _ = message;
}

/// Free any memory owned by a CoordinatorMessage
pub fn freeCoordinatorMessage(allocator: Allocator, message: *const CoordinatorMessage) void {
    // In our current design, coordinator messages don't own memory that needs freeing
    // The document memory is managed separately by the coordinator
    _ = allocator;
    _ = message;
}
