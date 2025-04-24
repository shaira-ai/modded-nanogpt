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

/// Message sent from coordinator to a worker
pub const CoordinatorMessage = struct {
    /// Type of message
    msg_type: CoordinatorMessageType,

    /// Worker ID this message is for
    worker_id: usize,

    /// Document data (only used for ProcessDocument)
    /// Memory is allocated by the coordinator and must NOT be freed by the worker
    document: ?[]const u8 = null,

    /// Path for dumping state (only used for DumpState)
    dump_path: ?[]const u8 = null,

    /// Pass number (1 or 2) for ProcessDocument
    pass: ?u8 = null,
};

/// Fixed-size error message buffer to avoid heap allocations
pub const MAX_ERROR_MSG_LEN = 256;

/// Message sent from a worker to the coordinator
pub const WorkerMessage = struct {
    /// Type of message
    msg_type: WorkerMessageType,

    /// Worker ID this message is from
    worker_id: usize,

    /// Document that was processed (only used for DocumentProcessed)
    /// This is the same pointer that was received in CoordinatorMessage
    document: ?[]const u8 = null,

    /// Error message (only used for Error)
    /// Using a stack-allocated buffer instead of heap-allocated string
    error_buffer: [MAX_ERROR_MSG_LEN]u8 = undefined,
    error_len: usize = 0,

    /// Pass number (1 or 2) for DocumentProcessed
    pass: ?u8 = null,

    /// Worker's CMS pointer (only used for ProvideCMS)
    worker_cms: ?*anyopaque = null,

    /// Get error message as a slice
    pub fn getErrorMessage(self: *const WorkerMessage) ?[]const u8 {
        if (self.msg_type != .Error or self.error_len == 0) {
            return null;
        }
        return self.error_buffer[0..self.error_len];
    }
};

/// Create a new CoordinatorMessage for processing a document
pub fn createProcessDocumentMessage(worker_id: usize, document: []const u8, pass: u8) CoordinatorMessage {
    return CoordinatorMessage{
        .msg_type = .ProcessDocument,
        .worker_id = worker_id,
        .document = document,
        .pass = pass,
    };
}

/// Create a new CoordinatorMessage for finding top K strings
pub fn createFindTopKMessage(worker_id: usize) CoordinatorMessage {
    return CoordinatorMessage{
        .msg_type = .FindTopK,
        .worker_id = worker_id,
    };
}

/// Create a new CoordinatorMessage for requesting CMS data
pub fn createRequestCMSMessage(worker_id: usize) CoordinatorMessage {
    return CoordinatorMessage{
        .msg_type = .RequestCMS,
        .worker_id = worker_id,
    };
}

/// Create a new CoordinatorMessage for dumping state
pub fn createDumpStateMessage(worker_id: usize, dump_path: []const u8) CoordinatorMessage {
    return CoordinatorMessage{
        .msg_type = .DumpState,
        .worker_id = worker_id,
        .dump_path = dump_path,
    };
}

/// Create a new CoordinatorMessage for shutting down
pub fn createShutdownMessage(worker_id: usize) CoordinatorMessage {
    return CoordinatorMessage{
        .msg_type = .Shutdown,
        .worker_id = worker_id,
    };
}

/// Create a new WorkerMessage for document processed
pub fn createDocumentProcessedMessage(worker_id: usize, document: []const u8, pass: u8) WorkerMessage {
    return WorkerMessage{
        .msg_type = .DocumentProcessed,
        .worker_id = worker_id,
        .document = document,
        .pass = pass,
    };
}

/// Create a new WorkerMessage for top K complete
pub fn createTopKCompleteMessage(worker_id: usize) WorkerMessage {
    return WorkerMessage{
        .msg_type = .TopKComplete,
        .worker_id = worker_id,
    };
}

/// Create a new WorkerMessage for providing CMS data
pub fn createProvideCMSMessage(worker_id: usize, worker_cms: *anyopaque) WorkerMessage {
    return WorkerMessage{
        .msg_type = .ProvideCMS,
        .worker_id = worker_id,
        .worker_cms = worker_cms,
    };
}

/// Create a new WorkerMessage for state dumped
pub fn createStateDumpedMessage(worker_id: usize) WorkerMessage {
    return WorkerMessage{
        .msg_type = .StateDumped,
        .worker_id = worker_id,
    };
}

/// Create a new WorkerMessage for error
pub fn createErrorMessage(worker_id: usize, error_msg: []const u8) WorkerMessage {
    var msg = WorkerMessage{
        .msg_type = .Error,
        .worker_id = worker_id,
        .error_len = 0,
    };

    // Copy error message to fixed buffer, truncating if necessary
    const copy_len = @min(error_msg.len, MAX_ERROR_MSG_LEN);
    @memcpy(msg.error_buffer[0..copy_len], error_msg[0..copy_len]);
    msg.error_len = copy_len;

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
