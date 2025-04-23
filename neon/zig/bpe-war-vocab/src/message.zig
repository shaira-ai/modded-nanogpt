const std = @import("std");
const Allocator = std.mem.Allocator;

/// Message types that can be sent from coordinator to workers
pub const CoordinatorMessageType = enum {
    /// Process a document (first or second pass)
    ProcessDocument,
    /// Find the top K strings based on current CMS data
    FindTopK,
    /// Merge this worker's CMS into the global CMS
    MergeCMS,
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
    /// CMS has been merged with the global CMS
    CMSMergeComplete,
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

    /// Pointer to globally merged CMS (only used for MergeCMS)
    global_cms: ?*anyopaque = null,
};

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
    error_msg: ?[]const u8 = null,

    /// Error memory is allocated by the worker and must be freed by the coordinator
    error_needs_free: bool = false,

    /// Pass number (1 or 2) for DocumentProcessed
    pass: ?u8 = null,
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

/// Create a new CoordinatorMessage for merging CMS
pub fn createMergeCMSMessage(worker_id: usize, global_cms: *anyopaque) CoordinatorMessage {
    return CoordinatorMessage{
        .msg_type = .MergeCMS,
        .worker_id = worker_id,
        .global_cms = global_cms,
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

/// Create a new WorkerMessage for CMS merge complete
pub fn createCMSMergeCompleteMessage(worker_id: usize) WorkerMessage {
    return WorkerMessage{
        .msg_type = .CMSMergeComplete,
        .worker_id = worker_id,
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
pub fn createErrorMessage(worker_id: usize, error_msg: []const u8, error_needs_free: bool) WorkerMessage {
    return WorkerMessage{
        .msg_type = .Error,
        .worker_id = worker_id,
        .error_msg = error_msg,
        .error_needs_free = error_needs_free,
    };
}

/// Free any memory owned by a WorkerMessage
pub fn freeWorkerMessage(allocator: Allocator, message: *const WorkerMessage) void {
    if (message.msg_type == .Error and message.error_needs_free and message.error_msg != null) {
        allocator.free(message.error_msg.?);
    }
}

/// Free any memory owned by a CoordinatorMessage
pub fn freeCoordinatorMessage(allocator: Allocator, message: *const CoordinatorMessage) void {
    // In our current design, coordinator messages don't own memory that needs freeing
    // The document memory is managed separately by the coordinator
    _ = allocator;
    _ = message;
}
