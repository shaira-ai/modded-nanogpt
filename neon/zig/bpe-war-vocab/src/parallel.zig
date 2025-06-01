const std = @import("std");
const Allocator = std.mem.Allocator;
const Thread = std.Thread;
const time = std.time;

const BakaCorasick = @import("baka_corasick.zig").BakaCorasick;
const fineweb = @import("data_loader.zig").FinewebDataLoader;
const VocabLearnerModule = @import("vocab_learner.zig");
const SampleStats = VocabLearnerModule.SampleStats;
const TokenStats = VocabLearnerModule.TokenStats;
const MatchInfo = VocabLearnerModule.MatchInfo;
const BoundedQueue = @import("spsc.zig").BoundedQueue;

// Compile-time configuration for queue type
const USE_SPMC_QUEUE = true;

/// Packed struct for coordinating thread pool state and notifications
const Sync = packed struct {
    /// Tracks the number of threads not searching for Tasks
    idle: u14 = 0,
    /// Tracks the number of threads spawned
    spawned: u14 = 0,
    /// what you see is what you get
    unused: bool = false,
    notified: bool = false,
    /// The current state of the thread pool
    state: enum(u2) {
        pending = 0,
        signaled,
        waking,
        shutdown,
    } = .pending,
};

/// An event which stores 1 semaphore token and is multi-threaded safe.
/// The event can be shutdown(), waking up all wait()ing threads and
/// making subsequent wait()'s return immediately.
const Event = struct {
    state: std.atomic.Value(u32) = std.atomic.Value(u32).init(EMPTY),

    const EMPTY = 0;
    const WAITING = 1;
    const NOTIFIED = 2;
    const SHUTDOWN = 3;

    fn wait(self: *Event) void {
        var acquire_with: u32 = EMPTY;
        var state = self.state.load(.acquire); // Use acquire ordering to see shutdown properly

        while (true) {
            if (state == SHUTDOWN) {
                return;
            }

            if (state == NOTIFIED) {
                state = self.state.cmpxchgWeak(
                    state,
                    acquire_with,
                    .acquire,
                    .monotonic,
                ) orelse return;
                continue;
            }

            // There is no notification to consume, we should wait on the event by ensuring its WAITING.
            if (state != WAITING) {
                state = self.state.cmpxchgWeak(
                    state,
                    WAITING,
                    .monotonic,
                    .monotonic,
                ) orelse continue;
                continue;
            }

            std.Thread.Futex.wait(&self.state, WAITING);
            state = self.state.load(.acquire); // Use acquire ordering when reloading after futex
            acquire_with = WAITING;
        }
    }

    fn notify(self: *Event) void {
        return self.wake(NOTIFIED, 1);
    }

    fn shutdown(self: *Event) void {
        return self.wake(SHUTDOWN, std.math.maxInt(u32));
    }

    fn wake(self: *Event, release_with: u32, wake_threads: u32) void {
        const state = self.state.swap(release_with, .release);

        if (state == WAITING) {
            std.Thread.Futex.wake(&self.state, wake_threads);
        }
    }
};

/// Intrusive linked list node for lock-free data structures
const Node = if (USE_SPMC_QUEUE) struct {
    next: ?*Node = null,

    /// A linked list of nodes
    const List = struct {
        head: *Node,
        tail: *Node,
    };

    /// Unbounded multi-producer, single-consumer queue
    const Queue = struct {
        stack: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),
        cache: ?*Node = null,

        const HAS_CACHE: usize = 0b01;
        const IS_CONSUMING: usize = 0b10;
        const PTR_MASK: usize = ~(HAS_CACHE | IS_CONSUMING);

        comptime {
            std.debug.assert(@alignOf(Node) >= ((IS_CONSUMING | HAS_CACHE) + 1));
        }

        fn push(noalias self: *Queue, list: List) void {
            var stack = self.stack.load(.monotonic);
            while (true) {
                // Attach the list to the stack
                list.tail.next = @as(?*Node, @ptrFromInt(stack & PTR_MASK));

                // Update the stack with the list, preserving consumer bits
                var new_stack = @intFromPtr(list.head);
                std.debug.assert(new_stack & ~PTR_MASK == 0);
                new_stack |= (stack & ~PTR_MASK);

                if (self.stack.cmpxchgWeak(stack, new_stack, .release, .monotonic)) |updated_stack| {
                    stack = updated_stack;
                } else {
                    break;
                }
            }
        }

        fn tryAcquireConsumer(self: *Queue) error{ Empty, Contended }!?*Node {
            var stack = self.stack.load(.monotonic);
            while (true) {
                if (stack & IS_CONSUMING != 0)
                    return error.Contended;
                if (stack & (HAS_CACHE | PTR_MASK) == 0)
                    return error.Empty;

                var new_stack = stack | HAS_CACHE | IS_CONSUMING;
                if (stack & HAS_CACHE == 0) {
                    std.debug.assert(stack & PTR_MASK != 0);
                    new_stack &= ~PTR_MASK;
                }

                if (self.stack.cmpxchgWeak(stack, new_stack, .acquire, .monotonic)) |updated_stack| {
                    stack = updated_stack;
                } else {
                    return self.cache orelse @as(*Node, @ptrFromInt(stack & PTR_MASK));
                }
            }
        }

        fn releaseConsumer(noalias self: *Queue, noalias consumer: ?*Node) void {
            var remove = IS_CONSUMING;
            if (consumer == null)
                remove |= HAS_CACHE;

            self.cache = consumer;
            _ = self.stack.fetchSub(remove, .release);
        }

        fn pop(noalias self: *Queue, noalias consumer_ref: *?*Node) ?*Node {
            if (consumer_ref.*) |node| {
                consumer_ref.* = node.next;
                return node;
            }

            var stack = self.stack.load(.monotonic);
            std.debug.assert(stack & IS_CONSUMING != 0);
            if (stack & PTR_MASK == 0) {
                return null;
            }

            stack = self.stack.swap(HAS_CACHE | IS_CONSUMING, .acquire);
            std.debug.assert(stack & IS_CONSUMING != 0);
            std.debug.assert(stack & PTR_MASK != 0);

            const node = @as(*Node, @ptrFromInt(stack & PTR_MASK));
            consumer_ref.* = node.next;
            return node;
        }
    };

    /// Bounded single-producer, multi-consumer ring buffer for work stealing
    const Buffer = struct {
        head: std.atomic.Value(Index) = std.atomic.Value(Index).init(0),
        tail: std.atomic.Value(Index) = std.atomic.Value(Index).init(0),
        array: [capacity]std.atomic.Value(*Node) = undefined,

        const Index = u32;
        const capacity = 256;

        comptime {
            std.debug.assert(std.math.maxInt(Index) >= capacity);
            std.debug.assert(std.math.isPowerOfTwo(capacity));
        }

        fn init() Buffer {
            var self = Buffer{};
            const dummy_node = @as(*Node, @ptrFromInt(@alignOf(Node)));
            for (&self.array) |*slot| {
                slot.* = std.atomic.Value(*Node).init(dummy_node);
            }
            return self;
        }

        fn push(noalias self: *Buffer, noalias list: *List) error{Overflow}!void {
            var head = self.head.load(.monotonic);
            var tail = self.tail.load(.monotonic);

            while (true) {
                var size = tail -% head;
                std.debug.assert(size <= capacity);

                if (size < capacity) {
                    var nodes: ?*Node = list.head;
                    while (size < capacity) : (size += 1) {
                        const node = nodes orelse break;
                        nodes = node.next;

                        self.array[tail % capacity].store(node, .unordered);
                        tail +%= 1;
                    }

                    self.tail.store(tail, .release);
                    list.head = nodes orelse return;
                    std.atomic.spinLoopHint();
                    head = self.head.load(.monotonic);
                    continue;
                }

                // Overflow half the tasks to make room
                var migrate = size / 2;
                if (self.head.cmpxchgWeak(head, head +% migrate, .acquire, .monotonic)) |updated_head| {
                    head = updated_head;
                } else {
                    // Link migrated nodes together
                    const first = self.array[head % capacity].load(.monotonic);
                    var current_head = head;
                    while (migrate > 0) : (migrate -= 1) {
                        const prev = self.array[current_head % capacity].load(.monotonic);
                        current_head +%= 1;
                        prev.next = if (current_head < head + migrate)
                            self.array[current_head % capacity].load(.monotonic)
                        else
                            null;
                    }

                    const last = self.array[(current_head -% 1) % capacity].load(.monotonic);
                    last.next = list.head;
                    list.tail.next = null;
                    list.head = first;
                    return error.Overflow;
                }
            }
        }

        fn pop(self: *Buffer) ?*Node {
            var head = self.head.load(.monotonic);
            const tail = self.tail.load(.monotonic);

            while (true) {
                const size = tail -% head;
                std.debug.assert(size <= capacity);
                if (size == 0) {
                    return null;
                }

                if (self.head.cmpxchgWeak(head, head +% 1, .acquire, .monotonic)) |updated_head| {
                    head = updated_head;
                } else {
                    return self.array[head % capacity].load(.monotonic);
                }
            }
        }

        const Stole = struct {
            node: *Node,
            pushed: bool,
        };

        fn consume(noalias self: *Buffer, noalias queue: *Queue) ?Stole {
            var consumer = queue.tryAcquireConsumer() catch return null;
            defer queue.releaseConsumer(consumer);

            const head = self.head.load(.monotonic);
            const tail = self.tail.load(.monotonic);

            const size = tail -% head;
            std.debug.assert(size <= capacity);
            std.debug.assert(size == 0);

            var pushed: Index = 0;
            while (pushed < capacity) : (pushed += 1) {
                const node = queue.pop(&consumer) orelse break;
                self.array[(tail +% pushed) % capacity].store(node, .unordered);
            }

            const node = queue.pop(&consumer) orelse blk: {
                if (pushed == 0) return null;
                pushed -= 1;
                break :blk self.array[(tail +% pushed) % capacity].load(.monotonic);
            };

            if (pushed > 0) self.tail.store(tail +% pushed, .release);
            return Stole{
                .node = node,
                .pushed = pushed > 0,
            };
        }

        fn steal(noalias self: *Buffer, noalias buffer: *Buffer) ?Stole {
            const head = self.head.load(.monotonic);
            const tail = self.tail.load(.monotonic);

            const size = tail -% head;
            std.debug.assert(size <= capacity);
            std.debug.assert(size == 0);

            while (true) : (std.atomic.spinLoopHint()) {
                const buffer_head = buffer.head.load(.acquire);
                const buffer_tail = buffer.tail.load(.acquire);

                const buffer_size = buffer_tail -% buffer_head;
                if (buffer_size > capacity) {
                    continue;
                }

                const steal_size = buffer_size - (buffer_size / 2);
                if (steal_size == 0) {
                    return null;
                }

                var i: Index = 0;
                while (i < steal_size) : (i += 1) {
                    const node = buffer.array[(buffer_head +% i) % capacity].load(.unordered);
                    self.array[(tail +% i) % capacity].store(node, .unordered);
                }

                if (buffer.head.cmpxchgWeak(buffer_head, buffer_head +% steal_size, .acq_rel, .monotonic)) |_| {
                    continue;
                } else {
                    const pushed = steal_size - 1;
                    const node = self.array[(tail +% pushed) % capacity].load(.monotonic);

                    if (pushed > 0) self.tail.store(tail +% pushed, .release);
                    return Stole{
                        .node = node,
                        .pushed = pushed > 0,
                    };
                }
            }
        }
    };
} else void;

const ProcessDocumentMessage = if (USE_SPMC_QUEUE) struct {
    id: usize,
    document: []const u8,
    automaton: *BakaCorasick,
} else struct {
    id: usize,
    document: []const u8,
};

const CountTokensInDocumentMessage = if (USE_SPMC_QUEUE) struct {
    id: usize,
    document: []const u8,
} else struct {
    id: usize,
    document: []const u8,
};

const DocumentProcessedMessage = if (USE_SPMC_QUEUE) struct {
    id: usize,
    work_task: *WorkTask,
} else struct {
    id: usize,
};

const TokenCountMessage = if (USE_SPMC_QUEUE) struct {
    id: usize,
    token_count: u64,
    work_task: *WorkTask,
} else struct {
    id: usize,
    token_count: u64,
};

const ErrorMessage = struct {
    worker_id: usize,
    message: []const u8,
};

const ResetMessage = struct {
    candidates: []SampleStats,
    automaton: *BakaCorasick,
};

// Intrusive work task for SPMC mode
const WorkTask = if (USE_SPMC_QUEUE) struct {
    node: Node = .{},
    data: union(enum) {
        ProcessDocument: ProcessDocumentMessage,
        CountTokensInDocument: CountTokensInDocumentMessage,
        Shutdown: usize,
    },
    recycle_next: ?*WorkTask = null,

    pub fn fromNode(node_ptr: *Node) *WorkTask {
        return @fieldParentPtr("node", node_ptr);
    }

    pub fn toList(self: *WorkTask) Node.List {
        return Node.List{
            .head = &self.node,
            .tail = &self.node,
        };
    }
} else void;

const SubmissionQueueEntry = if (USE_SPMC_QUEUE) WorkTask else union(enum) {
    Reset: ResetMessage,
    ProcessDocument: ProcessDocumentMessage,
    CountTokensInDocument: CountTokensInDocumentMessage,
    Shutdown: usize,
};

const CompletionQueueEntry = union(enum) {
    DocumentProcessed: DocumentProcessedMessage,
    TokenCount: TokenCountMessage,
    Error: ErrorMessage,
};

// Worker implementation with Event integration
pub const Worker = struct {
    allocator: Allocator,
    id: usize,
    vocab_learner: *VocabLearnerModule.VocabLearner,
    candidate_stats: []SampleStats,
    candidate_automaton: *BakaCorasick,
    token_idx_to_least_end_pos: []u32,
    lookbacks: std.ArrayList(u64),
    dp_solution: std.ArrayList(u32),
    matches: std.ArrayList(MatchInfo),

    submission_queue: if (USE_SPMC_QUEUE)
        *Node.Queue
    else
        *BoundedQueue(SubmissionQueueEntry, 256),

    completion_queue: *BoundedQueue(CompletionQueueEntry, 256),

    run_buffer: if (USE_SPMC_QUEUE) Node.Buffer else void,
    workers: if (USE_SPMC_QUEUE) []Worker else void,
    next_steal_target: if (USE_SPMC_QUEUE) usize else void,
    shutdown_flag: if (USE_SPMC_QUEUE) *std.atomic.Value(bool) else void,

    coordinator: if (USE_SPMC_QUEUE) *ParallelDP else void,

    thread: ?Thread = null,
    running: bool = false,
    candidates_length: usize,
    debug: bool,
    token_count: u64 = 0,
    current_n_candidates: usize = 0,

    pub fn init(
        allocator: Allocator,
        id: usize,
        vocab_learner: *VocabLearnerModule.VocabLearner,
        candidate_automaton: *BakaCorasick,
        candidates_length: usize,
        submission_queue: if (USE_SPMC_QUEUE)
            *Node.Queue
        else
            *BoundedQueue(SubmissionQueueEntry, 256),
        completion_queue: *BoundedQueue(CompletionQueueEntry, 256),
        debug: bool,
    ) !Worker {
        const lookbacks = std.ArrayList(u64).init(allocator);
        const dp_solution = std.ArrayList(u32).init(allocator);
        const matches = std.ArrayList(MatchInfo).init(allocator);

        const token_idx_to_least_end_pos = try allocator.alloc(u32, candidates_length);
        const candidate_stats = try allocator.alloc(SampleStats, candidates_length);

        return Worker{
            .allocator = allocator,
            .id = id,
            .vocab_learner = vocab_learner,
            .candidate_stats = candidate_stats,
            .candidate_automaton = candidate_automaton,
            .token_idx_to_least_end_pos = token_idx_to_least_end_pos,
            .lookbacks = lookbacks,
            .dp_solution = dp_solution,
            .matches = matches,
            .submission_queue = submission_queue,
            .completion_queue = completion_queue,
            .run_buffer = if (USE_SPMC_QUEUE) Node.Buffer.init() else {},
            .workers = if (USE_SPMC_QUEUE) undefined else {},
            .next_steal_target = if (USE_SPMC_QUEUE) 0 else {},
            .shutdown_flag = if (USE_SPMC_QUEUE) undefined else {},
            .coordinator = if (USE_SPMC_QUEUE) undefined else {},
            .candidates_length = candidates_length,
            .debug = debug,
        };
    }

    pub fn deinit(self: *Worker) void {
        if (self.thread) |thread| {
            thread.join();
            self.thread = null;
        }

        self.lookbacks.deinit();
        self.dp_solution.deinit();
        self.matches.deinit();
        self.allocator.free(self.token_idx_to_least_end_pos);
        self.allocator.free(self.candidate_stats);
    }

    pub fn start(self: *Worker) !void {
        if (self.thread != null) return;

        self.running = true;
        self.thread = try Thread.spawn(.{}, Worker.run, .{self});

        if (self.debug) {
            std.debug.print("[Worker {d}] Started\n", .{self.id});
        }
    }

    pub fn stop(self: *Worker) void {
        self.running = false;
    }

    // Conditional method signatures for exact SPSC compatibility
    fn processDocument(self: *Worker, document: []const u8, doc_id: usize, automaton_param: if (USE_SPMC_QUEUE) *BakaCorasick else void, work_task_param: if (USE_SPMC_QUEUE) *WorkTask else void) !void {
        const automaton = if (USE_SPMC_QUEUE) automaton_param else self.candidate_automaton;

        try self.vocab_learner.evaluateCandidatesOnDocumentDP(
            self.candidate_stats[0..self.current_n_candidates],
            automaton,
            document,
            &self.lookbacks,
            &self.dp_solution,
            &self.matches,
            self.token_idx_to_least_end_pos[0..self.current_n_candidates],
        );

        const result_msg = if (USE_SPMC_QUEUE)
            CompletionQueueEntry{
                .DocumentProcessed = DocumentProcessedMessage{
                    .id = doc_id,
                    .work_task = work_task_param,
                },
            }
        else
            CompletionQueueEntry{
                .DocumentProcessed = DocumentProcessedMessage{
                    .id = doc_id,
                },
            };

        var pushed = self.completion_queue.push(result_msg);
        while (!pushed) {
            std.time.sleep(1 * std.time.ns_per_ms);
            pushed = self.completion_queue.push(result_msg);
        }
    }

    fn countTokensInDocument(self: *Worker, document: []const u8, doc_id: usize, work_task_param: if (USE_SPMC_QUEUE) *WorkTask else void) !void {
        const token_count = try self.vocab_learner.getDocumentTokenCount(document, &self.lookbacks, &self.dp_solution);

        const result_msg = if (USE_SPMC_QUEUE)
            CompletionQueueEntry{
                .TokenCount = TokenCountMessage{
                    .id = doc_id,
                    .token_count = token_count,
                    .work_task = work_task_param,
                },
            }
        else
            CompletionQueueEntry{
                .TokenCount = TokenCountMessage{
                    .id = doc_id,
                    .token_count = token_count,
                },
            };

        var pushed = self.completion_queue.push(result_msg);
        while (!pushed) {
            std.time.sleep(1 * std.time.ns_per_ms);
            pushed = self.completion_queue.push(result_msg);
        }
    }

    // SPMC work dequeue
    fn dequeueWork(self: *Worker) ?*WorkTask {
        if (USE_SPMC_QUEUE) {
            // Check local buffer first
            if (self.run_buffer.pop()) |node| {
                return WorkTask.fromNode(node);
            }

            // Try to consume from global queue
            if (self.run_buffer.consume(self.submission_queue)) |stole| {
                return WorkTask.fromNode(stole.node);
            }

            // Work stealing from other workers
            for (0..self.workers.len) |_| {
                const target_idx = self.next_steal_target % self.workers.len;
                self.next_steal_target +%= 1;

                if (target_idx == self.id) continue;

                if (self.run_buffer.steal(&self.workers[target_idx].run_buffer)) |stole| {
                    return WorkTask.fromNode(stole.node);
                }
            }

            return null;
        } else {
            @panic("dequeueWork() should not be called in SPSC mode");
        }
    }

    fn run(self: *Worker) !void {
        if (self.debug) {
            std.debug.print("[Worker {d}] Running\n", .{self.id});
        }

        var is_waking = false;
        while (self.running) {
            if (USE_SPMC_QUEUE) {
                // Check shutdown flag first
                if (self.shutdown_flag.load(.monotonic)) {
                    self.running = false;
                    break;
                }

                // proper kprotty pattern is: wait first, then look for work
                is_waking = self.coordinator.wait(is_waking) catch {
                    self.running = false;
                    break;
                };

                // Now that we're awake, look for work
                var found_work = false;
                while (self.dequeueWork()) |work_task| {
                    found_work = true;

                    // if we found work and we're the waking thread, notify the next thread < -- >
                    if (is_waking) {
                        self.coordinator.notify(is_waking);
                        is_waking = false;
                    }

                    switch (work_task.data) {
                        .ProcessDocument => |process_data| {
                            self.processDocument(process_data.document, process_data.id, process_data.automaton, work_task) catch |err| {
                                const error_msg = std.fmt.allocPrint(self.allocator, "Error processing document: {s}", .{@errorName(err)}) catch "OutOfMemory";
                                defer if (!std.mem.eql(u8, error_msg, "OutOfMemory")) self.allocator.free(error_msg);
                                const error_message = CompletionQueueEntry{
                                    .Error = ErrorMessage{
                                        .worker_id = self.id,
                                        .message = error_msg,
                                    },
                                };
                                _ = self.completion_queue.push(error_message);
                            };
                        },
                        .CountTokensInDocument => |count_tokens_in_document| {
                            self.countTokensInDocument(count_tokens_in_document.document, count_tokens_in_document.id, work_task) catch |err| {
                                const error_msg = std.fmt.allocPrint(self.allocator, "Error counting tokens in document: {s}", .{@errorName(err)}) catch "OutOfMemory";
                                defer if (!std.mem.eql(u8, error_msg, "OutOfMemory")) self.allocator.free(error_msg);
                                const error_message = CompletionQueueEntry{
                                    .Error = ErrorMessage{
                                        .worker_id = self.id,
                                        .message = error_msg,
                                    },
                                };
                                _ = self.completion_queue.push(error_message);
                            };
                        },
                        .Shutdown => {
                            if (self.debug) {
                                std.debug.print("[Worker {d}] Shutting down\n", .{self.id});
                            }
                            self.running = false;
                        },
                    }
                }

                // and if we're the waking thread but found no work,
                // we must pass on the waking status to avoid deadlock
                if (is_waking and !found_work) {
                    self.coordinator.notify(is_waking);
                    is_waking = false;
                }
            } else {
                if (self.submission_queue.pop()) |msg| {
                    switch (msg) {
                        .Reset => |reset_message| {
                            const coordinator_stats = reset_message.candidates;
                            const automaton = reset_message.automaton;
                            self.current_n_candidates = coordinator_stats.len;
                            if (self.candidate_stats.len < self.current_n_candidates) {
                                @panic("oh no!1");
                            }
                            for (self.candidate_stats[0..self.current_n_candidates], 0..) |*stats, idx| {
                                stats.sampled_occurrences = 0;
                                stats.sampled_savings = 0;
                                stats.token_id = coordinator_stats[idx].token_id;
                            }
                            self.candidate_automaton = automaton;
                        },
                        .ProcessDocument => |process_data| {
                            self.processDocument(process_data.document, process_data.id, {}, {}) catch |err| {
                                const error_msg = try std.fmt.allocPrint(self.allocator, "Error processing document: {s}", .{@errorName(err)});
                                const error_message = CompletionQueueEntry{
                                    .Error = ErrorMessage{
                                        .worker_id = self.id,
                                        .message = error_msg,
                                    },
                                };
                                _ = self.completion_queue.push(error_message);
                            };
                        },
                        .CountTokensInDocument => |count_tokens_in_document| {
                            self.countTokensInDocument(count_tokens_in_document.document, count_tokens_in_document.id, {}) catch |err| {
                                const error_msg = try std.fmt.allocPrint(self.allocator, "Error counting tokens in document: {s}", .{@errorName(err)});
                                const error_message = CompletionQueueEntry{
                                    .Error = ErrorMessage{
                                        .worker_id = self.id,
                                        .message = error_msg,
                                    },
                                };
                                _ = self.completion_queue.push(error_message);
                            };
                        },
                        .Shutdown => {
                            if (self.debug) {
                                std.debug.print("[Worker {d}] Shutting down\n", .{self.id});
                            }
                            self.running = false;
                        },
                    }
                } else {
                    // Still use small sleep for SPSC mode for now
                    std.time.sleep(1 * std.time.ns_per_ms);
                }
            }
        }

        if (self.debug) {
            std.debug.print("[Worker {d}] Stopped\n", .{self.id});
        }
    }
};

// coordinator implementation
pub const ParallelDP = struct {
    allocator: Allocator,
    vocab_learner: *VocabLearnerModule.VocabLearner,
    debug: bool,

    submission_queues: if (USE_SPMC_QUEUE)
        Node.Queue
    else
        []BoundedQueue(SubmissionQueueEntry, 256),

    completion_queues: []BoundedQueue(CompletionQueueEntry, 256),
    workers: []Worker,
    num_workers: usize,
    next_document_id: usize = 0,

    sync: if (USE_SPMC_QUEUE) std.atomic.Value(u32) else void,
    idle_event: if (USE_SPMC_QUEUE) Event else void,

    // Conditional outstanding jobs tracking
    n_outstanding_jobs: if (USE_SPMC_QUEUE)
        std.atomic.Value(usize)
    else
        []usize,

    pending_documents: if (USE_SPMC_QUEUE) void else [][]const u8,
    pending_documents_free_list: if (USE_SPMC_QUEUE) void else []usize,
    n_free_pending_documents: if (USE_SPMC_QUEUE) void else usize,
    queue_depth_for_tokenize: if (USE_SPMC_QUEUE) void else usize,
    queue_depth_for_dp: if (USE_SPMC_QUEUE) void else usize,

    processed_count: usize = 0,
    started_workers: bool = false,

    work_task_pool: if (USE_SPMC_QUEUE) []WorkTask else void,
    work_task_pool_index: if (USE_SPMC_QUEUE) usize else void,
    recycled_tasks: if (USE_SPMC_QUEUE) std.atomic.Value(?*WorkTask) else void,
    shutdown_flag: if (USE_SPMC_QUEUE) std.atomic.Value(bool) else void,

    pub fn init(
        allocator: Allocator,
        vocab_learner: *VocabLearnerModule.VocabLearner,
        debug: bool,
    ) !ParallelDP {
        const num_workers = try Thread.getCpuCount();

        // Initialize conditional queue types
        var submission_queues: if (USE_SPMC_QUEUE)
            Node.Queue
        else
            []BoundedQueue(SubmissionQueueEntry, 256) = undefined;

        if (USE_SPMC_QUEUE) {
            submission_queues = Node.Queue{};
        } else {
            submission_queues = try allocator.alloc(BoundedQueue(SubmissionQueueEntry, 256), num_workers);
            errdefer allocator.free(submission_queues);
            for (0..num_workers) |i| {
                submission_queues[i] = try BoundedQueue(SubmissionQueueEntry, 256).init(allocator);
            }
        }

        var completion_queues = try allocator.alloc(BoundedQueue(CompletionQueueEntry, 256), num_workers);
        errdefer allocator.free(completion_queues);

        // Initialize queues
        for (0..num_workers) |i| {
            completion_queues[i] = try BoundedQueue(CompletionQueueEntry, 256).init(allocator);
        }

        // Initialize central coordination
        const sync: if (USE_SPMC_QUEUE) std.atomic.Value(u32) else void = if (USE_SPMC_QUEUE)
            std.atomic.Value(u32).init(@bitCast(Sync{}))
        else {};

        const idle_event: if (USE_SPMC_QUEUE) Event else void = if (USE_SPMC_QUEUE)
            Event{}
        else {};

        // Allocate worker array
        const workers = try allocator.alloc(Worker, num_workers);
        errdefer allocator.free(workers);

        // Conditional SPSC-only initialization
        const pending_documents: if (USE_SPMC_QUEUE) void else [][]const u8 = if (!USE_SPMC_QUEUE) blk: {
            const queue_depth_for_tokenize = 20;
            const queue_depth_for_dp = 20;
            const larger_queue_depth = @max(queue_depth_for_tokenize, queue_depth_for_dp);
            const docs = try allocator.alloc([]const u8, num_workers * larger_queue_depth);
            errdefer allocator.free(docs);
            for (docs) |*ptr| {
                ptr.* = &[_]u8{};
            }
            break :blk docs;
        } else {};

        const pending_documents_free_list: if (USE_SPMC_QUEUE) void else []usize = if (!USE_SPMC_QUEUE) blk: {
            const queue_depth_for_tokenize = 20;
            const queue_depth_for_dp = 20;
            const larger_queue_depth = @max(queue_depth_for_tokenize, queue_depth_for_dp);
            const list = try allocator.alloc(usize, num_workers * larger_queue_depth);
            errdefer allocator.free(list);
            for (list, 0..) |*ptr, i| {
                ptr.* = i;
            }
            break :blk list;
        } else {};

        const n_free_pending_documents: if (USE_SPMC_QUEUE) void else usize = if (!USE_SPMC_QUEUE) blk: {
            const queue_depth_for_tokenize = 20;
            const queue_depth_for_dp = 20;
            const larger_queue_depth = @max(queue_depth_for_tokenize, queue_depth_for_dp);
            break :blk num_workers * larger_queue_depth;
        } else {};

        const queue_depth_for_tokenize: if (USE_SPMC_QUEUE) void else usize = if (!USE_SPMC_QUEUE) 20 else {};
        const queue_depth_for_dp: if (USE_SPMC_QUEUE) void else usize = if (!USE_SPMC_QUEUE) 20 else {};

        // Conditional outstanding jobs tracking
        var n_outstanding_jobs: if (USE_SPMC_QUEUE)
            std.atomic.Value(usize)
        else
            []usize = undefined;

        if (USE_SPMC_QUEUE) {
            n_outstanding_jobs = std.atomic.Value(usize).init(0);
        } else {
            n_outstanding_jobs = try allocator.alloc(usize, num_workers);
            errdefer allocator.free(n_outstanding_jobs);
            @memset(n_outstanding_jobs, 0);
        }

        // work task pool for SPMC with recycling
        const work_task_pool: if (USE_SPMC_QUEUE) []WorkTask else void = if (USE_SPMC_QUEUE)
            try allocator.alloc(WorkTask, 5000)
        else {};

        const work_task_pool_index: if (USE_SPMC_QUEUE) usize else void = if (USE_SPMC_QUEUE) 0 else {};
        const recycled_tasks: if (USE_SPMC_QUEUE) std.atomic.Value(?*WorkTask) else void = if (USE_SPMC_QUEUE) std.atomic.Value(?*WorkTask).init(null) else {};
        const shutdown_flag: if (USE_SPMC_QUEUE) std.atomic.Value(bool) else void = if (USE_SPMC_QUEUE) std.atomic.Value(bool).init(false) else {};

        const automaton = try BakaCorasick.init(allocator);
        errdefer automaton.deinit();

        if (debug) {
            std.debug.print("[ParallelDP] Initialized with {d} workers\n", .{num_workers});
        }

        return ParallelDP{
            .allocator = allocator,
            .vocab_learner = vocab_learner,
            .debug = debug,
            .submission_queues = submission_queues,
            .completion_queues = completion_queues,
            .workers = workers,
            .num_workers = num_workers,
            .sync = sync,
            .idle_event = idle_event,
            .pending_documents = pending_documents,
            .pending_documents_free_list = pending_documents_free_list,
            .n_free_pending_documents = n_free_pending_documents,
            .queue_depth_for_tokenize = queue_depth_for_tokenize,
            .queue_depth_for_dp = queue_depth_for_dp,
            .n_outstanding_jobs = n_outstanding_jobs,
            .work_task_pool = work_task_pool,
            .work_task_pool_index = work_task_pool_index,
            .recycled_tasks = recycled_tasks,
            .shutdown_flag = shutdown_flag,
        };
    }

    pub fn deinit(self: *ParallelDP) void {
        // Signal shutdown for SPMC mode
        if (USE_SPMC_QUEUE) {
            self.shutdown_flag.store(true, .release);
            // Use coordinated shutdown
            self.shutdown();
            std.time.sleep(50 * std.time.ns_per_ms);
        } else {
            // SPSC mode: send shutdown to each worker's queue
            for (0..self.num_workers) |i| {
                const shutdown_msg = SubmissionQueueEntry{
                    .Shutdown = i,
                };
                _ = self.submission_queues[i].push(shutdown_msg);
            }
        }

        // Stop and deinit all workers
        for (0..self.num_workers) |i| {
            self.workers[i].stop();
            self.workers[i].deinit();
            self.completion_queues[i].deinit(self.allocator);
        }

        // Conditional cleanup
        if (USE_SPMC_QUEUE) {
            self.allocator.free(self.work_task_pool);
        } else {
            for (0..self.num_workers) |i| {
                self.submission_queues[i].deinit(self.allocator);
            }
            self.allocator.free(self.submission_queues);
            self.allocator.free(self.n_outstanding_jobs);
            self.allocator.free(self.pending_documents);
            self.allocator.free(self.pending_documents_free_list);
        }

        self.allocator.free(self.completion_queues);
        self.allocator.free(self.workers);
    }

    fn sendMessageToWorker(self: *ParallelDP, worker_id: usize, msg: SubmissionQueueEntry, expect_response: bool) bool {
        if (!USE_SPMC_QUEUE) {
            const result = self.submission_queues[worker_id].push(msg);
            if (result and expect_response) {
                self.n_outstanding_jobs[worker_id] += 1;
            }
            return result;
        } else {
            @panic("sendMessageToWorker not supported in SPMC mode");
        }
    }

    fn scheduleWork(self: *ParallelDP, work_task: *WorkTask) void {
        if (USE_SPMC_QUEUE) {
            work_task.node.next = null;
            const list = work_task.toList();
            self.submission_queues.push(list);
            _ = self.n_outstanding_jobs.fetchAdd(1, .monotonic);

            // Use sophisticated notification system
            self.notify(false);
        } else {
            @panic("scheduleWork should only be called in SPMC mode");
        }
    }

    // Sophisticated notification system with throttling
    fn notify(self: *ParallelDP, is_waking: bool) void {
        if (!USE_SPMC_QUEUE) return;

        // Fast path to check the Sync state to avoid calling into notifySlow().
        // If we're waking, then we need to update the state regardless
        if (!is_waking) {
            const sync = @as(Sync, @bitCast(self.sync.load(.monotonic)));
            if (sync.notified) {
                return;
            }
        }

        return self.notifySlow(is_waking);
    }

    fn notifySlow(self: *ParallelDP, is_waking: bool) void {
        if (!USE_SPMC_QUEUE) return;

        var sync = @as(Sync, @bitCast(self.sync.load(.monotonic)));
        while (sync.state != .shutdown) {
            const can_wake = is_waking or (sync.state == .pending);
            if (is_waking) {
                std.debug.assert(sync.state == .waking);
            }

            var new_sync = sync;
            new_sync.notified = true;
            if (can_wake and sync.idle > 0) { // wake up an idle thread
                new_sync.state = .signaled;
            } else if (can_wake and sync.spawned < self.num_workers) { // can spawn more threads
                new_sync.state = .signaled;
                new_sync.spawned += 1;
            } else if (is_waking) { // no other thread to pass on "waking" status
                new_sync.state = .pending;
            } else if (sync.notified) { // nothing to update
                return;
            }

            // Release barrier synchronizes with Acquire in wait()
            // to ensure pushes to run queues happen before observing a posted notification.
            sync = @as(Sync, @bitCast(self.sync.cmpxchgWeak(
                @as(u32, @bitCast(sync)),
                @as(u32, @bitCast(new_sync)),
                .release,
                .monotonic,
            ) orelse {
                // We signaled to notify an idle thread
                if (can_wake and sync.idle > 0) {
                    return self.idle_event.notify();
                }

                return;
            }));
        }
    }

    fn wait(self: *ParallelDP, is_waking: bool) error{Shutdown}!bool {
        if (!USE_SPMC_QUEUE) return false;

        var is_idle = false;
        var _is_waking = is_waking;
        var sync = @as(Sync, @bitCast(self.sync.load(.monotonic)));

        while (true) {
            if (sync.state == .shutdown) return error.Shutdown;
            if (_is_waking) std.debug.assert(sync.state == .waking);

            // Consume a notification made by notify().
            if (sync.notified) {
                var new_sync = sync;
                new_sync.notified = false;
                if (is_idle)
                    new_sync.idle -= 1;
                if (sync.state == .signaled)
                    new_sync.state = .waking;

                // Acquire barrier synchronizes with notify()
                // to ensure that pushes to run queue are observed after wait() returns.
                sync = @as(Sync, @bitCast(self.sync.cmpxchgWeak(
                    @as(u32, @bitCast(sync)),
                    @as(u32, @bitCast(new_sync)),
                    .acquire,
                    .monotonic,
                ) orelse {
                    return _is_waking or (sync.state == .signaled);
                }));

                // No notification to consume.
                // Mark this thread as idle before sleeping on the idle_event.
            } else if (!is_idle) {
                var new_sync = sync;
                new_sync.idle += 1;
                if (_is_waking)
                    new_sync.state = .pending;

                sync = @as(Sync, @bitCast(self.sync.cmpxchgWeak(
                    @as(u32, @bitCast(sync)),
                    @as(u32, @bitCast(new_sync)),
                    .monotonic,
                    .monotonic,
                ) orelse {
                    _is_waking = false;
                    is_idle = true;
                    continue;
                }));

                // Wait for a signal by either notify() or shutdown() without wasting cpu cycles.
            } else {
                self.idle_event.wait();
                sync = @as(Sync, @bitCast(self.sync.load(.monotonic)));
            }
        }
    }

    /// Marks the thread pool as shutdown
    fn shutdown(self: *ParallelDP) void {
        if (!USE_SPMC_QUEUE) return;

        var sync = @as(Sync, @bitCast(self.sync.load(.monotonic)));
        while (sync.state != .shutdown) {
            var new_sync = sync;
            new_sync.notified = true;
            new_sync.state = .shutdown;
            new_sync.idle = 0;

            // Full barrier to synchronize with both wait() and notify()
            sync = @as(Sync, @bitCast(self.sync.cmpxchgWeak(
                @as(u32, @bitCast(sync)),
                @as(u32, @bitCast(new_sync)),
                .acq_rel,
                .monotonic,
            ) orelse {
                // Wake up any threads sleeping on the idle_event.
                if (sync.idle > 0) self.idle_event.shutdown();
                return;
            }));
        }
    }

    fn allocateWorkTask(self: *ParallelDP) !*WorkTask {
        if (!USE_SPMC_QUEUE) {
            @panic("allocateWorkTask() should only be called in SPMC mode");
        }

        var recycled_head = self.recycled_tasks.load(.acquire);
        while (recycled_head) |task| {
            const next = task.recycle_next;
            if (self.recycled_tasks.cmpxchgWeak(recycled_head, next, .release, .acquire)) |updated_head| {
                recycled_head = updated_head;
            } else {
                task.node = Node{};
                task.recycle_next = null;
                return task;
            }
        }

        // No recycled tasks available, allocate from pool
        if (self.work_task_pool_index >= self.work_task_pool.len) {
            return error.WorkTaskPoolExhausted;
        }

        const work_task = &self.work_task_pool[self.work_task_pool_index];
        self.work_task_pool_index += 1;
        work_task.node = Node{};
        work_task.recycle_next = null;
        return work_task;
    }

    fn recycleWorkTask(self: *ParallelDP, task: *WorkTask) void {
        if (!USE_SPMC_QUEUE) {
            @panic("recycleWorkTask() should only be called in SPMC mode");
        }

        // Lock-free push to recycled stack
        var current_head = self.recycled_tasks.load(.acquire);
        while (true) {
            task.recycle_next = current_head;
            if (self.recycled_tasks.cmpxchgWeak(current_head, task, .release, .acquire)) |updated_head| {
                current_head = updated_head;
            } else {
                break;
            }
        }
    }

    fn resetWorkTaskPool(self: *ParallelDP) void {
        if (USE_SPMC_QUEUE) {
            self.work_task_pool_index = 0;
            self.recycled_tasks.store(null, .release);
        }
    }

    fn canScheduleWork(self: *ParallelDP) bool {
        if (USE_SPMC_QUEUE) {
            return !self.shutdown_flag.load(.monotonic) and
                self.n_outstanding_jobs.load(.monotonic) < (self.num_workers * 50);
        } else {
            return false;
        }
    }

    pub fn initWorkers(
        self: *ParallelDP,
        candidates: []SampleStats,
        automaton: *BakaCorasick,
    ) !void {
        if (self.debug) {
            std.debug.print("[ParallelDP] Initializing {d} workers with {d} candidates\n", .{ self.num_workers, candidates.len });
        }

        if (!self.started_workers) {
            for (0..self.num_workers) |i| {
                const submission_queue = if (USE_SPMC_QUEUE)
                    &self.submission_queues
                else
                    &self.submission_queues[i];

                self.workers[i] = try Worker.init(self.allocator, i, self.vocab_learner, automaton, candidates.len, submission_queue, &self.completion_queues[i], self.debug);

                // Set up SPMC-specific worker fields
                if (USE_SPMC_QUEUE) {
                    self.workers[i].workers = self.workers;
                    self.workers[i].shutdown_flag = &self.shutdown_flag;
                    self.workers[i].coordinator = self;
                }
            }

            // Start all workers
            for (0..self.num_workers) |i| {
                try self.workers[i].start();
            }

            // Initialize sync state with spawned thread count
            if (USE_SPMC_QUEUE) {
                var sync = @as(Sync, @bitCast(self.sync.load(.monotonic)));
                sync.spawned = @intCast(self.num_workers);
                self.sync.store(@bitCast(sync), .release);
            }

            self.started_workers = true;
        }

        if (USE_SPMC_QUEUE) {
            for (0..self.num_workers) |i| {
                self.workers[i].current_n_candidates = candidates.len;
                for (self.workers[i].candidate_stats[0..candidates.len], 0..) |*stats, idx| {
                    stats.sampled_occurrences = 0;
                    stats.sampled_savings = 0;
                    stats.token_id = candidates[idx].token_id;
                }
                self.workers[i].candidate_automaton = automaton;
            }
        } else {
            for (0..self.num_workers) |i| {
                const msg = SubmissionQueueEntry{
                    .Reset = .{
                        .candidates = candidates,
                        .automaton = automaton,
                    },
                };
                _ = self.sendMessageToWorker(i, msg, false);
            }
        }
    }

    pub fn initWorkersForCorpusTokenCount(self: *ParallelDP) !void {
        if (self.debug) {
            std.debug.print("[ParallelDP] Initializing {d} workers for corpus token count\n", .{self.num_workers});
        }

        if (!self.started_workers) {
            return error.FixMePlz;
        }

        if (USE_SPMC_QUEUE) {
            self.n_outstanding_jobs.store(0, .release);
            self.shutdown_flag.store(false, .release);
        }
    }

    fn addToPendingDocuments(self: *ParallelDP, document: []const u8) usize {
        if (!USE_SPMC_QUEUE) {
            self.n_free_pending_documents -= 1;
            const index = self.pending_documents_free_list[self.n_free_pending_documents];
            self.pending_documents[index] = document;
            return index;
        } else {
            @panic("addToPendingDocuments should not be called in SPMC mode");
        }
    }

    fn removeFromPendingDocuments(self: *ParallelDP, index: usize, comptime DataLoaderType: type) void {
        if (!USE_SPMC_QUEUE) {
            const document = self.pending_documents[index];

            if (DataLoaderType.NEEDS_DEALLOCATION) {
                self.allocator.free(document);
            }
            self.pending_documents_free_list[self.n_free_pending_documents] = index;
            self.pending_documents[index] = &[_]u8{};
            self.n_free_pending_documents += 1;
        } else {
            if (DataLoaderType.NEEDS_DEALLOCATION) {
                // Document will be freed when work task is reset
            }
        }
    }

    pub fn processDocuments(
        self: *ParallelDP,
        loader: anytype,
        sample_size: usize,
        candidates: []SampleStats,
        candidate_automaton: *BakaCorasick,
    ) !void {
        const DataLoaderType = @TypeOf(loader.*);
        const start_time = time.nanoTimestamp();

        if (USE_SPMC_QUEUE) {
            self.resetWorkTaskPool();
        }

        try self.initWorkers(candidates, candidate_automaton);

        var documents_processed: usize = 0;
        var documents_submitted: usize = 0;
        var running = true;

        while (running) {
            var did_anything = false;

            if (USE_SPMC_QUEUE) {
                for (0..self.num_workers) |i| {
                    if (self.completion_queues[i].pop()) |msg| {
                        _ = self.n_outstanding_jobs.fetchSub(1, .monotonic);
                        did_anything = true;

                        switch (msg) {
                            .DocumentProcessed => |doc_processed_msg| {
                                // Recycle the work task for reuse
                                self.recycleWorkTask(doc_processed_msg.work_task);
                                documents_processed += 1;
                            },
                            .TokenCount => {
                                // Ignore leftover messages from previous operations
                            },
                            .Error => |error_data| {
                                std.debug.print("[Coordinator] Worker {d} error: {s}\n", .{ i, error_data.message });
                                return error.WorkerError;
                            },
                        }
                    }
                }

                // Submit work when capacity allows
                if (documents_submitted < sample_size and self.canScheduleWork()) {
                    const doc = try loader.nextDocumentStringLoop();
                    did_anything = true;

                    const work_task = try self.allocateWorkTask();
                    work_task.* = WorkTask{
                        .data = .{
                            .ProcessDocument = ProcessDocumentMessage{
                                .id = documents_submitted, // Use as document ID for SPMC
                                .document = doc,
                                .automaton = candidate_automaton,
                            },
                        },
                    };

                    self.scheduleWork(work_task);
                    documents_submitted += 1;

                    if (DataLoaderType.NEEDS_DEALLOCATION) {
                        // Document is now owned by work task, will be freed during task reset
                    }
                }
            } else {
                for (0..self.num_workers) |i| {
                    if (self.completion_queues[i].pop()) |msg| {
                        self.n_outstanding_jobs[i] -= 1;
                        did_anything = true;
                        switch (msg) {
                            .DocumentProcessed => |doc_processed_msg| {
                                self.removeFromPendingDocuments(doc_processed_msg.id, DataLoaderType);
                                documents_processed += 1;
                            },
                            .TokenCount => {
                                // Ignore leftover messages
                            },
                            .Error => |error_data| {
                                std.debug.print("[Coordinator] Worker {d} error: {s}\n", .{ i, error_data.message });
                                @panic("oh no!2");
                            },
                        }
                    }

                    if (documents_submitted < sample_size and self.n_outstanding_jobs[i] < self.queue_depth_for_dp) {
                        const doc = try loader.nextDocumentStringLoop();
                        did_anything = true;

                        const doc_idx = self.addToPendingDocuments(doc);

                        const msg = SubmissionQueueEntry{
                            .ProcessDocument = ProcessDocumentMessage{
                                .id = doc_idx,
                                .document = doc,
                            },
                        };

                        _ = self.sendMessageToWorker(i, msg, true);
                        documents_submitted += 1;
                    }
                }
            }

            if (documents_processed == sample_size) {
                // Copy the stats from all workers
                for (self.workers) |worker| {
                    for (worker.candidate_stats, 0..) |stats, idx| {
                        candidates[idx].sampled_occurrences += stats.sampled_occurrences;
                        candidates[idx].sampled_savings += stats.sampled_savings;
                    }
                }
                running = false;
            }

            if (!did_anything) {
                std.time.sleep(1);
            }
        }

        if (self.debug) {
            std.debug.print("[ParallelDP] Initially submitted {d} documents to workers\n", .{documents_submitted});
        }

        const elapsed = time.nanoTimestamp() - start_time;
        if (self.debug) {
            std.debug.print("[ParallelDP] Processed {d} documents in {d:.2}ms\n", .{ documents_processed, @as(f64, @floatFromInt(elapsed)) / time.ns_per_ms });
        }
    }

    pub fn updateMaxSavingsByTokenizingCandidates(
        self: *ParallelDP,
        candidates: []u32,
    ) !void {
        const start_time = time.nanoTimestamp();

        if (USE_SPMC_QUEUE) {
            self.resetWorkTaskPool();
        }

        try self.initWorkersForCorpusTokenCount();

        var documents_processed: usize = 0;
        var documents_submitted: usize = 0;

        while (true) {
            var did_anything = false;

            for (0..self.num_workers) |i| {
                if (self.completion_queues[i].pop()) |msg| {
                    if (USE_SPMC_QUEUE) {
                        _ = self.n_outstanding_jobs.fetchSub(1, .monotonic);
                    } else {
                        self.n_outstanding_jobs[i] -= 1;
                    }
                    did_anything = true;
                    switch (msg) {
                        .TokenCount => |token_count_msg| {
                            const token_id: u32 = @intCast(token_count_msg.id);
                            const token_count: u16 = @intCast(token_count_msg.token_count);
                            const stats: *TokenStats = &self.vocab_learner.candidate_stats[token_id];
                            const old_token_count = stats.len_in_tokens;
                            if (token_count < old_token_count) {
                                stats.len_in_tokens = token_count;
                                const old_est_savings = stats.est_total_savings;
                                const occurrence_count = stats.n_nonoverlapping_occurrences;
                                const new_savings: f64 = @floatFromInt(occurrence_count * (token_count - 1));
                                stats.est_total_savings = @min(old_est_savings, new_savings);
                            }
                            // Recycle the work task for reuse in SPMC mode
                            if (USE_SPMC_QUEUE) {
                                self.recycleWorkTask(token_count_msg.work_task);
                            }
                            documents_processed += 1;
                        },
                        .DocumentProcessed => {
                            // Ignore leftover messages
                        },
                        .Error => |error_data| {
                            std.debug.print("[Coordinator] Worker {d} error: {s}\n", .{ i, error_data.message });
                            if (USE_SPMC_QUEUE) {
                                return error.WorkerError;
                            } else {
                                @panic("oh no!4");
                            }
                        },
                    }
                }

                const can_send = if (USE_SPMC_QUEUE)
                    self.canScheduleWork()
                else
                    self.n_outstanding_jobs[i] < self.queue_depth_for_tokenize;

                if (can_send and documents_submitted < candidates.len) {
                    const doc = self.vocab_learner.getTokenStr(candidates[documents_submitted]);
                    did_anything = true;

                    if (USE_SPMC_QUEUE) {
                        const work_task = try self.allocateWorkTask();
                        work_task.* = WorkTask{
                            .data = .{
                                .CountTokensInDocument = CountTokensInDocumentMessage{
                                    .id = documents_submitted,
                                    .document = doc,
                                },
                            },
                        };
                        self.scheduleWork(work_task);
                    } else {
                        const msg = SubmissionQueueEntry{
                            .CountTokensInDocument = CountTokensInDocumentMessage{
                                .id = documents_submitted,
                                .document = doc,
                            },
                        };
                        _ = self.sendMessageToWorker(i, msg, true);
                    }
                    documents_submitted += 1;
                }
            }

            if (documents_processed == candidates.len) {
                if (self.debug) {
                    const elapsed = time.nanoTimestamp() - start_time;
                    std.debug.print("[ParallelDP] Tokenized {d} candidates in {d:.2}ms\n", .{ documents_processed, @as(f64, @floatFromInt(elapsed)) / time.ns_per_ms });
                }
                return;
            }

            if (!did_anything) {
                std.time.sleep(1);
            }
        }
    }

    pub fn getCorpusTokenCount(self: *ParallelDP, loader: anytype) !u64 {
        const DataLoaderType = @TypeOf(loader.*);
        const start_time = time.nanoTimestamp();

        if (USE_SPMC_QUEUE) {
            self.resetWorkTaskPool();
            self.shutdown_flag.store(false, .release);
        }

        try self.initWorkersForCorpusTokenCount();

        var documents_processed: usize = 0;
        var documents_submitted: usize = 0;

        try loader.rewind();
        var maybe_doc = try loader.nextDocumentString();
        var ret: u64 = 0;

        while (true) {
            var did_anything = false;

            for (0..self.num_workers) |i| {
                if (self.completion_queues[i].pop()) |msg| {
                    if (USE_SPMC_QUEUE) {
                        _ = self.n_outstanding_jobs.fetchSub(1, .monotonic);
                    } else {
                        self.n_outstanding_jobs[i] -= 1;
                    }
                    did_anything = true;
                    switch (msg) {
                        .TokenCount => |token_count_msg| {
                            if (!USE_SPMC_QUEUE) {
                                self.removeFromPendingDocuments(token_count_msg.id, DataLoaderType);
                            } else {
                                // Recycle the work task for reuse
                                self.recycleWorkTask(token_count_msg.work_task);
                            }
                            ret += token_count_msg.token_count;
                            documents_processed += 1;
                        },
                        .DocumentProcessed => {
                            // Ignore leftover messages
                        },
                        .Error => |error_data| {
                            std.debug.print("[Coordinator] Worker {d} error: {s}\n", .{ i, error_data.message });
                            if (USE_SPMC_QUEUE) {
                                return error.WorkerError;
                            } else {
                                @panic("oh no!6");
                            }
                        },
                    }
                }

                if (USE_SPMC_QUEUE) {
                    if (i == 0 and self.canScheduleWork()) {
                        if (maybe_doc) |doc| {
                            did_anything = true;

                            const work_task = try self.allocateWorkTask();
                            work_task.* = WorkTask{
                                .data = .{
                                    .CountTokensInDocument = CountTokensInDocumentMessage{
                                        .id = documents_submitted,
                                        .document = doc,
                                    },
                                },
                            };
                            self.scheduleWork(work_task);
                            documents_submitted += 1;
                            maybe_doc = try loader.nextDocumentString();

                            if (DataLoaderType.NEEDS_DEALLOCATION) {
                                // Document is now owned by work task
                            }
                        }
                    }
                } else {
                    if (self.n_outstanding_jobs[i] < self.queue_depth_for_tokenize) {
                        if (maybe_doc) |doc| {
                            did_anything = true;

                            const doc_idx = self.addToPendingDocuments(doc);

                            const msg = SubmissionQueueEntry{
                                .CountTokensInDocument = CountTokensInDocumentMessage{
                                    .id = doc_idx,
                                    .document = doc,
                                },
                            };
                            _ = self.sendMessageToWorker(i, msg, true);
                            documents_submitted += 1;
                            maybe_doc = try loader.nextDocumentString();
                        }
                    }
                }
            }

            if (maybe_doc == null and documents_processed == documents_submitted) {
                if (self.debug) {
                    const elapsed = time.nanoTimestamp() - start_time;
                    std.debug.print("[ParallelDP] Processed {d} documents in {d:.2}ms\n", .{ documents_processed, @as(f64, @floatFromInt(elapsed)) / time.ns_per_ms });
                }
                return ret;
            }

            if (!did_anything) {
                std.time.sleep(1);
            }
        }
    }
};
