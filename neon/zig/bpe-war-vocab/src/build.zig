const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Create executable for tokenset_combiner
    const tokenset_combiner = b.addExecutable(.{
        .name = "tokenset_combiner",
        .root_source_file = b.path("tokenset_combiner.zig"),
        .target = target,
        .optimize = optimize,
    });
    b.installArtifact(tokenset_combiner);

    // Add run step for tokenset_combiner
    const run_tokenset_combiner = b.addRunArtifact(tokenset_combiner);
    if (b.args) |args| {
        run_tokenset_combiner.addArgs(args);
    }
    const run_tokenset_combiner_step = b.step("run-tokenset-combiner", "Run the tokenset combiner");
    run_tokenset_combiner_step.dependOn(&run_tokenset_combiner.step);

    // Create executable for tokenset_filter
    const tokenset_filter = b.addExecutable(.{
        .name = "tokenset_filter",
        .root_source_file = b.path("tokenset_filter.zig"),
        .target = target,
        .optimize = optimize,
    });
    b.installArtifact(tokenset_filter);

    // Add run step for tokenset_filter
    const run_tokenset_filter = b.addRunArtifact(tokenset_filter);
    if (b.args) |args| {
        run_tokenset_filter.addArgs(args);
    }
    const run_tokenset_filter_step = b.step("run-tokenset-filter", "Run the tokenset filter");
    run_tokenset_filter_step.dependOn(&run_tokenset_filter.step);

    // Create executable for vocab_reader
    const vocab_reader = b.addExecutable(.{
        .name = "vocab_reader",
        .root_source_file = b.path("vocab_reader.zig"),
        .target = target,
        .optimize = optimize,
    });
    b.installArtifact(vocab_reader);

    // Create executable for vocab_learner
    const vocab_learner = b.addExecutable(.{
        .name = "vocab_learner",
        .root_source_file = b.path("vocab_learner.zig"),
        .target = target,
        .optimize = optimize,
    });
    b.installArtifact(vocab_learner);
}
