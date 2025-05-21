const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const tokenset_combiner = b.addExecutable(.{
        .name = "tokenset_combiner",
        .root_source_file = b.path("src/tokenset_combiner.zig"),
        .target = target,
        .optimize = optimize,
    });
    b.installArtifact(tokenset_combiner);

    const run_tokenset_combiner = b.addRunArtifact(tokenset_combiner);
    if (b.args) |args| {
        run_tokenset_combiner.addArgs(args);
    }
    const run_tokenset_combiner_step = b.step("run-tokenset-combiner", "Run the tokenset combiner");
    run_tokenset_combiner_step.dependOn(&run_tokenset_combiner.step);

    const tokenset_filter = b.addExecutable(.{
        .name = "tokenset_filter",
        .root_source_file = b.path("src/tokenset_filter.zig"),
        .target = target,
        .optimize = optimize,
    });
    b.installArtifact(tokenset_filter);

    const run_tokenset_filter = b.addRunArtifact(tokenset_filter);
    if (b.args) |args| {
        run_tokenset_filter.addArgs(args);
    }
    const run_tokenset_filter_step = b.step("run-tokenset-filter", "Run the tokenset filter");
    run_tokenset_filter_step.dependOn(&run_tokenset_filter.step);

    const vocab_reader = b.addExecutable(.{
        .name = "vocab_reader",
        .root_source_file = b.path("src/vocab_reader.zig"),
        .target = target,
        .optimize = optimize,
    });
    b.installArtifact(vocab_reader);

    const vocab_learner = b.addExecutable(.{
        .name = "vocab_learner",
        .root_source_file = b.path("src/vocab_learner.zig"),
        .target = target,
        .optimize = optimize,
    });
    b.installArtifact(vocab_learner);
}
