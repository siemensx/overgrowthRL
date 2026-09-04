// Stage 2/3 determinism fix (research-log OGRL-20260815-038): posix_spawn the
// target binary with ASLR disabled.
//
// Root cause, confirmed empirically: identical binary, identical --benchmark-seed,
// identical inputs (RNG draw counts and camera position both verified bit-identical
// across runs) still produced diverging combat trajectories -- 10 back-to-back
// same-seed runs launched normally split into a ~50/50 bimodal pattern, matching
// within a cluster and diverging at one fixed step between clusters. 10/10 runs
// launched through THIS binary (ASLR disabled) were bit-identical. The remaining
// variable between runs was address-space layout itself, which macOS randomizes
// per-process by default (the engine links as a PIE binary) -- almost certainly
// interacting with alignment-sensitive NEON/auto-vectorized floating-point code
// somewhere in the animation/physics pipeline, producing a handful of distinct-
// but-internally-consistent numerical outcomes depending on runtime address
// alignment, not a logic bug.
//
// Usage: noaslr_launcher <binary> [args...] -- replaces THIS process's own
// image with the given binary (POSIX_SPAWN_SETEXEC: same PID, execve-style,
// not a fork), with ASLR disabled. Using SETEXEC (rather than a plain
// posix_spawn, which creates a *new* child PID) matters in practice: every RL
// harness script samples RSS/CPU via `ps -p <pid>` on the PID it launched --
// with a plain fork-style spawn that PID stays the tiny launcher forever
// (observed: reported peak_rss_mib collapsed to ~1.3MB instead of the real
// engine's ~500MB the first time this shipped, see research-log OGRL-20260815-038).
// SETEXEC makes this launcher's PID become the engine's PID, so every
// existing ps-based sampler keeps working unmodified.
#include <spawn.h>
#include <stdio.h>
#include <unistd.h>

#ifndef _POSIX_SPAWN_DISABLE_ASLR
#define _POSIX_SPAWN_DISABLE_ASLR 0x0100
#endif
#ifndef POSIX_SPAWN_SETEXEC
#define POSIX_SPAWN_SETEXEC 0x0040
#endif

extern char **environ;

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <binary> [args...]\n", argv[0]);
        return 2;
    }
    posix_spawnattr_t attr;
    posix_spawnattr_init(&attr);
    posix_spawnattr_setflags(&attr, _POSIX_SPAWN_DISABLE_ASLR | POSIX_SPAWN_SETEXEC);

    pid_t pid;  // unused with SETEXEC (this process becomes argv[1] in place)
    int rc = posix_spawn(&pid, argv[1], NULL, &attr, &argv[1], environ);
    // Only reachable if posix_spawn/exec failed -- on success this process
    // image is gone and control never returns here.
    posix_spawnattr_destroy(&attr);
    fprintf(stderr, "posix_spawn (SETEXEC) failed: %d\n", rc);
    return 1;
}
