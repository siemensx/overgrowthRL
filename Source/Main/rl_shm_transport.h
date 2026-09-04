// Stage 5.4: POSIX shared-memory transport between this engine process and a
// Python-side driver, one segment per worker (matches the plan's per-worker,
// single-producer/single-consumer design -- no cross-worker sharing, no
// fan-in/fan-out).
//
// Protocol (lock-step request/response, matching real Gym env.step()
// semantics): each real step, the engine extracts the *current* observation
// (the state resulting from last step's action + physics, exactly as a Gym
// env's step() return value would), publishes it, then blocks until Python
// has read it and published the next action; that action is staged into
// RLAction so the immediately-following RLAction::Apply() call (still the
// only thing that ever touches Input::PlayerInput) picks it up. This module
// never talks to the input system directly -- RLAction remains the single
// injection point, matching AGENTS.md's "one deliberate access point" for
// script-facing state.
//
// Synchronization: two NAMED POSIX semaphores (sem_open), not the mutex the
// plan's own phrasing warns against, and deliberately not unnamed
// process-shared semaphores (sem_init(..., pshared=1, ...) is unimplemented
// on Darwin -- returns ENOSYS -- so named semaphores are not a style choice
// here, they are the only POSIX option this platform supports for
// cross-process semaphores). The payload buffers themselves carry no lock or
// atomic at all: POSIX guarantees sem_post/sem_wait form a release/acquire
// pair, so everything written before a sem_post is visible after the
// matching sem_wait with no additional synchronization needed -- the
// handshake IS the synchronization primitive, which is a stronger read of
// "atomics not mutexes" than sprinkling std::atomic on the buffer would be.
//
// Known limitation, not yet solved: Darwin does not implement
// sem_timedwait, so Step() blocks indefinitely if the Python side stops
// responding. This mirrors real env.step() coupling (the env genuinely
// cannot proceed without an action) rather than being an oversight, but a
// watchdog/timeout is worth adding before unattended long-running training,
// not before.
#pragma once

#include <string>

#include "rl_observation.h"

class Engine;

namespace RLShmTransport {

// `name` must start with '/' and stay well under Darwin's ~31-byte POSIX
// IPC name limit (this module derives two semaphore names from it by
// appending single characters, so keep `name` itself to ~24 bytes or less).
// Creates (or re-attaches to) the shm segment and both semaphores.
//
// `act_period` (Stage 6, OGRL-20260816-021 Sec 1.3(a)/2.2(a)): the shm
// request/response handshake -- observation extract, publish, block for an
// action -- only runs once every `act_period` physics ticks (default 1,
// matching the original 120Hz-decisions behavior exactly). On the other
// act_period-1 ticks, Step() returns immediately without touching the shm
// buffers or semaphores; RLAction::Apply() still runs every tick regardless
// (unchanged call site in engine.cpp) and re-applies whatever action was
// most recently staged, which Input::PlayerInput's count-negation semantics
// read as a HELD key rather than a new press -- exactly the behavior a
// human holding a controller button produces. This is not a new mechanism,
// it is relying on one that already existed for the right reason. See
// rl_shm_transport.cpp's Step() for the exact tick accounting.
bool Configure(const std::string& name, int controller_id, const RLObservation::ObservationConfig& config, int act_period = 1);
bool Enabled();

// The controller selected by --rl-action-controller-id, or -1 when the
// transport is inactive. Avatar setup uses this to keep external slots out
// of native keyboard/gamepad polling.
int ControllerId();

// Called once per real step, in place of where RLAction::Apply() used to be
// invoked directly -- Apply() is still called immediately after this, now
// picking up whatever action Step() staged. No-op (returns true) if
// Configure() was never called. Returns false if the Python side set
// shutdown_requested, so the caller can end the run gracefully rather than
// treating it as an error.
bool Step(Engine* engine);

// Draw the match-only diagnostics immediately before the render pass. The
// render-phase lifetime prevents a physics/render cadence mismatch from
// making the overlay flicker.
void DrawMatchOverlay(Engine* engine);

// Consume the match-only F8 edge before the action transport blocks waiting
// for the next policy action. DrawMatchOverlay calls this too for render-only
// frames, but the update call is the normal path for live gameplay.
void UpdateMatchOverlayInput(Engine* engine);

// Match-only presentation state shared with the AngelScript combat-debug
// helpers. Training and ordinary game sessions always report true.
bool MatchOverlayVisible();

// Unmaps and unlinks the shm segment and both semaphores. Safe to call even
// if Configure() was never called or already shut down.
void Shutdown();

}  // namespace RLShmTransport
