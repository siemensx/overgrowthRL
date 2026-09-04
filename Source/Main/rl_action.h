// Stage 5.3 (research-log OGRL-20260816-005): action ingestion.
//
// Injects a synthetic legal-input state directly into Input::PlayerInput's
// key_down map for one controller_id, replicating the exact update pattern
// Input::ProcessController uses for real hardware (Source/UserInput/input.cpp:554-628)
// so every existing script call site (GetInputDown/GetInputPressed/GetMoveXAxis/
// GetMoveYAxis in Data/Scripts/aschar.as and playercontrol.as) picks it up with
// ZERO script changes -- this is deliberately not a new script API.
//
// Solves a real design problem discovered while building this: playercontrol.as's
// GetTargetVelocity() converts the raw move axes into world space using
// camera.GetFlatFacing() -- i.e. movement is camera-relative by construction,
// but AGENTS.md's action contract requires body-relative movement ("the agent
// does not operate a camera"). Rather than fork the script's movement math
// (which would put RL-specific logic inside the compatibility boundary), this
// keeps the camera in sync with the controlled character's own facing every
// step, so camera-relative and body-relative are the same thing by construction
// and the unmodified script logic is correct for a camera-less agent for free.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

class Engine;

namespace RLAction {

// controller_id must match the controlled character's MovementObject::controller_id.
void Configure(bool enabled, int controller_id);
bool Enabled();

// True for the external controller used by a rendered human-duel match.
// Controller 0 remains the native keyboard/mouse/gamepad player.
bool IsExternalController(int controller_id);

// Canonical episode boundary for the shm transport. Clears every held axis,
// button and action-history sample before a freshly reset world is advanced.
// This is intentionally separate from Configure(): physical engine workers
// remain alive across episodes, while their action state must not.
void ResetForEpisode();

// Continuous body-relative movement axes, each in [-1, 1] (matches
// AS_GetMoveXAxis/AS_GetMoveYAxis's existing [-1,1] range from key depth
// differencing). x = right(+)/left(-), y = forward(+)/back(-).
void SetMoveAxes(float move_x, float move_y);

// Discrete legal buttons from the input audit (Data/Scripts/playercontrol.as):
// "jump", "crouch", "attack", "grab", "drop", "walk".
void SetButton(const std::string& name, bool held);

// Returns the exact legal action state applied by the most recent
// RLAction::Apply() call, in the transport order
// [move_x, move_y, jump, crouch, attack, grab, drop, walk].  The native
// replay recorder samples this after input injection, so a trace contains
// what the engine consumed rather than only what Python intended.
void GetCurrentAction(std::vector<float>* out);

// Native equivalence replay supplies the already-applied action at the
// verified physics-tick boundary. This prevents startup/load-loop ticks from
// shifting a CSV cursor before the first archived state tick.
void SetNativeReplayMode(bool enabled);

// Called once per real (non-loading-screen) step, immediately after
// Input::ProcessControllers -- syncs the phantom camera's facing to the
// controlled character's own facing, then applies the current action state
// into that controller's PlayerInput::key_down map.
void Apply(Engine* engine);

// Test/validation harness (research-log OGRL-20260816-006): loads a sparse,
// step-indexed action script -- lines "step,move_x,move_y,button=0|1,...",
// each step's settings held until the next explicit line -- and applies the
// entry for the current step automatically from Apply(). This is what proved
// out timing-sensitive combos (e.g. an aerial attack, which the script gates
// on GetInputPressed while airborne, not GetInputDown) without needing the
// full Python-side action transport (Stage 5.4) to exist yet.
bool LoadScript(const std::string& path);
void SetScriptStep(uint64_t step);

// OGRL-20260817-031: a loaded script's "step" column is a DECISION index
// (one row per act_period physics ticks -- watch.py/tape.py record one row
// per decision, not per tick), but Apply() used to advance g_step_counter
// once per physics TICK unconditionally. With the training-standard
// act_period=4 (30Hz decisions), that meant a replayed script consumed 4
// recorded decisions' worth of "step" value for every 1 tick of real
// held-input time -- i.e. every replay played 4x too fast. Call this once,
// after LoadScript(), with the SAME act_period the script was recorded at
// (ticks_per_decision -- 1 preserves the old every-tick behavior for a
// script that genuinely was recorded per-tick, e.g. the original Stage 5.3
// timing-combo scripts). Apply() then only advances the decision index once
// every `ticks_per_decision` calls, matching how RLShmTransport already
// holds a live decision across the ticks between requests.
void SetScriptPeriod(int ticks_per_decision);

// True once the loaded script's last recorded decision has been passed --
// i.e. Apply() is now holding the final entry indefinitely rather than
// following real recorded data. Always false with no script loaded.
bool ScriptFinished();

// OGRL-20260817-033 / revised OGRL-20260819-039: how long to linger after the
// script's last recorded decision before the caller quits. Call once after
// LoadScript(). 0 (default) preserves the original instant-quit behavior.
//
// The FIRST SettleTicks() of that window keeps SIMULATING, because the
// recording's final action is usually the decisive one and its consequence
// lands AFTER the last scripted tick -- measured on tape 1905_w0_e58 the
// knockout registers at tick 234 while the script ends at tick 232. The
// REMAINDER is spent FROZEN (Engine sets paused=true), not simulating.
//
// That distinction is the entire point of this revision. The original version
// simulated for the whole hold, so a ~2s recording was followed by ~5s of the
// arena's own round logic running on stale held input -- the agent standing
// still, being killed, and a fresh round starting. Viewers reasonably read
// that footage as "the replay", and it contradicted the tape's own outcome.
void SetScriptHoldSeconds(float seconds);

// Ticks since the script's last recorded decision was passed; -1 if the
// script has not finished or none is loaded. Lets the caller stage
// settle -> freeze -> quit.
int64_t TicksSinceScriptFinished();

// Ticks to keep SIMULATING after the script ends, before freezing. Constant,
// exposed so Engine and this module cannot disagree about the boundary.
int64_t SettleTicks();

// True once the whole hold window has elapsed and the caller should quit,
// using the headless-safe quitting_ = true pattern RLShmTransport's own
// graceful shutdown already uses -- NOT Input::RequestQuit() directly, which
// can hang forever in headless/benchmark mode waiting on a save-changes
// dialog that needs rendering to resolve (see rl_shm_transport.cpp).
bool ReadyToQuit();

// Stage 5.1: controller_id passed to the most recent Configure() call, -1 if
// none. Lets other modules (RLObservation) check whether a given
// MovementObject::controller_id is the one this module is driving, before
// pulling action history from it -- a character not receiving RL-injected
// input has no legal action trace to report, and should stay zero-filled
// rather than silently show this module's unrelated state.
int ControllerId();

// The last `steps` action snapshots applied by Apply(), most recent first,
// each flattened to 6 floats (move_x, move_y, jump, crouch, attack, grab) --
// matches RLObservation's action-history block layout exactly (one flag per
// legal discrete action that feeds the observation; "drop"/"walk" are legal
// inputs but not part of that layout). `out` is resized to steps * 6 and
// zero-padded for steps before the first Apply() call.
void GetRecentHistory(int steps, std::vector<float>* out);

}  // namespace RLAction
