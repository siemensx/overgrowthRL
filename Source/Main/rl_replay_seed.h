// OGRL-20260817-030: deterministic tape replay for RLAction::LoadScript's
// scripted-input replay mode (Tools/rl/replay_ghost.py).
//
// Problem this fixes: replay_ghost.py deliberately skips the shm transport,
// which is the ONLY place a seed/curriculum-difficulty gets applied
// (Engine::ResetRLTrainingScenario / SoftResetRLTrainingScenario, both
// gated on RLBenchmark::Enabled() || RLShmTransport::Enabled()). A replay
// session has neither enabled, so it was reproducing the recorded button
// presses against a freshly-rolled, unseeded opponent -- not the original
// fight. See replay_ghost.py's module docstring for the full user-facing
// explanation.
//
// Fix: reuse the exact same reseed-then-respawn recipe
// SoftResetRLTrainingScenario already uses mid-training (seed RNG streams +
// game timer, then Level::Message("set_rl_*") + Level::Message("post_reset")
// -- see rl_shm_transport.cpp's SendScenarioMessages), just applied ONCE,
// the moment the level finishes its own natural (unseeded) initial load,
// instead of on every shm-driven reset. This deliberately does NOT hook
// before the initial level load (Engine::Initialize()'s QueueState call) --
// it doesn't need to: every training episode after the very first "pseudo-
// reset" already uses exactly this same after-the-fact reseed+respawn
// pattern (see env.py's _used_initial_observation comment), and it's the
// mechanism soft reset's own distribution-equivalence validation already
// covers, so this introduces no new unvalidated code path.
#pragma once

class Engine;

namespace RLReplaySeed {

// difficulty/weapons < 0 and opponents/species < 0 mean "leave that
// curriculum axis alone, don't send its set_rl_* message" -- matches
// arena_level_1v1_unarmed.as's own rl_difficulty=-1.0f "unset" convention,
// so a caller can reseed without forcing every axis to a specific value.
void Configure(unsigned int seed, float difficulty, int opponents, float weapons, int species,
               int reset_mode = 0, int controlled_character_id = -1);
bool Enabled();

// Call once per real engine tick, BEFORE RLAction::Apply() -- so the first
// scripted action in the loaded script lands on the POST-reseed/respawn
// state, not the level's natural unseeded one. No-ops every tick until the
// level has actually finished loading (SceneGraph/Level exist), then applies
// at most once, ever, for the lifetime of the process.
void MaybeApply(Engine* engine);

}  // namespace RLReplaySeed
