#include "rl_replay_seed.h"

#include <Main/engine.h>
#include <Main/rl_action.h>
#include <Main/rl_equivalence.h>
#include <Main/scenegraph.h>
#include <Objects/movementobject.h>
#include <Game/level.h>
#include <Math/rng_streams.h>
#include <Threading/rand.h>

#include <cstdio>
#include <cstdlib>

namespace RLReplaySeed {
namespace {
bool g_configured = false;
bool g_applied = false;
unsigned int g_seed = 0;
float g_difficulty = -1.0f;
int g_opponents = -1;
float g_weapons = -1.0f;
int g_species = -1;
int g_reset_mode = 0;
int g_controlled_character_id = -1;
}  // namespace

void Configure(unsigned int seed, float difficulty, int opponents, float weapons, int species,
               int reset_mode, int controlled_character_id) {
    g_configured = true;
    g_applied = false;
    g_seed = seed;
    g_difficulty = difficulty;
    g_opponents = opponents;
    g_weapons = weapons;
    g_species = species;
    g_reset_mode = reset_mode == 1 ? 1 : 0;
    g_controlled_character_id = controlled_character_id;
}

bool Enabled() { return g_configured; }

void MaybeApply(Engine* engine) {
    if (!g_configured || g_applied) {
        return;
    }
    SceneGraph* scenegraph = engine->GetSceneGraph();
    if (scenegraph == nullptr || scenegraph->level == nullptr) {
        return;  // level hasn't finished its own natural initial load yet -- try again next tick
    }
    g_applied = true;  // exactly once, ever, even if something below turns out to be a no-op

    char buf[128];
    if (g_reset_mode == 1) {
        // Soft training resets push the curriculum axes before the level
        // script's post_reset handler, matching rl_shm_transport.cpp.
        if (g_difficulty >= 0.0f) {
            std::snprintf(buf, sizeof(buf), "set_rl_difficulty %g", static_cast<double>(g_difficulty));
            scenegraph->level->Message(buf);
        }
        if (g_opponents >= 0) {
            std::snprintf(buf, sizeof(buf), "set_rl_opponents %d", g_opponents);
            scenegraph->level->Message(buf);
        }
        if (g_weapons >= 0.0f) {
            std::snprintf(buf, sizeof(buf), "set_rl_weapons %g", static_cast<double>(g_weapons));
            scenegraph->level->Message(buf);
        }
        if (g_species >= 0) {
            std::snprintf(buf, sizeof(buf), "set_rl_species %d", g_species);
            scenegraph->level->Message(buf);
        }
        engine->SoftResetRLTrainingScenario(g_seed);
    } else {
        // Hard training resets rebuild the level before applying the scenario
        // axes and invoking post_reset. A standalone replay must use this
        // production path instead of substituting a lighter reset.
        if (!engine->ResetRLTrainingScenario(g_seed)) {
            std::fprintf(stderr, "RL replay hard reset failed for seed %u\n", g_seed);
            return;
        }
        scenegraph = engine->GetSceneGraph();
        if (scenegraph == nullptr || scenegraph->level == nullptr) {
            std::fprintf(stderr, "RL replay hard reset produced no level for seed %u\n", g_seed);
            return;
        }
        if (g_difficulty >= 0.0f) {
            std::snprintf(buf, sizeof(buf), "set_rl_difficulty %g", static_cast<double>(g_difficulty));
            scenegraph->level->Message(buf);
        }
        if (g_opponents >= 0) {
            std::snprintf(buf, sizeof(buf), "set_rl_opponents %d", g_opponents);
            scenegraph->level->Message(buf);
        }
        if (g_weapons >= 0.0f) {
            std::snprintf(buf, sizeof(buf), "set_rl_weapons %g", static_cast<double>(g_weapons));
            scenegraph->level->Message(buf);
        }
        if (g_species >= 0) {
            std::snprintf(buf, sizeof(buf), "set_rl_species %d", g_species);
            scenegraph->level->Message(buf);
        }
        scenegraph->level->Message("post_reset");
    }
    if (g_controlled_character_id >= 0) {
        MovementObject* target = nullptr;
        for (Object* object : scenegraph->movement_objects_) {
            MovementObject* mo = static_cast<MovementObject*>(object);
            if (mo->GetID() == g_controlled_character_id) {
                target = mo;
                break;
            }
        }
        if (target != nullptr) {
            for (Object* object : scenegraph->movement_objects_) {
                MovementObject* mo = static_cast<MovementObject*>(object);
                // Match the recorded level identity before the first physics
                // tick. AvatarControlManager will perform the normal
                // controlled/PC-script transition on its own update, just as
                // it did during training; setting controlled here would move
                // that transition one digest tick early.
                mo->is_player = (mo == target);
            }
        }
    }
    // The script loader was configured before the natural startup level. The
    // reset must put its tick cursor back at native episode tick zero, just as
    // RLShmTransport clears held controls at a training reset boundary.
    RLAction::ResetForEpisode();
    RLEquivalence::ApplyReplayAction(0);
    RLEquivalence::OnEpisodeReset(g_seed, 0, g_difficulty, g_opponents,
                                  g_weapons, g_species);
}

}  // namespace RLReplaySeed
