//-----------------------------------------------------------------------------
//           Name: main.cpp
//      Developer: Wolfire Games LLC
//    Description:
//        License: Read below
//-----------------------------------------------------------------------------
//
//   Copyright 2022 Wolfire Games LLC
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.
//
//-----------------------------------------------------------------------------
#ifndef __PS4__
#include <Internal/config.h>
#include <Internal/crashreport.h>
#include <Internal/error.h>
#include <Internal/filesystem.h>
#include <Internal/profiler.h>
#include <Internal/dialogues.h>

#include <Logging/consolehandler.h>
#include <Logging/filehandler.h>
#include <Logging/logdata.h>
#include <Logging/ramhandler.h>

#include <Main/engine.h>
#include <Main/altmain.h>
#include <Main/rl_benchmark.h>
#include <Main/rl_subsystem_timers.h>
#include <Main/rl_equivalence.h>
#include <Main/rl_action.h>
#include <Main/rl_obs_test.h>
#include <Main/rl_shm_transport.h>
#include <Main/rl_replay_seed.h>
#include <Math/rng_streams.h>

#include <Threading/rand.h>
#include <Threading/thread_sanity.h>

#include <Graphics/graphics.h>
#include <Memory/allocation.h>
#include <Version/version.h>
#include <Compat/platformsetup.h>
#include <Timing/timingevent.h>

#include <SDL.h>
#include <RecastAlloc.h>
#include <DetourAlloc.h>
#include <tclap/CmdLine.h>
#if ENABLE_FPU_SIGNALS == 1
#include <fenv.h>
#endif
#ifdef UNIT_TESTS
#include <UnitTests/testmain.h>
#endif
#include <sstream>

extern Config config;
extern bool mem_track_enable;
extern bool g_draw_vr;
ProfilerContext* g_profiler_ctx;

static bool debug_output = false;
static bool spam_output = false;
static bool clear_log = false;
static bool quit_after_load = false;
static bool no_dialogues = false;
static bool disable_rendering = false;
static bool load_all_levels = false;
static bool clear_cache = false;
static bool clear_cache_dry_run = false;
static bool level_load_stress = false;

static std::string overloadedWriteDir;
static std::string overloadedWorkingDir;

Allocation alloc;

RamHandler ram_handler;

static void* rcAllocReplacement(size_t size, rcAllocHint) {
    return og_malloc(size, OG_MALLOC_RC);
}

static void rcFreeReplacement(void* ptr) {
    og_free(ptr);
}

static void* dtAllocReplacement(size_t size, dtAllocHint) {
    return og_malloc(size, OG_MALLOC_DT);
}

static void dtFreeReplacement(void* ptr) {
    og_free(ptr);
}

#if defined PLATFORM_WINDOWS
#include <windows.h>
typedef enum PROCESS_DPI_AWARENESS {
    PROCESS_DPI_UNAWARE = 0,
    PROCESS_SYSTEM_DPI_AWARE = 1,
    PROCESS_PER_MONITOR_DPI_AWARE = 2
} PROCESS_DPI_AWARENESS;
// typedef BOOL (WINAPI * SETPROCESSDPIAWARE_T)(void);
typedef HRESULT(WINAPI* SETPROCESSDPIAWARENESS_T)(PROCESS_DPI_AWARENESS);
#ifdef OG_DEBUG
#include <DbgHelp.h>
#endif
#endif

int GameMain(int argc, char* argv[]) {
    RegisterMainThreadID();

#ifdef PLATFORM_WINDOWS
    HMODULE shcore = LoadLibraryA("Shcore.dll");
    SETPROCESSDPIAWARENESS_T SetProcessDpiAwareness = NULL;
    if (shcore) {
        SetProcessDpiAwareness = (SETPROCESSDPIAWARENESS_T)GetProcAddress(shcore, "SetProcessDpiAwareness");
    }
    // HMODULE user32 = LoadLibraryA("User32.dll");
    // SETPROCESSDPIAWARE_T SetProcessDPIAware = NULL;
    // if (user32) {
    //     SetProcessDPIAware = (SETPROCESSDPIAWARE_T) GetProcAddress(user32, "SetProcessDPIAware");
    // }

    if (SetProcessDpiAwareness) {
        if (SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE) != S_OK) {
            LOGW << "Couldn't set process dpi awareness (per monitor)" << std::endl;
        }
    } else if (SetProcessDPIAware != nullptr) {
        if (!SetProcessDPIAware()) {
            LOGW << "Couldn't set process dpi awareness (system)" << std::endl;
        }
    }

    // if (user32) {
    //     FreeLibrary(user32);
    // }
    if (shcore) {
        FreeLibrary(shcore);
    }
#ifdef OG_DEBUG
    SymSetOptions(SYMOPT_LOAD_LINES);
    HANDLE handle = GetCurrentProcess();
    SymInitialize(handle, NULL, TRUE);
#endif  // OG_DEBUG
#endif

    rand_ts_seed((unsigned int)time(NULL));

    alloc.Init();

    dtAllocSetCustom(dtAllocReplacement, dtFreeReplacement);
    rcAllocSetCustom(rcAllocReplacement, rcFreeReplacement);

    // Initialize profiler
    ProfilerContext profiler_context;
    profiler_context.Init(&alloc.stack);
    g_profiler_ctx = &profiler_context;

    PROFILER_ENTER(&profiler_context, "SetUpEnvironment");
    SetUpEnvironment(argv[0], overloadedWriteDir.c_str(), overloadedWorkingDir.c_str());
    PROFILER_LEAVE(&profiler_context);

    // Have to wait until we SetUpEnvironment before we can get write dir.
    FileHandler fileLogHandler(GetLogfilePath(), clear_log ? 0 : 10 * 1024 * 1024, 40 * 1024 * 1024);

    STIMING_INIT(GetWritePath(CoreGameModID).c_str());

    STIMING_ADDVISUALIZATION(STUpdate, vec3(0.7, 0.3, 0.3),     // Redish
                             STDraw, vec3(0.3, 0.7, 0.3),       // Greenish
                             STDrawSwap, vec3(0.3, 0.3, 0.7));  // Blueish

    STIMING_SETVISUALIZATIONSCALE(16, 32);

    STIMING_INITVISUALIZATION();

    LogTypeMask level =
        LogSystem::info | LogSystem::warning | LogSystem::error | LogSystem::fatal;
    if (debug_output) {
        level |= LogSystem::debug;
    }
    if (spam_output) {
        level |= LogSystem::spam;
    }
    LogSystem::RegisterLogHandler(level, &fileLogHandler);

    LOGI << "Starting program. Version " << GetBuildVersion() << "_" << GetBuildIDString() << " " << GetBuildTimestamp() << " " << GetArch() << " " << GetPlatform() << std::endl;

#ifdef NDEBUG
    LOGI << "Deploy (Release)" << std::endl;
#else
    LOGI << "Debug" << std::endl;
#endif

    PROFILER_ENTER(&profiler_context, "Engine initialize");
    // Engine* engine = (Engine*)alloc.stack.Alloc(sizeof(Engine));
    // new(engine) Engine;
    Engine* engine = new Engine();
    engine->Initialize();
    if (RLBenchmark::Enabled()) {
        rand_ts_seed(RLBenchmark::Seed());
        srand(RLBenchmark::Seed());
        RngStreams::SeedEpisode(RLBenchmark::Seed());
        RLBenchmark::OnEngineInitialized();
    }
    Dialog::Initialize();
    PROFILER_LEAVE(&profiler_context);

    if (clear_cache) {
        ClearCache(false);
    }
    if (clear_cache_dry_run) {
        ClearCache(true);
    }

    // Main loop
    while (!engine->quitting_) {
        STIMING_STARTFRAME();

        STIMING_START_COARSE(STUpdate);
        engine->Update();
        STIMING_END_COARSE(STUpdate);

        bool time_to_draw = true;
        static uint32_t last_time = 0;

        if (!engine->quitting_ && disable_rendering == false) {
            Graphics* graphics = Graphics::Instance();
            if (!graphics->config_.vSync() && graphics->config_.limit_fps_in_game()) {
                PROFILER_ZONE_IDLE(g_profiler_ctx, "SDL_Sleep");
                if (!g_draw_vr) {
                    time_to_draw = false;

                    int max_frame_rate = graphics->config_.max_frame_rate();
                    if (max_frame_rate < 15) {
                        max_frame_rate = 15;
                    }
                    if (max_frame_rate > 500) {
                        max_frame_rate = 500;
                    }

                    int ticks_to_wait = 1000 / max_frame_rate;
                    int diff = ticks_to_wait - (SDL_TS_GetTicks() - last_time) - 1;
                    if (diff > ticks_to_wait - 1) {
                        diff = ticks_to_wait - 1;
                    }

                    if (diff > 1) {
                        SDL_Delay(1);
                    } else if (diff < 1) {
                        time_to_draw = true;
                    }
                }
            }

            if (time_to_draw) {
                STIMING_START_COARSE(STDraw);
                engine->Draw();
                last_time = SDL_TS_GetTicks();
                STIMING_END_COARSE(STDraw);
            }
        }

        if (time_to_draw && !disable_rendering) {
            STIMING_START_COARSE(STDrawSwap);
            Graphics::Instance()->SwapToScreen();
            STIMING_END_COARSE(STDrawSwap);

            Graphics::Instance()->ClearGLState();
        }

        STIMING_ENDFRAME();

        SDL_Delay(0);  // Allow other threads to run
        PROFILER_TICK(g_profiler_ctx);
    }

    STIMING_FINALIZE();

    RLBenchmark::Report();
    RLEquivalence::Finalize();

    LOGI << "Final check if savefile needs to be written..." << std::endl;
    Engine::Instance()->save_file_.ExecuteQueuedWrite();

    LOGI << "Cleanly disposing of loaded assets..." << std::endl;
    engine->Dispose();
    // engine->~Engine();
    delete engine;
    // alloc.stack.Free(engine);

    profiler_context.Dispose(&alloc.stack);

    LOGI << "Program terminated successfully." << std::endl;
    DisposeEnvironment();

    LogSystem::DeregisterLogHandler(&fileLogHandler);

    alloc.Dispose();

#if defined PLATFORM_WINDOWS && defined OG_DEBUG
    SymCleanup(handle);
#endif

    LOGE << "Shutting down" << std::endl;
    LogSystem::Flush();
    return 0;
}

int main(int argc, char* argv[]) {
#if ENABLE_FPU_SIGNALS == 1
    feenableexcept(FE_INVALID | FE_OVERFLOW);
#endif

    //_CrtSetDbgFlag(_CRTDBG_CHECK_ALWAYS_DF);
    // Check for the command-line arg that XCode automatically adds when starting through the debugger,
    // and remove it so as not to confuse the parser
    for (int i = 0; i < argc - 1; i++) {
        if (strcmp(argv[i], "-NSDocumentRevisionsDebugMode") == 0) {
            strcpy(argv[i], "");
            strcpy(argv[i + 1], "");
        }
    };

    try {
        std::string full_version = std::string(GetBuildVersion()) + "_" + GetBuildIDString();
        // Set up command-line-parser
        TCLAP::CmdLine cmd("Overgrowth", ' ', full_version.c_str());

        TCLAP::ValueArg<std::string> configurationArg("c", "config", "Configuration string", false, "", "string");
        cmd.add(configurationArg);

        TCLAP::ValueArg<std::string> levelArg("l", "level", "Load level on startup", false, "", "string");
        cmd.add(levelArg);

        TCLAP::ValueArg<std::string> writeDirArg("", "write-dir", "Force set write directory to something else than system default.", false, "", "string");
        cmd.add(writeDirArg);

        TCLAP::ValueArg<std::string> workingDirArg("", "working-dir", "Force set the working dir for the application.", false, "", "string");
        cmd.add(workingDirArg);

        TCLAP::ValueArg<std::string> ogdaManifest("", "ogda-manifest", "Ogda generated manifest of game assets.", false, "", "string");
        cmd.add(ogdaManifest);

        TCLAP::SwitchArg ddsconvertSwitch("", "ddsconvert", "Start game with DDSConvert.", cmd, false);
        TCLAP::SwitchArg debugOutput("d", "debug-output", "Start game with debug output", cmd, false);
        TCLAP::SwitchArg spamOutput("s", "spam-output", "Start game with spammy debug output", cmd, false);
        TCLAP::SwitchArg quitAfterLoad("", "quit-after-load", "Turn of the game after level load is performed", cmd, false);
        TCLAP::SwitchArg noDialogues("", "no-dialogues", "Skip creating dialogues and instead do automatic responses", cmd, false);
        TCLAP::SwitchArg clearLog("", "clear-log", "Empty the log instead of appending to it", cmd, false);
        TCLAP::SwitchArg disableRendering("", "disable-rendering", "Disable the draw loop", cmd, false);
        TCLAP::SwitchArg loadAllLevels("", "load-all-levels", "Load all levels specified in the ogda build manifest", cmd, false);
        TCLAP::SwitchArg clearCache("", "clear-cache", "Clear the write folder of known cache files", cmd, false);
        TCLAP::SwitchArg clearCacheDryRun("", "clear-cache-dry-run", "Clear the write folder of known cache files (dry run)", cmd, false);
        TCLAP::SwitchArg levelLoadStress("", "level-load-stress", "Load levels in a loop", cmd, false);
        TCLAP::SwitchArg benchmark("", "benchmark", "Run an exact production-timestep benchmark", cmd, false);
        TCLAP::ValueArg<int> benchmarkWarmupSteps("", "benchmark-warmup-steps", "Completed timesteps before measurement", false, 120, "integer");
        TCLAP::ValueArg<int> benchmarkSteps("", "benchmark-steps", "Completed timesteps to measure (safety cap when --benchmark-measure-seconds is set)", false, 600, "integer");
        TCLAP::ValueArg<int> benchmarkSeed("", "benchmark-seed", "Deterministic benchmark seed", false, 1, "integer");
        TCLAP::ValueArg<double> benchmarkMeasureSeconds("", "benchmark-measure-seconds", "Measure for a fixed wall-clock duration instead of a fixed step count (Stage 0.3 overlapping-window concurrency measurement)", false, 0.0, "seconds");
        TCLAP::ValueArg<std::string> benchmarkBarrierDir("", "benchmark-barrier-dir", "Directory used to synchronize the measurement start of concurrently-launched benchmark workers", false, "", "string");
        TCLAP::ValueArg<int> benchmarkBarrierWorkers("", "benchmark-barrier-workers", "Number of concurrent workers to wait for at the readiness barrier before --benchmark-barrier-dir is used", false, 0, "integer");
        TCLAP::SwitchArg benchmarkSubsystemTimers("", "benchmark-subsystem-timers", "Report opt-in per-subsystem timing breakdown alongside the benchmark result (Stage 0.5)", cmd, false);
        TCLAP::ValueArg<double> benchmarkProgressSeconds("", "benchmark-progress-seconds", "Emit an RL_BENCHMARK_PROGRESS line every N seconds of measurement, for long sustained runs (Stage 0.8)", false, 0.0, "seconds");
        TCLAP::ValueArg<std::string> equivalenceDigestPath("", "equivalence-digest", "Stage 1: record a per-step state digest (position/velocity/health/etc + hash chain) to this path", false, "", "string");
        TCLAP::ValueArg<std::string> equivalenceTracePath("", "equivalence-trace", "Stage 1: record the legal-input trace (controller_id-addressed) to this path", false, "", "string");
        TCLAP::ValueArg<std::string> equivalenceExpectedPath("", "equivalence-expected", "OGRL-20260820-048: compare each rendered replay tick against a native recorded digest", false, "", "string");
        TCLAP::ValueArg<std::string> equivalenceReportPath("", "equivalence-report", "OGRL-20260820-048: write the rendered replay verification result to this path", false, "", "string");
        TCLAP::SwitchArg benchmarkResetAfterWarmup("", "benchmark-reset-after-warmup", "Stage 4 Approach B: reload the training scenario in-process after warmup, then measure the immediately-following steps", cmd, false);
        TCLAP::ValueArg<int> rlActionControllerId("", "rl-action-controller-id", "Stage 5.3: controller_id to drive with injected RL actions instead of real input", false, -1, "integer");
        TCLAP::SwitchArg rlActionTestForward("", "rl-action-test-forward", "Stage 5.3 smoke test: drive the RL-action controller with a constant forward move axis, to verify the injection path end to end", cmd, false);
        TCLAP::ValueArg<std::string> rlActionScriptPath("", "rl-action-script", "Stage 5.3: path to a sparse step-indexed action script (step,move_x,move_y,jump,crouch,attack,grab,drop,walk), for testing timing-sensitive combos", false, "", "string");
        TCLAP::ValueArg<int> rlObsPeriod("", "rl-obs-period", "Stage 5.1/5.5: extract an observation for the --rl-action-controller-id character every N steps (1 = every step, matching Stage 6's obs_period ablation concept); 0 disables extraction entirely", false, 0, "integer");
        TCLAP::ValueArg<int> rlObsDumpSteps("", "rl-obs-dump-steps", "Stage 5.1 smoke test: print this many extracted observations (schema version, LOS rule version, every named field), separate from --rl-obs-period's extraction cadence", false, 0, "integer");
        TCLAP::ValueArg<std::string> rlShmName("", "rl-shm-name", "Stage 5.4: enable the shm observation/action transport for --rl-action-controller-id under this POSIX IPC name (must start with '/', stay short -- Darwin's name limit is ~31 bytes)", false, "", "string");
        TCLAP::ValueArg<int> rlActPeriod("", "rl-act-period", "Stage 6 (OGRL-20260816-021): run the shm request/response handshake every N physics ticks instead of every tick (1 = every tick / 120Hz decisions, the original behavior; 4 = 30Hz decisions, matching vanilla AI's own control period and AGENTS.md's action contract). RLAction holds the last decision across the ticks in between.", false, 1, "integer");
        TCLAP::SwitchArg rlObsFovNpcMatched("", "rl-obs-fov-npc-matched", "Stage 5.2: use the FOV-matched vision profile (approximates the vanilla AI's own FOV-cone + stochastic bone-raycast perception, Data/Scripts/aschar.as::GetVisibleCharacters) instead of the default omnidirectional geometric LOS, for --rl-obs-period", cmd, false);
        // OGRL-20260817-030: deterministic tape replay (RLReplaySeed) -- reseeds
        // RNG + re-applies the curriculum scenario axes once, right after the
        // level's own natural initial load, so --rl-action-script replay
        // (Tools/rl/replay_ghost.py) can reproduce the SAME opponent a tape was
        // originally recorded against, not just the same button presses. See
        // rl_replay_seed.h's module comment for why this doesn't need the shm
        // transport at all. Unset (-1) is the default no-op for every axis
        // except seed, which requires an explicit non-negative value to enable
        // the feature at all.
        TCLAP::ValueArg<int> rlReplaySeed("", "rl-replay-seed", "OGRL-20260817-030: deterministic seed to reproduce a recorded tape's opponent -- unset (-1, default) leaves replay unseeded exactly as before", false, -1, "integer");
        TCLAP::ValueArg<double> rlReplayDifficulty("", "rl-replay-difficulty", "OGRL-20260817-030: curriculum difficulty (0..1) to apply alongside --rl-replay-seed, matching the tape's recorded difficulty -- unset (-1, default) leaves difficulty at the level's own default", false, -1.0, "float");
        TCLAP::ValueArg<int> rlReplayOpponents("", "rl-replay-opponents", "OGRL-20260817-030: opponent count to apply alongside --rl-replay-seed -- unset (-1, default) leaves it at the level's own default", false, -1, "integer");
        TCLAP::ValueArg<double> rlReplayWeapons("", "rl-replay-weapons", "OGRL-20260817-030: armed-round probability to apply alongside --rl-replay-seed -- unset (-1, default) leaves it at the level's own default", false, -1.0, "float");
        TCLAP::ValueArg<int> rlReplaySpecies("", "rl-replay-species", "OGRL-20260817-030: opponent species mode to apply alongside --rl-replay-seed -- unset (-1, default) leaves it at the level's own default", false, -1, "integer");
        TCLAP::ValueArg<int> rlReplayResetMode("", "rl-replay-reset-mode", "OGRL-20260820-048: recorded reset mode (0 hard, 1 soft)", false, 0, "integer");
        TCLAP::ValueArg<int> rlReplayControlledCharacterId("", "rl-replay-controlled-character-id", "OGRL-20260820-048: recorded character object ID receiving controller 0", false, -1, "integer");
        TCLAP::SwitchArg rlReplayScriptActions("", "rl-replay-script-actions", "OGRL-20260820-049: use the recorded decision-cadence script instead of the per-tick native action scheduler (diagnostic)", cmd, false);
        // OGRL-20260817-033: a scripted replay's final recorded action is very
        // often the decisive one (an attack connecting, a knockout) -- without
        // this, the auto-quit added in OGRL-20260817-031 closes the window
        // the instant that action is reached, before a human watching has any
        // chance to see it resolve. 0 (default) preserves the -031 instant-quit
        // behavior for any caller that doesn't set it.
        TCLAP::ValueArg<double> rlActionScriptHoldSeconds("", "rl-action-script-hold-seconds", "OGRL-20260817-033: hold the final recorded action on screen this many seconds (real simulated time) after a --rl-action-script recording ends, before auto-quitting -- 0 (default) quits instantly", false, 0.0, "float");
        cmd.add(rlActionControllerId);
        cmd.add(rlActionScriptPath);
        cmd.add(rlObsPeriod);
        cmd.add(rlObsDumpSteps);
        cmd.add(rlShmName);
        cmd.add(rlActPeriod);
        cmd.add(rlReplaySeed);
        cmd.add(rlReplayDifficulty);
        cmd.add(rlReplayOpponents);
        cmd.add(rlReplayWeapons);
        cmd.add(rlReplaySpecies);
        cmd.add(rlReplayResetMode);
        cmd.add(rlReplayControlledCharacterId);
        cmd.add(rlActionScriptHoldSeconds);
        cmd.add(benchmarkWarmupSteps);
        cmd.add(benchmarkSteps);
        cmd.add(benchmarkSeed);
        cmd.add(benchmarkMeasureSeconds);
        cmd.add(benchmarkBarrierDir);
        cmd.add(benchmarkBarrierWorkers);
        cmd.add(benchmarkProgressSeconds);
        cmd.add(equivalenceDigestPath);
        cmd.add(equivalenceTracePath);
        cmd.add(equivalenceExpectedPath);
        cmd.add(equivalenceReportPath);
#ifdef UNIT_TESTS
        TCLAP::SwitchArg runUnitTests("", "run-unit-tests", "Run all unit tests", cmd, false);
#endif

        // Actually parse the command line
        cmd.parse(argc, argv);

        // Extract information from command-line-parser
        bool runWithDDSConvert = ddsconvertSwitch.getValue();
        std::string levelname = levelArg.getValue();
        std::string configuration = configurationArg.getValue();
        std::string manifest = ogdaManifest.getValue();

        overloadedWriteDir = writeDirArg.getValue();
        overloadedWorkingDir = workingDirArg.getValue();
        debug_output = debugOutput.getValue();
        spam_output = spamOutput.getValue();
        clear_log = clearLog.getValue();
        quit_after_load = quitAfterLoad.getValue();
        no_dialogues = noDialogues.getValue();
        disable_rendering = disableRendering.getValue();
        load_all_levels = loadAllLevels.getValue();
        clear_cache = clearCache.getValue();
        clear_cache_dry_run = clearCacheDryRun.getValue();
        level_load_stress = levelLoadStress.getValue();
        if (benchmark.getValue() && (benchmarkWarmupSteps.getValue() < 0 || benchmarkSteps.getValue() <= 0 || benchmarkSeed.getValue() < 0)) {
            std::cerr << "benchmark warmup and seed must be non-negative, and benchmark steps must be positive" << std::endl;
            return 2;
        }
        if (benchmarkResetAfterWarmup.getValue() && (!benchmark.getValue() || benchmarkWarmupSteps.getValue() <= 0)) {
            std::cerr << "benchmark reset requires --benchmark and at least one warmup step" << std::endl;
            return 2;
        }
        if (!rlShmName.getValue().empty() && rlActionControllerId.getValue() < 0) {
            std::cerr << "--rl-shm-name requires --rl-action-controller-id (selects which character it drives/observes)" << std::endl;
            return 2;
        }
        if (rlActPeriod.getValue() < 1) {
            std::cerr << "--rl-act-period must be at least 1" << std::endl;
            return 2;
        }
        if (rlReplaySeed.getValue() >= 0) {
            RLReplaySeed::Configure(static_cast<unsigned int>(rlReplaySeed.getValue()),
                                     static_cast<float>(rlReplayDifficulty.getValue()),
                                     rlReplayOpponents.getValue(),
                                     static_cast<float>(rlReplayWeapons.getValue()),
                                     rlReplaySpecies.getValue(),
                                     rlReplayResetMode.getValue(),
                                     rlReplayControlledCharacterId.getValue());
        }
        RLBenchmark::Configure(benchmark.getValue(), static_cast<uint64_t>(benchmarkWarmupSteps.getValue()), static_cast<uint64_t>(benchmarkSteps.getValue()), static_cast<unsigned int>(benchmarkSeed.getValue()),
                               benchmarkMeasureSeconds.getValue(), benchmarkBarrierDir.getValue(), benchmarkBarrierWorkers.getValue(),
                               benchmarkProgressSeconds.getValue(), benchmarkResetAfterWarmup.getValue());
        RLSubsystemTimers::SetEnabled(benchmarkSubsystemTimers.getValue());
        if (!equivalenceExpectedPath.getValue().empty()) {
            RLEquivalence::ConfigureReplay(equivalenceExpectedPath.getValue(), equivalenceReportPath.getValue());
        } else if (!equivalenceDigestPath.getValue().empty() || !equivalenceTracePath.getValue().empty()) {
            RLEquivalence::Configure(RLEquivalence::Mode::kRecord, equivalenceDigestPath.getValue(), equivalenceTracePath.getValue(), static_cast<unsigned int>(benchmarkSeed.getValue()));
        }
        if (rlActionControllerId.getValue() >= 0) {
            RLAction::Configure(true, rlActionControllerId.getValue());
            if (rlActionTestForward.getValue()) {
                RLAction::SetMoveAxes(0.0f, 1.0f);
            }
            if (!rlActionScriptPath.getValue().empty()) {
                if (!RLAction::LoadScript(rlActionScriptPath.getValue())) {
                    std::cerr << "failed to load --rl-action-script: " << rlActionScriptPath.getValue() << std::endl;
                    return 2;
                }
                // OGRL-20260817-031: a script written by watch.py/tape.py's
                // jsonl_to_ghost_csv (one row per DECISION, not per tick)
                // needs --rl-act-period to match the act_period it was
                // recorded at, or it replays act_period times too fast and
                // never stops -- see rl_action.h's SetScriptPeriod comment.
                RLAction::SetScriptPeriod(rlActPeriod.getValue());
                RLAction::SetScriptHoldSeconds(static_cast<float>(rlActionScriptHoldSeconds.getValue()));
                if (!equivalenceExpectedPath.getValue().empty() && !rlReplayScriptActions.getValue()) {
                    RLAction::SetNativeReplayMode(true);
                }
            }
            if (rlObsPeriod.getValue() > 0 || rlObsDumpSteps.getValue() > 0) {
                // --rl-obs-dump-steps alone (no explicit --rl-obs-period) means
                // "just show me some observations" -- default to extracting
                // every step so the requested dumps actually get printed.
                const int period = rlObsPeriod.getValue() > 0 ? rlObsPeriod.getValue() : 1;
                const RLObservation::FovProfile fov_profile = rlObsFovNpcMatched.getValue() ? RLObservation::FovProfile::kNpcMatched : RLObservation::FovProfile::kOmnidirectional;
                RLObsTest::Configure(rlActionControllerId.getValue(), period, rlObsDumpSteps.getValue(), fov_profile);
            }
            if (!rlShmName.getValue().empty()) {
                RLObservation::ObservationConfig obs_config;  // default shape for v1; not yet CLI-configurable
                if (!RLShmTransport::Configure(rlShmName.getValue(), rlActionControllerId.getValue(), obs_config, rlActPeriod.getValue())) {
                    std::cerr << "failed to set up --rl-shm-name transport: " << rlShmName.getValue() << std::endl;
                    return 2;
                }
            }
        }
        if (RLBenchmark::Enabled()) {
            disable_rendering = true;
        }

        std::stringstream configurationStream(configuration);
        config.Load(configurationStream, false, true);

        // If command line specified a level, skip main menu and jump to loading that level
        if (!levelname.empty()) {
            std::stringstream ss;
            ss << "debug_load_level: " << levelname << std::endl;
            ss << "main_menu: false" << std::endl;
            config.Load(ss, false, true);
        }

        if (!manifest.empty()) {
            std::stringstream ss;
            ss << "ogda_manifest: " << manifest << std::endl;
            config.Load(ss, false, true);
        }

        if (quit_after_load) {
            std::stringstream ss;
            ss << "quit_after_load: true" << std::endl;
            config.Load(ss, false, true);
        }

        if (no_dialogues) {
            std::stringstream ss;
            ss << "no_dialogues: true" << std::endl;
            config.Load(ss, false, true);
        }

        if (load_all_levels) {
            std::stringstream ss;
            ss << "load_all_levels: true" << std::endl;
            config.Load(ss, false, true);
        }

        if (level_load_stress) {
            std::stringstream ss;
            ss << "level_load_stress: true" << std::endl;
            config.Load(ss, false, true);
        }

        /******************************************************/
        // Register logging handlers
        ConsoleHandler consoleHandler;
        LogTypeMask level =
            LogSystem::info | LogSystem::warning | LogSystem::error | LogSystem::fatal;

        if (debug_output) {
            level |= LogSystem::debug;
        }

        if (spam_output) {
            level |= LogSystem::spam;
        }

        // Choose which main function to run
        int ret;
        if (runWithDDSConvert) {
            LogSystem::RegisterLogHandler(level, &consoleHandler);
            ret = DDSConvertMain(argc, argv, overloadedWriteDir.c_str(), overloadedWorkingDir.c_str());
            LogSystem::DeregisterLogHandler(&consoleHandler);
#ifdef UNIT_TESTS
        } else if (runUnitTests.getValue()) {
            ret = RunUnitTests();
#endif
        } else {
            LogSystem::RegisterLogHandler(level, &consoleHandler);
            LogSystem::RegisterLogHandler(level, &ram_handler);
            ret = RunWithCrashReport(argc, argv, &GameMain);
            LogSystem::DeregisterLogHandler(&ram_handler);
            LogSystem::DeregisterLogHandler(&consoleHandler);
        }

        return ret;
    } catch (TCLAP::ArgException& e) {
        std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
    }
    return 0;
}

#else
#include <cstdio>
#include <sys/stat.h>

struct PS4Engine {
    void Initialize();
};

void PS4Engine::Initialize() {
    FILE* file = fopen("/app0/Data/", "rb");
}

int main(int argc, char* argv[]) {
    ConsoleHandler consoleHandler;
    LogTypeMask level =
        LogSystem::info | LogSystem::warning | LogSystem::error | LogSystem::fatal;

    if (debug_output) {
        level |= LogSystem::debug;
    }
    if (spam_output) {
        level |= LogSystem::spam;
    }

    PS4Engine engine;
    engine.Initialize();

    LogSystem::DeregisterLogHandler(&consoleHandler);
    return 0;
}
#endif
