"""Gym-shaped wrapper tying together RLShmTransport (shm_env.ShmEnv),
RLObservation's schema (obs_schema.py), and the reward function (reward.py)
into a single env.reset()/env.step(action) surface a PPO trainer can drive,
plus ownership of the underlying engine subprocess's lifecycle (launch,
attach, clean shutdown) so a training script only has to talk to this class.

Action convention (matches Source/Main/rl_shm_transport.cpp's fixed 8-float
action slot exactly, in this order): [move_x, move_y, jump, crouch, attack,
grab, drop, walk]. move_x/move_y are continuous in [-1, 1]; the remaining six
are interpreted as already-sampled 0.0/1.0 (or any float, thresholded at 0.5
on the C++ side) -- the policy's discrete heads are expected to sample before
calling step(), not pass raw logits.
"""

from __future__ import annotations

import os
import sys
import subprocess
import tempfile
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

import noaslr
import paths
from obs_schema import ObsLayout, DEFAULT_LAYOUT, SCHEMA_VERSION
from reward import RewardComputer, RewardConfig
from shm_env import ShmEnv

ACTION_DIM = 8
CONTINUOUS_ACTION_DIM = 2   # move_x, move_y
DISCRETE_ACTION_DIM = 6     # jump, crouch, attack, grab, drop, walk

_STALE_CLEANUP_LOCK = threading.Lock()
_STALE_CLEANED_PARENTS: set[Path] = set()


def _cleanup_stale_write_dirs(write_dir_parent: Path) -> None:
    """Startup sweep for write-dirs (and their sibling .log files) orphaned
    by a prior engine process that never reached its own close() -- a hard
    kill, a crash, an abruptly terminated parent script. close()'s own
    cleanup (below) is real but can only run on a clean exit path, and this
    project has now hit the disk-full failure mode this guards against
    TWICE (OGRL-20260815-034, and again 2026-08-17: 203 accumulated
    write-dirs / 3.7GB brought free space down to ~2GB and triggered
    run11's disk-safety-net stop). Only ever deletes a write-dir NOT
    referenced by any live process's own --write-dir argument, so this can
    never touch something actually in use -- if the liveness check itself
    fails for any reason, it does nothing rather than risk deleting
    something live. Best-effort, called once per env launch; cheap when
    there's nothing stale to find."""
    write_dir_parent = write_dir_parent.resolve()
    # Vector launch constructs several OvergrowthEnv objects concurrently.
    # Running this liveness sweep independently in every constructor creates a
    # race: constructor A can create a write-dir, then constructor B can scan
    # before A's child process appears in ps and delete A's live directory.
    # One successful sweep per parent process is sufficient; clean shutdown
    # removes the directories created by this process, while the next Python
    # process gets a fresh stale sweep.
    with _STALE_CLEANUP_LOCK:
        if write_dir_parent in _STALE_CLEANED_PARENTS:
            return
        if not write_dir_parent.exists():
            _STALE_CLEANED_PARENTS.add(write_dir_parent)
            return
        try:
            ps_output = subprocess.run(["ps", "-eo", "command"], capture_output=True, text=True, timeout=5).stdout
        except (OSError, subprocess.TimeoutExpired):
            return
        import shutil
        for entry in write_dir_parent.iterdir():
            if entry.suffix == ".log":
                continue  # handled alongside its write-dir below, not standalone
            if f"--write-dir {entry}" in ps_output:
                continue  # a live process still owns this one
            shutil.rmtree(entry, ignore_errors=True)
            log_path = write_dir_parent / f"{entry.name}.log"
            log_path.unlink(missing_ok=True)
        _STALE_CLEANED_PARENTS.add(write_dir_parent)


class OvergrowthEnv:
    def __init__(
        self,
        repo_root: str | Path,
        level: str = "arenas/oval_arena.xml",
        jumpkick_wariness: int = 0,
        throw_aggression_launch: float = 1.0,
        shm_name: str = "/ogrl_env0",
        controller_id: int = 0,
        seed: int = 1,
        layout: ObsLayout = DEFAULT_LAYOUT,
        reward_config: RewardConfig | None = None,
        write_dir_parent: str | Path | None = None,
        max_engine_steps: int = 50_000_000,  # effectively unbounded; --benchmark needs a finite cap
        binary_path: str | Path | None = None,
        launch_timeout_seconds: float = 60.0,
        frame_stack: int = 1,
        render: bool = False,
        time_scale_mult: int = 100,
        act_period: int = 1,
        equivalence_digest_path: str | Path | None = None,
        equivalence_trace_path: str | Path | None = None,
        log_attacks: bool = False,
        auto_camera: bool = False,
        spectator_fov: float | None = None,
        extra_config_lines: list[str] | None = None,
        keep_artifacts: bool = False,
        engine_priority: str | None = None,
        engine_affinity: str | None = None,
    ):
        # act_period (OGRL-20260816-021 Sec 1.3(a)/2.2(a), Stage 6): decision
        # rate divisor -- 1 = every physics tick is a decision (120Hz, the
        # original/default), 4 = every 4th tick (30Hz, matching vanilla AI's
        # own control period). See rl_shm_transport.cpp's Step() for what
        # actually happens on the engine side; this is just the CLI plumbing.
        self.act_period = act_period
        self.log_attacks = log_attacks
        self.auto_camera = bool(auto_camera)
        self.spectator_fov = float(spectator_fov) if spectator_fov is not None else None
        self.extra_config_lines = list(extra_config_lines or [])
        self.keep_artifacts = bool(keep_artifacts)
        # Optional per-engine overrides used by VecOvergrowthEnv for standby
        # placement experiments. None preserves the process-wide environment
        # defaults, so existing callers and the active-worker path are
        # unchanged.
        self.engine_priority = engine_priority
        self.engine_affinity = engine_affinity
        # render=True is "watch mode" (Tools/rl/ppo/watch.py): a real window,
        # no --benchmark fast-forward, so wall-clock time and in-game time
        # match -- letting a human actually watch the character move at
        # normal speed, instead of the ~100x-sped-up headless mode training
        # uses. time_scale_mult is exposed separately (not just tied to
        # render) since a slow-motion *rendered* replay (e.g. time_scale_mult=20)
        # is also a reasonable thing to want when inspecting a specific moment.
        self.render = render
        self.time_scale_mult = time_scale_mult
        self.equivalence_digest_path = Path(equivalence_digest_path) if equivalence_digest_path else None
        self.equivalence_trace_path = Path(equivalence_trace_path) if equivalence_trace_path else None
        self.repo_root = Path(repo_root)
        self.level = level
        self.jumpkick_wariness = int(jumpkick_wariness)
        self.throw_aggression_launch = float(throw_aggression_launch)
        self.shm_name = shm_name
        self.controller_id = controller_id
        self.seed = seed
        self.layout = layout
        self.reward_computer = RewardComputer(layout, reward_config)
        self.max_engine_steps = max_engine_steps
        # Frame stacking (Atari-DQN-style): concatenates the last N raw
        # observations (oldest first) into what reset()/step() return. The
        # policy here is a plain MLP with no recurrent core and no other
        # access to history beyond RLObservation's own action-history block
        # (which records past *actions*, not past *observations*) -- without
        # this, it cannot tell "steady at 80% health" from "just dropped to
        # 80% and falling" within a single forward pass, since both look
        # identical in one instantaneous frame. Reward computation is
        # unaffected -- it always uses the single-frame raw values
        # (self._prev_values), never the stacked/returned array, since it
        # needs the actual current/previous readings, not a flattened window.
        self.frame_stack = max(1, frame_stack)
        # Optional storage optimization for the hot observation path. The
        # legacy deque+concatenate implementation remains the default until a
        # paired benchmark proves that the preallocated ring preserves both
        # throughput and the owned-array API expected by callers.
        self._prealloc_frame_stack = os.environ.get("OGRL_PREALLOC_FRAME_STACK", "0") != "0"
        self._frame_stack_buffer: deque = deque(maxlen=self.frame_stack)
        self._frame_stack_array = (
            np.empty((self.frame_stack, self.layout.total_floats), dtype=np.float32)
            if self._prealloc_frame_stack else None
        )
        self._frame_stack_initialized = False
        self.binary_path = paths.engine_binary(self.repo_root, binary_path)
        self._launch_timeout = launch_timeout_seconds

        write_dir_parent = Path(write_dir_parent) if write_dir_parent else self.repo_root / ".rl_write_dirs"
        write_dir_parent.mkdir(parents=True, exist_ok=True)
        _cleanup_stale_write_dirs(write_dir_parent)
        self._write_dir = Path(tempfile.mkdtemp(prefix=f"env-{shm_name.strip('/')}-", dir=write_dir_parent))

        self._process: subprocess.Popen | None = None
        self._shm: ShmEnv | None = None
        self._prev_values: list | None = None
        self._episode_steps = 0
        # Engine::ResetRLTrainingScenario requires its baseline to be captured,
        # which only happens once the engine's own initial level load has
        # completed. ShmEnv connects earlier because the transport is created
        # during CLI processing. The first reset therefore drains the engine's
        # natural initial observation before issuing the requested reset. This
        # preserves readiness synchronization without silently training on an
        # episode whose seed/difficulty/opponents were never requested.
        self._used_initial_observation = False
        # episode_count counts requested scenario resets -- the initial natural
        # observation is only drained as a readiness handshake -- and is what
        # --hard-reset-every gates on (OGRL-20260817-028 Sec1.2). It deliberately
        # lives on the physical engine process, not on whatever vector slot
        # currently happens to be playing it (a standby moves between slots).
        self.episode_count = 0
        # last_reset_seed: the REAL seed most recently used to reset this
        # env, for episodes.jsonl (OGRL-20260817-028 Sec8.6 -- ghost replay
        # needs the actual seed, and train_vec.py was logging the worker
        # index as a placeholder before this existed).
        self.last_reset_seed: int | None = None
        self.last_reset_seconds: float = 0.0
        self._launch()

    # --- lifecycle ---

    def _launch(self) -> None:
        config_lines = [
            f"global_time_scale_mult: {self.time_scale_mult}",
            "skip_loading_pause: true",
            "has_detected_settings: true",
        ]
        if self.render:
            # Reproduced live 2026-09-11 on a play_match.py human duel: a
            # character's own this_mo.position/velocity goes NaN at the
            # moment of a knockout (aschar.as's CheckForNANPosAndVel logs
            # "Invalid position/velocity at N" and Breakpoint(0)s, but never
            # sanitizes -- the NaN is left live), and something derived from
            # that NaN transform -- NOT necessarily a blood decal; setting
            # blood:0 here did not stop it -- ends up in scenegraph.cpp's
            # decal/light cluster culling (proj_point.w() = nan), which
            # LOG_ASSERT_GTEQs every frame forever (Release builds compile
            # assert() out, so it never actually stops) while the NaN
            # corrupts the shared min/max accumulator for the WHOLE cluster
            # pass, not just the one bad object -- hence the entire frame
            # going black, not one bad decal. The REAL fix is in
            # scenegraph.cpp's PrepareLightsAndDecals (skip a non-finite
            # decal/light instead of letting it poison the shared
            # accumulator) -- verified live across 5 resets on the exact map
            # that reproduced this twice. blood:0 stays here as a harmless,
            # independent reduction in decal volume for a rendered session
            # (never mattered when training always used --disable-rendering),
            # not as the fix. Reproduced repeatedly on a freshly generated
            # arena (t_train_101_duel.xml, no baked navmesh); NOT reproduced
            # across 4+ resets on the long-tested oval_arena_human_duel.xml,
            # so the underlying NaN-velocity physics bug is real but
            # conditional, not a defect of the human-duel script itself, and
            # still open (needs a Bullet-side repro, not a render-side one).
            # Never surfaced in ~233k headless training episodes because
            # training never renders, so this whole code path never runs
            # there.
            config_lines.append("blood: 0")
            if self.auto_camera:
                # Render-only spectator aid. The policy never receives camera
                # state; legacy controlled-character target selection can
                # nevertheless read the camera, so this is diagnostic-only.
                config_lines.append("auto_camera: true")
                config_lines.append("rl_spectator_camera: true")
                config_lines.append("rl_spectator_camera_smooth: 0.92")
                if self.spectator_fov is not None:
                    config_lines.append(f"chase_camera_fov: {self.spectator_fov:g}")
        # Ground-truth attack telemetry (aschar.as's g_rl_log_attacks). Off for
        # training -- it costs log volume and nothing reads it there -- and
        # switched on only by the analysis tools that parse it.
        if self.log_attacks:
            config_lines.append("rl_log_attacks: true")
        if self.jumpkick_wariness:
            # Seeds every bot's got_hit_by_leg_cannon_count at spawn -- see
            # enemycontrol.as's ResetMind. 0 is stock behaviour.
            config_lines.append(f"rl_jumpkick_wariness: {int(self.jumpkick_wariness)}")
        if self.throw_aggression_launch != 1.0:
            # Also passed at LAUNCH, not only per-episode via the level script's
            # SetConfigValueFloat: enemycontrol reads it in a global initialiser
            # at character-spawn time, and the launch config is the path already
            # verified to reach GetConfigValue* (the wariness A/B).
            config_lines.append(f"rl_throw_aggression: {self.throw_aggression_launch}")
        config_lines.extend(self.extra_config_lines)
        config_str = "\n".join(config_lines)
        command = [str(self.binary_path), "--write-dir", str(self._write_dir), "--working-dir", str(self.repo_root)]
        if self.render:
            # No --disable-rendering, no --benchmark: --benchmark forces
            # disable_rendering=true unconditionally (Source/Main/main.cpp)
            # and drives physics as fast as possible rather than pacing to
            # real time, so it's not just "benchmark mode with a window" --
            # it has to be skipped entirely for a human to actually watch
            # anything. Physics still steps at a fixed 120Hz internally
            # either way (an engine property, not a --benchmark one); what
            # changes is real-time pacing and whether frames get drawn.
            command += ["--no-dialogues"]
        else:
            command += [
                "--disable-rendering",
                "--no-dialogues",
                "--benchmark",
                "--benchmark-warmup-steps", "0",
                "--benchmark-steps", str(self.max_engine_steps),
                "--benchmark-seed", str(self.seed),
            ]
        command += [
            "--level", self.level,
            "--config", config_str,
            "--rl-shm-name", self.shm_name,
            "--rl-action-controller-id", str(self.controller_id),
            "--rl-act-period", str(self.act_period),
        ]
        if self.equivalence_digest_path is not None:
            self.equivalence_digest_path.parent.mkdir(parents=True, exist_ok=True)
            command += ["--equivalence-digest", str(self.equivalence_digest_path)]
        if self.equivalence_trace_path is not None:
            self.equivalence_trace_path.parent.mkdir(parents=True, exist_ok=True)
            command += ["--equivalence-trace", str(self.equivalence_trace_path)]
        command = noaslr.wrap_command(command)
        log_path = self._write_dir.parent / f"{self._write_dir.name}.log"
        self._log_file = open(log_path, "w")
        popen_kwargs = {}
        if sys.platform == "win32":
            # 2026-09-20: engines launched from a scheduled task inherit
            # BelowNormal priority (measured: every Overgrowth.exe and the
            # trainer at BelowNormal while Chrome/Defender/Update run Normal).
            # The sync collector waits on the slowest of N engines every step,
            # so any preempted engine is the straggler for all of them.
            #   OGRL_ENGINE_PRIORITY: normal (default) | above | high | inherit
            #   OGRL_ENGINE_AFFINITY: hex mask, e.g. 0xFFF to keep 12C/14T
            #       Core Ultra engines off the two LP E-cores (CPUs 12-13)
            pri = (self.engine_priority or os.environ.get("OGRL_ENGINE_PRIORITY", "normal")).lower()
            flags = {"normal": 0x00000020, "above": 0x00008000, "high": 0x00000080}.get(pri, 0)
            if flags:
                popen_kwargs["creationflags"] = flags
        self._process = subprocess.Popen(command, cwd=self.repo_root, stdout=self._log_file, stderr=subprocess.STDOUT, **popen_kwargs)
        if sys.platform == "win32":
            try:
                self._apply_windows_scheduling(
                    self.engine_priority or os.environ.get("OGRL_ENGINE_PRIORITY", "normal"),
                    self.engine_affinity or os.environ.get("OGRL_ENGINE_AFFINITY"),
                    "launch",
                )
            except Exception as _e:  # never let a scheduling tweak break a launch
                print(f"[env] scheduling override not applied: {_e}", flush=True)

        deadline = time.perf_counter() + self._launch_timeout
        last_error = None
        while time.perf_counter() < deadline:
            if self._process.poll() is not None:
                self._fail_launch(
                    f"engine process exited early (code {self._process.returncode}) while connecting to {self.shm_name} -- see {log_path}"
                )
            try:
                self._shm = ShmEnv(self.shm_name, obs_floats=self.layout.total_floats, connect_retries=1, connect_retry_delay=0.0)
                break
            except OSError as exc:
                last_error = exc
                time.sleep(0.2)
        if self._shm is None:
            self._fail_launch(f"timed out connecting to {self.shm_name} after {self._launch_timeout}s: {last_error}")

        if self._shm.obs_floats != self.layout.total_floats:
            self._fail_launch(f"obs_floats mismatch: engine publishes {self._shm.obs_floats}, this layout expects {self.layout.total_floats}")
        if self._shm.schema_version != SCHEMA_VERSION:
            self._fail_launch(f"schema_version mismatch: engine publishes {self._shm.schema_version}, obs_schema.py expects {SCHEMA_VERSION} -- rebuild the engine or update obs_schema.py")

    def _apply_windows_scheduling(self, priority: str | None, affinity: str | None, reason: str) -> None:
        """Apply and verify a live engine's optional Windows scheduling role."""
        if sys.platform != "win32" or self._process is None:
            return
        import ctypes
        k32 = ctypes.windll.kernel32
        k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        k32.SetPriorityClass.restype = ctypes.c_int
        k32.GetPriorityClass.argtypes = [ctypes.c_void_p]
        k32.GetPriorityClass.restype = ctypes.c_uint32
        k32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        k32.SetProcessAffinityMask.restype = ctypes.c_int
        k32.GetProcessAffinityMask.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)
        ]
        k32.GetProcessAffinityMask.restype = ctypes.c_int
        h = k32.OpenProcess(0x0200 | 0x0400, False, self._process.pid)  # SET_INFORMATION|QUERY_INFORMATION
        if not h:
            raise ctypes.WinError()
        try:
            priority_classes = {"normal": 0x00000020, "above": 0x00008000, "high": 0x00000080}
            cls = priority_classes.get((priority or "").lower())
            if cls is not None and not k32.SetPriorityClass(h, cls):
                raise ctypes.WinError()
            applied_mask = None
            if affinity:
                requested_mask = ctypes.c_size_t(int(affinity, 16))
                if not k32.SetProcessAffinityMask(h, requested_mask):
                    raise ctypes.WinError()
                system_mask = ctypes.c_size_t()
                applied_mask = ctypes.c_size_t()
                if not k32.GetProcessAffinityMask(h, ctypes.byref(applied_mask), ctypes.byref(system_mask)):
                    raise ctypes.WinError()
                if applied_mask.value != requested_mask.value:
                    raise RuntimeError(f"requested {affinity}, Windows applied 0x{applied_mask.value:X}")
            applied_priority = k32.GetPriorityClass(h)
            if not applied_priority:
                raise ctypes.WinError()
            print(
                f"[env] scheduling applied reason={reason} pid={self._process.pid} "
                f"priority=0x{applied_priority:X} "
                f"affinity={f'0x{applied_mask.value:X}' if applied_mask is not None else 'unchanged'}",
                flush=True,
            )
        finally:
            k32.CloseHandle(h)

    def apply_engine_scheduling(self, priority: str | None, affinity: str | None, reason: str) -> None:
        """Move a live engine between active and standby scheduling roles."""
        self.engine_priority = priority
        self.engine_affinity = affinity
        try:
            self._apply_windows_scheduling(priority, affinity, reason)
        except Exception as exc:  # scheduling is optional; gameplay must continue
            print(f"[env] scheduling role {reason} not applied for {self.shm_name}: {exc}", flush=True)

    def _fail_launch(self, message: str) -> None:
        """Clean a failed launch while retaining its small diagnostic log."""
        log_path = self._write_dir.parent / f"{self._write_dir.name}.log"
        log_bytes = None
        try:
            self._log_file.flush()
            log_bytes = log_path.read_bytes()
        except OSError:
            pass
        try:
            self.close()
        except Exception as cleanup_error:  # noqa: BLE001 - preserve the launch error
            print(f"[env] launch cleanup failed for {self.shm_name}: {cleanup_error}", flush=True)
        if log_bytes:
            safe_name = "".join(
                ch if ch.isalnum() or ch in "-_" else "_"
                for ch in self.shm_name.strip("/")
            )
            failure_dir = self.repo_root / "Tools" / "rl" / "runs" / "engine_failures"
            failure_dir.mkdir(parents=True, exist_ok=True)
            failure_log = failure_dir / f"{safe_name}.log"
            if failure_log.exists():
                failure_log = failure_dir / f"{safe_name}_{time.time_ns()}.log"
            try:
                failure_log.write_bytes(log_bytes)
                message = f"{message}; engine startup log preserved at {failure_log}"
            except OSError as log_error:
                print(f"[env] could not preserve failed launch log: {log_error}", flush=True)
        raise RuntimeError(message)

    def close(self) -> None:
        if self._shm is not None:
            try:
                self._shm.request_shutdown()
            except OSError:
                pass
        if self._process is not None:
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)
        if self._shm is not None:
            self._shm.close()
        self._log_file.close()
        # Per OGRL-20260815-034 (the disk-full incident): never leave a
        # write-dir behind. Best-effort -- a failed cleanup here shouldn't
        # mask the real close() outcome. The log file is a SIBLING of
        # _write_dir (see _launch()'s log_path), not inside it -- found
        # 2026-08-17 that this was never being removed even on a clean
        # exit, leaking one file per env launch forever regardless of how
        # cleanly the process closed; _cleanup_stale_write_dirs() above
        # catches anything that still slips past this (an abrupt kill).
        import shutil
        if not self.keep_artifacts:
            shutil.rmtree(self._write_dir, ignore_errors=True)
            log_path = self._write_dir.parent / f"{self._write_dir.name}.log"
            log_path.unlink(missing_ok=True)

    def __enter__(self) -> "OvergrowthEnv":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    # --- Gym-shaped API ---

    @property
    def observation_dim(self) -> int:
        return self.layout.total_floats * self.frame_stack

    def _stacked(self, values: list) -> np.ndarray:
        frame = np.asarray(values, dtype=np.float32)
        if self._prealloc_frame_stack:
            frame = frame.reshape(self.layout.total_floats)
            if not self._frame_stack_initialized:
                self._frame_stack_array[...] = frame
                self._frame_stack_initialized = True
            else:
                if self.frame_stack > 1:
                    np.copyto(self._frame_stack_array[:-1], self._frame_stack_array[1:])
                np.copyto(self._frame_stack_array[-1], frame)
            # Return an owned array just like np.concatenate did; callers may
            # retain observations after the next engine step mutates the ring.
            return self._frame_stack_array.reshape(-1).copy()
        self._frame_stack_buffer.append(frame)
        while len(self._frame_stack_buffer) < self.frame_stack:
            # Startup padding: repeat the first frame rather than zero-fill,
            # so the network's very first observation isn't a discontinuous
            # mix of a real frame and zeros it will never see again mid-episode.
            self._frame_stack_buffer.appendleft(self._frame_stack_buffer[0])
        return np.concatenate(list(self._frame_stack_buffer))  # oldest first, newest last

    def reset(
        self,
        seed: int | None = None,
        soft: bool = False,
        difficulty: float | None = None,
        opponents: int = 1,
        weapons: float = 0.0,
        species: int = 0,
        armed_count: int = 0,
        weapon_type: int = 0,
        throw_aggression: float = 1.0,
    ) -> np.ndarray:
        """Reset the requested scenario, including on a fresh engine.

        The first call drains the engine's natural post-load observation before
        sending the reset request. The returned observation therefore always
        belongs to the caller's requested seed and scenario.
        """
        if not self._used_initial_observation:
            self._shm.wait_for_observation()
            self._used_initial_observation = True
        reset_seed = seed if seed is not None else self.seed
        reset_start = time.perf_counter()
        obs = self._shm.reset(reset_seed, soft=soft, difficulty=difficulty, opponents=opponents, weapons=weapons, species=species,
                          armed_count=armed_count, weapon_type=weapon_type, throw_aggression=throw_aggression)
        self.last_reset_seconds = time.perf_counter() - reset_start  # OGRL-20260817-028 Sec8.2: perf.reset_seconds source
        self.episode_count += 1
        self.last_reset_seed = reset_seed
        self._prev_values = obs.values
        self._episode_steps = 0
        if self._prealloc_frame_stack:
            self._frame_stack_initialized = False
        else:
            self._frame_stack_buffer.clear()
        self.reward_computer.reset_episode()  # clears the stall-tax streak (OGRL-20260816-018) -- otherwise a
                                                # stall run from the tail of one episode taxes the start of the next
        return self._stacked(obs.values)

    def write_action(self, action: np.ndarray) -> None:
        """Publish an action without waiting for its observation.

        VecOvergrowthEnv uses this only behind the Windows batch-wait feature
        flag. Keeping the conversion here preserves the exact action wire
        format of the ordinary step path.
        """
        action = np.asarray(action, dtype=np.float32).reshape(ACTION_DIM)
        move_x, move_y = float(action[0]), float(action[1])
        buttons = [bool(v > 0.5) for v in action[2:8]]
        self._shm.write_action(move_x, move_y, buttons[0], buttons[1], buttons[2], buttons[3], buttons[4], buttons[5])

    def step(self, action: np.ndarray, action_already_written: bool = False) -> tuple[np.ndarray, float, bool, dict]:
        if not action_already_written:
            self.write_action(action)
        _tw = time.perf_counter()
        obs = self._shm.wait_for_observation()
        if not action_already_written:
            self.last_wait_seconds = time.perf_counter() - _tw   # engine sim + IPC + scheduling, no Python
        self._episode_steps += 1

        reward, reward_info = self.reward_computer.compute(self._prev_values, obs.values)
        self._prev_values = obs.values

        info = {"reward_components": reward_info, "episode_steps": self._episode_steps, "engine_step": obs.step}
        return self._stacked(obs.values), reward, obs.done, info

    def set_reward_config(self, reward_config: RewardConfig) -> None:
        """Lets a curriculum change reward weights mid-training without
        reconnecting -- RewardComputer is stateless besides its config."""
        self.reward_computer.config = reward_config
