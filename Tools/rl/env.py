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
import subprocess
import tempfile
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

import noaslr
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
    ):
        # act_period (OGRL-20260816-021 Sec 1.3(a)/2.2(a), Stage 6): decision
        # rate divisor -- 1 = every physics tick is a decision (120Hz, the
        # original/default), 4 = every 4th tick (30Hz, matching vanilla AI's
        # own control period). See rl_shm_transport.cpp's Step() for what
        # actually happens on the engine side; this is just the CLI plumbing.
        self.act_period = act_period
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
        self._frame_stack_buffer: deque = deque(maxlen=self.frame_stack)
        self.binary_path = Path(binary_path) if binary_path else self.repo_root / "BuildArm64/Overgrowth.app/Contents/MacOS/Overgrowth"
        self._launch_timeout = launch_timeout_seconds

        write_dir_parent = Path(write_dir_parent) if write_dir_parent else self.repo_root / ".rl_write_dirs"
        write_dir_parent.mkdir(parents=True, exist_ok=True)
        _cleanup_stale_write_dirs(write_dir_parent)
        self._write_dir = Path(tempfile.mkdtemp(prefix=f"env-{shm_name.strip('/')}-", dir=write_dir_parent))

        self._process: subprocess.Popen | None = None
        self._shm: ShmEnv | None = None
        self._prev_values: list | None = None
        self._episode_steps = 0
        # Engine::ResetRLTrainingScenario requires its baseline to have been
        # captured, which only happens once the engine's own *initial* level
        # load (started at engine launch, independent of when this client
        # happens to connect) fully completes -- shm segment creation
        # (RLShmTransport::Configure) runs during CLI arg processing, well
        # before that load finishes, so a client that connects quickly and
        # immediately calls reset() races the baseline capture and reliably
        # loses. The initial level load already produces a fresh, valid
        # starting state on its own -- no engine-side reset is needed to use
        # it -- so the first reset() call just consumes that natural first
        # observation instead of requesting a redundant, racy one.
        self._used_initial_observation = False
        # episode_count: real resets only (the pseudo-reset that consumes the
        # engine's own initial observation, above, doesn't count) -- this is
        # what --hard-reset-every gates on (OGRL-20260817-028 Sec1.2), so it
        # deliberately lives on the physical engine process, not on whatever
        # vector slot currently happens to be playing it (a standby moves
        # between slots over its lifetime -- see vec_env.py).
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
        config_str = "\n".join([
            f"global_time_scale_mult: {self.time_scale_mult}",
            "skip_loading_pause: true",
            "has_detected_settings: true",
        ])
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
        self._process = subprocess.Popen(command, cwd=self.repo_root, stdout=self._log_file, stderr=subprocess.STDOUT)

        deadline = time.monotonic() + self._launch_timeout
        last_error = None
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise RuntimeError(
                    f"engine process exited early (code {self._process.returncode}) while connecting to {self.shm_name} -- see {log_path}"
                )
            try:
                self._shm = ShmEnv(self.shm_name, obs_floats=self.layout.total_floats, connect_retries=1, connect_retry_delay=0.0)
                break
            except OSError as exc:
                last_error = exc
                time.sleep(0.2)
        if self._shm is None:
            raise RuntimeError(f"timed out connecting to {self.shm_name} after {self._launch_timeout}s: {last_error}")

        if self._shm.obs_floats != self.layout.total_floats:
            raise ValueError(f"obs_floats mismatch: engine publishes {self._shm.obs_floats}, this layout expects {self.layout.total_floats}")
        if self._shm.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version mismatch: engine publishes {self._shm.schema_version}, obs_schema.py expects {SCHEMA_VERSION} -- rebuild the engine or update obs_schema.py")

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
    ) -> np.ndarray:
        """soft/difficulty/opponents/weapons/species are the OGRL-20260817-028
        Sec1/Sec3.1 curriculum hook -- forwarded straight to ShmEnv.reset(),
        see its docstring for exact semantics. Not applied to the very first
        reset() of a freshly-launched engine (see _used_initial_observation's
        comment in __init__): that one just consumes the natural first
        observation from the engine's own initial level load, which the
        level script drives with its own default difficulty, not the
        RL-requested one -- a one-episode-per-worker-lifetime cold-start
        artifact, unchanged from before this hook existed and not worth a
        redundant extra reset just to eliminate."""
        if not self._used_initial_observation:
            self._used_initial_observation = True
            obs = self._shm.wait_for_observation()
            self.last_reset_seconds = 0.0  # not a real reset -- see _used_initial_observation's comment
        else:
            reset_seed = seed if seed is not None else self.seed
            reset_start = time.monotonic()
            obs = self._shm.reset(reset_seed, soft=soft, difficulty=difficulty, opponents=opponents, weapons=weapons, species=species)
            self.last_reset_seconds = time.monotonic() - reset_start  # OGRL-20260817-028 Sec8.2: perf.reset_seconds source
            self.episode_count += 1
            self.last_reset_seed = reset_seed
        self._prev_values = obs.values
        self._episode_steps = 0
        self._frame_stack_buffer.clear()
        self.reward_computer.reset_episode()  # clears the stall-tax streak (OGRL-20260816-018) -- otherwise a
                                                # stall run from the tail of one episode taxes the start of the next
        return self._stacked(obs.values)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        action = np.asarray(action, dtype=np.float32).reshape(ACTION_DIM)
        move_x, move_y = float(action[0]), float(action[1])
        buttons = [bool(v > 0.5) for v in action[2:8]]
        self._shm.write_action(move_x, move_y, buttons[0], buttons[1], buttons[2], buttons[3], buttons[4], buttons[5])
        obs = self._shm.wait_for_observation()
        self._episode_steps += 1

        reward, reward_info = self.reward_computer.compute(self._prev_values, obs.values)
        self._prev_values = obs.values

        info = {"reward_components": reward_info, "episode_steps": self._episode_steps, "engine_step": obs.step}
        return self._stacked(obs.values), reward, obs.done, info

    def set_reward_config(self, reward_config: RewardConfig) -> None:
        """Lets a curriculum change reward weights mid-training without
        reconnecting -- RewardComputer is stateless besides its config."""
        self.reward_computer.config = reward_config
