"""Stage 5.4 (+ reset extension): Python-side client for RLShmTransport's
shared-memory transport (Source/Main/rl_shm_transport.{h,cpp}).

Deliberately zero third-party dependencies (no posix_ipc, no nanobind on this
side per the plan) -- everything here is stdlib `mmap`/`ctypes` bindings to
the same libc calls the C++ side uses (shm_open, sem_open/sem_post/sem_wait),
so the two sides are two independent implementations of one shared protocol
rather than one wrapping the other. That also sidesteps a real compatibility
risk: Python's own multiprocessing.shared_memory has its own naming/creation
conventions that aren't guaranteed to match a raw C shm_open() caller.

Protocol (see rl_shm_transport.h for the authoritative description): lock-step
request/response. The engine posts the obs semaphore after writing this
step's observation, then blocks on the action semaphore. This client must
always follow wait_for_observation() with exactly one of write_action() or
reset(seed) before the next wait_for_observation() call, or the engine stalls
(correctly -- that coupling is the whole point of a synchronous
env.step()-shaped transport). reset() already does its own follow-up
wait_for_observation() internally and returns the fresh Observation directly,
matching Gym's env.reset() -> obs contract -- callers should not call
wait_for_observation() again themselves after reset().
"""

from __future__ import annotations

import ctypes
import ctypes.util
import math
import mmap
import os
import struct
import numpy as np
import time
from dataclasses import dataclass

# Platform backend. The mapped wire format is byte-identical everywhere; only
# the naming and the handle types differ. The engine side of this split lives
# in Source/Main/rl_ipc_platform.{h,cpp} and the two MUST agree on the object
# names, or the client attaches to nothing and blocks forever.
_IS_WINDOWS = os.name == "nt"

if _IS_WINDOWS:
    from ctypes import wintypes

    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)

    _k32.OpenFileMappingW.restype = wintypes.HANDLE
    _k32.OpenFileMappingW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
    _k32.OpenSemaphoreW.restype = wintypes.HANDLE
    _k32.OpenSemaphoreW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
    _k32.WaitForSingleObject.restype = wintypes.DWORD
    _k32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    _k32.WaitForMultipleObjects.restype = wintypes.DWORD
    _k32.WaitForMultipleObjects.argtypes = [
        wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE), wintypes.BOOL, wintypes.DWORD
    ]
    _k32.ReleaseSemaphore.restype = wintypes.BOOL
    _k32.ReleaseSemaphore.argtypes = [wintypes.HANDLE, ctypes.c_long, ctypes.POINTER(ctypes.c_long)]
    _k32.CloseHandle.argtypes = [wintypes.HANDLE]

    _FILE_MAP_ALL_ACCESS = 0x000F001F
    _SEMAPHORE_ALL_ACCESS = 0x1F0003
    _INFINITE = 0xFFFFFFFF
    _SEM_FAILED = None

    def _object_name(posix_name: str) -> str:
        """/ogrl_vec0 -> Local\\ogrl_vec0. Must match RLIpc::ToObjectName."""
        return "Local\\" + posix_name.lstrip("/")

    def _shm_probe(name: str):
        """Return an opaque handle if the engine's mapping exists, else None.

        Deliberately does NOT use mmap(tagname=...) to test existence: on
        Windows that CREATES a pagefile-backed mapping when none exists, so a
        client started before the engine would silently attach to its own
        zero-filled segment instead of failing and retrying.
        """
        h = _k32.OpenFileMappingW(_FILE_MAP_ALL_ACCESS, False, _object_name(name))
        return h if h else None

    def _shm_map(handle, name: str, size: int) -> mmap.mmap:
        # The named section already exists (handle proves it), so this opens
        # rather than creates.
        return mmap.mmap(-1, size, tagname=_object_name(name))

    def _shm_close(handle) -> None:
        if handle:
            _k32.CloseHandle(handle)

    def _sem_open(name: str):
        h = _k32.OpenSemaphoreW(_SEMAPHORE_ALL_ACCESS, False, _object_name(name))
        return h if h else None

    _WAIT_TIMEOUT = 0x00000102
    _WAIT_OBJECT_0 = 0x00000000
    _WAIT_FAILED = 0xFFFFFFFF
    _MAXIMUM_WAIT_OBJECTS = 64

    # ctypes without argtypes/restype re-derives conversions on every call;
    # these two run 2x per engine step (2026-09-20 profile: _sem_post 54 s / 100 updates).
    try:
        _k32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        _k32.WaitForSingleObject.restype = ctypes.c_uint32
        _k32.ReleaseSemaphore.argtypes = [ctypes.c_void_p, ctypes.c_long, ctypes.c_void_p]
        _k32.ReleaseSemaphore.restype = ctypes.c_int
    except Exception:
        pass

    def _sem_wait(sem, timeout_s: float | None = None) -> bool:
        """True if acquired, False on timeout. None means wait forever."""
        ms = _INFINITE if timeout_s is None else max(0, int(timeout_s * 1000))
        result = _k32.WaitForSingleObject(sem, ms)
        if result == _WAIT_OBJECT_0:
            return True
        if result == _WAIT_TIMEOUT:
            return False
        if result == _WAIT_FAILED:
            raise ctypes.WinError(ctypes.get_last_error())
        raise RuntimeError(f"WaitForSingleObject returned unexpected status 0x{result:X}")

    def _sem_wait_many(sems, timeout_s: float | None = None) -> tuple[list[int], list[int]]:
        """Wait on many semaphore handles with one Windows wait set.

        Wait-any is repeated until every handle has been acquired. This keeps
        the protocol identical to N independent WaitForSingleObject calls --
        each semaphore is consumed exactly once -- while moving the wakeup
        arbitration into the kernel. The returned indices are acquired and
        still-pending; a timeout is therefore recoverable per worker by the
        existing VecOvergrowthEnv path.
        """
        if len(sems) > _MAXIMUM_WAIT_OBJECTS:
            raise ValueError(f"Windows wait set has {len(sems)} handles; maximum is {_MAXIMUM_WAIT_OBJECTS}")
        pending = list(range(len(sems)))
        acquired: list[int] = []
        deadline = None if timeout_s is None else time.perf_counter() + timeout_s
        while pending:
            if deadline is None:
                timeout_ms = _INFINITE
            else:
                timeout_ms = max(0, int(max(0.0, deadline - time.perf_counter()) * 1000))
            handles = (wintypes.HANDLE * len(pending))(*[
                int(sems[i].value) if hasattr(sems[i], "value") else int(sems[i])
                for i in pending
            ])
            result = _k32.WaitForMultipleObjects(len(pending), handles, False, timeout_ms)
            if _WAIT_OBJECT_0 <= result < _WAIT_OBJECT_0 + len(pending):
                acquired.append(pending.pop(result - _WAIT_OBJECT_0))
                continue
            if result == _WAIT_TIMEOUT:
                break
            if result == _WAIT_FAILED:
                raise ctypes.WinError(ctypes.get_last_error())
            raise RuntimeError(f"WaitForMultipleObjects returned unexpected status 0x{result:X}")
        return acquired, pending

    def _sem_post(sem) -> None:
        _k32.ReleaseSemaphore(sem, 1, None)

    def _sem_close(sem) -> None:
        if sem:
            _k32.CloseHandle(sem)

    def _last_error() -> int:
        return ctypes.get_last_error()

else:
    _libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)

    _libc.shm_open.restype = ctypes.c_int
    _libc.shm_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_uint]
    _libc.close.argtypes = [ctypes.c_int]

    _SEM_T_P = ctypes.c_void_p
    _libc.sem_open.restype = _SEM_T_P
    _libc.sem_open.argtypes = [ctypes.c_char_p, ctypes.c_int]  # only the O_RDWR-no-create form is used here
    _libc.sem_wait.argtypes = [_SEM_T_P]
    _libc.sem_trywait.restype = ctypes.c_int
    _libc.sem_trywait.argtypes = [_SEM_T_P]
    _libc.sem_post.argtypes = [_SEM_T_P]
    _libc.sem_close.argtypes = [_SEM_T_P]

    _O_RDWR = os.O_RDWR
    _SEM_FAILED = ctypes.cast(-1, _SEM_T_P).value

    def _shm_probe(name: str):
        fd = _libc.shm_open(name.encode("utf-8"), _O_RDWR, 0o600)
        return fd if fd >= 0 else None

    def _shm_map(handle, name: str, size: int) -> mmap.mmap:
        return mmap.mmap(handle, size)

    def _shm_close(handle) -> None:
        if handle is not None and handle >= 0:
            os.close(handle)

    def _sem_open(name: str):
        # sem_open's oflag only recognizes O_CREAT (+ optionally O_EXCL); 0
        # means "open the existing named semaphore," which is what a client
        # attaching to an engine-created semaphore wants -- O_RDWR is a
        # shm_open/file-open concept, not a valid sem_open flag.
        s = _libc.sem_open(name.encode("utf-8"), 0)
        return None if s == _SEM_FAILED else s

    def _sem_wait(sem, timeout_s: float | None = None) -> bool:
        """True if acquired, False on timeout. None means wait forever.

        macOS does not implement sem_timedwait (it is declared but always fails
        with ENOSYS), so a bounded wait has to poll sem_trywait. The sleep is
        short enough not to add meaningful latency to a ~1ms step and long
        enough not to spin a core.
        """
        if timeout_s is None:
            _libc.sem_wait(sem)
            return True
        # Adaptive backoff. A step normally completes in well under a
        # millisecond, so sleeping a fixed 0.5ms per wait would dominate the
        # step time and roughly halve throughput. Spin-yield first (covers the
        # overwhelming majority of waits at no latency cost), then back off so a
        # genuinely stalled engine costs no CPU while we count down to the
        # deadline.
        if _libc.sem_trywait(sem) == 0:
            return True
        deadline = time.perf_counter() + timeout_s
        spins = 0
        while True:
            if _libc.sem_trywait(sem) == 0:
                return True
            if time.perf_counter() >= deadline:
                return False
            spins += 1
            if spins < 2000:
                time.sleep(0)          # yield, no timer
            elif spins < 4000:
                time.sleep(0.0002)
            else:
                time.sleep(0.005)

    def _sem_wait_many(sems, timeout_s: float | None = None) -> tuple[list[int], list[int]]:
        """Portable fallback used only by tests; Windows owns the optimization."""
        acquired: list[int] = []
        pending: list[int] = []
        for i, sem in enumerate(sems):
            if _sem_wait(sem, timeout_s):
                acquired.append(i)
            else:
                pending.append(i)
        return acquired, pending

    def _sem_post(sem) -> None:
        _libc.sem_post(sem)

    def _sem_close(sem) -> None:
        if sem is not None:
            _libc.sem_close(sem)

    def _last_error() -> int:
        return ctypes.get_errno()

# Must match Source/Main/rl_shm_transport.cpp's ShmHeader exactly, field for
# field IN ORDER: sixteen packed fields (eleven uint32_t + five added by
# OGRL-20260817-028 Sec1/Sec3.1 for soft reset + the per-episode curriculum
# hook: reset_mode:u32, reset_difficulty:f32, reset_opponents:u32,
# reset_weapons:f32, reset_species:u32), no padding (the C++ side uses
# #pragma pack(1)). (name, struct type char) pairs, not just a format string,
# so field lookups can't repeat the OGRL-20260816-009 off-by-one bug (wrote
# shutdown_requested to the wrong index by miscounting a positional tuple),
# and so a mixed-type header (this one has both I and f fields, the original
# all-uint32 layout didn't) can't silently mis-pack a float as an int either.
_HEADER_FIELD_TYPES = [
    ("magic", "I"), ("schema_version", "I"), ("los_rule_version", "I"),
    ("obs_floats", "I"), ("action_floats", "I"),
    ("episode_done", "I"), ("shutdown_requested", "I"), ("step_counter", "I"),
    ("reset_requested", "I"), ("reset_seed", "I"), ("reset_ok", "I"),
    ("reset_mode", "I"), ("reset_difficulty", "f"), ("reset_opponents", "I"),
    ("reset_weapons", "f"), ("reset_species", "I"),
    ("reset_armed_count", "I"), ("reset_weapon_type", "I"), ("reset_throw_aggression", "f"),
]
_HEADER_FIELDS = [name for name, _ in _HEADER_FIELD_TYPES]
_HEADER_FORMAT = "<" + "".join(t for _, t in _HEADER_FIELD_TYPES)
_HEADER_SIZE = struct.calcsize(_HEADER_FORMAT)
assert _HEADER_SIZE == 76, _HEADER_SIZE  # 64 + armed_count/weapon_type/throw_aggression
_MAGIC = 0x4C524730

# reset_mode values (rl_shm_transport.cpp's ShmHeader.reset_mode)
RESET_HARD = 0
RESET_SOFT = 1

_ACTION_FLOATS = 8  # move_x, move_y, jump, crouch, attack, grab, drop, walk


# How long a worker may go silent before it is declared dead and rebuilt.
# Generous by default: a hard reset with a cold navmesh bake legitimately takes
# seconds, and a false positive costs a restarted episode. Override for tests.
_DEFAULT_WAIT_TIMEOUT = float(os.environ.get("OGRL_SHM_WAIT_TIMEOUT", "120"))


class ShmWaitTimeout(TimeoutError):
    """The engine did not publish within the deadline.

    Raised instead of blocking forever. On 2026-09-05 a run deadlocked with the
    trainer parked in sem_wait and every engine idle at 0% CPU, waiting for an
    action that could never arrive because the trainer was waiting for them --
    an unbreakable cycle that a bounded wait turns into a recoverable error.
    """


def _unpack_header(raw: bytes) -> dict:
    return dict(zip(_HEADER_FIELDS, struct.unpack(_HEADER_FORMAT, raw)))


def _pack_header(fields: dict) -> bytes:
    return struct.pack(_HEADER_FORMAT, *(fields[name] for name in _HEADER_FIELDS))


non_finite_observation_count = 0  # module-level, process-wide -- see wait_for_observation()'s sanitization comment


@dataclass
class Observation:
    step: int
    done: bool
    values: object  # owned ndarray by default; list when the compatibility gate is disabled


class ShmEnv:
    """Attaches to a shm segment an already-running engine process created via
    --rl-shm-name <name> --rl-action-controller-id <id>. The engine must be
    started first (it owns segment/semaphore creation); this class only
    attaches to an existing segment, it never creates one -- a client racing
    the engine to create the segment would be a protocol violation, not a
    convenience.
    """

    def __init__(self, name: str, obs_floats: int, connect_retries: int = 50, connect_retry_delay: float = 0.1):
        self._name = name
        self._obs_floats_hint = obs_floats
        self._fd = -1
        self._mm: mmap.mmap | None = None
        self._obs_sem = None
        self._action_sem = None
        self._observation_ready = False
        self._connect(connect_retries, connect_retry_delay)

    def _connect(self, retries: int, delay: float) -> None:
        # OGRL-20260816-023: `magic` alone is not sufficient to prove this
        # header is FROM THIS ENGINE PROCESS'S Configure() call -- it's a
        # fixed constant, unchanged across every schema version, so a client
        # that reads between the engine's shm_open(O_CREAT) and its own
        # memset()+field-write sequence can observe a technically-valid
        # magic left over from whatever PREVIOUS process last wrote this
        # (unlinked-and-recreated, same name) segment, paired with STALE
        # obs_floats/schema_version from that earlier writer -- not zeroed
        # garbage, which the magic check would catch, but a plausible-looking
        # wrong answer. Reproduced live: a 4-engine parallel launch (Configure()
        # calls racing each other and 4 client connects) once read
        # obs_floats=243 (an old schema's size) against a binary that reports
        # 260 in isolation. Caught here by re-reading the header after a
        # short settle delay and requiring it to be UNCHANGED -- a header
        # still being written won't stay stable across two reads a beat
        # apart; a genuinely finished one will.
        name_bytes = self._name.encode("utf-8")
        last_err = None
        header = None
        for attempt in range(retries):
            fd = _shm_probe(self._name)
            if fd is None:
                last_err = _last_error()
                time.sleep(delay)
                continue
            header_probe = _shm_map(fd, self._name, _HEADER_SIZE)
            first_read = _unpack_header(header_probe[:_HEADER_SIZE])
            if first_read["magic"] != _MAGIC:
                header_probe.close()
                _shm_close(fd)
                last_err = None
                time.sleep(delay)
                continue
            time.sleep(min(0.05, max(delay, 0.01)))  # settle window for the race above
            second_read = _unpack_header(header_probe[:_HEADER_SIZE])
            header_probe.close()
            if second_read != first_read:
                # Header was still being written -- not a real connection
                # failure, just not ready yet. Retry from scratch rather than
                # trusting either read.
                _shm_close(fd)
                last_err = None
                time.sleep(delay)
                continue
            self._fd = fd
            header = second_read
            break
        else:
            raise OSError(last_err, f"shm_open({self._name}) failed after {retries} retries -- is the engine running with --rl-shm-name {self._name}?")

        if header["action_floats"] != _ACTION_FLOATS:
            raise ValueError(f"action_floats mismatch: engine says {header['action_floats']}, client expects {_ACTION_FLOATS}")

        self.schema_version = header["schema_version"]
        self.los_rule_version = header["los_rule_version"]
        self.obs_floats = header["obs_floats"]
        total_size = _HEADER_SIZE + self.obs_floats * 4 + _ACTION_FLOATS * 4
        self._mm = _shm_map(self._fd, self._name, total_size)

        self._obs_sem = _sem_open(self._name + "o")
        self._action_sem = _sem_open(self._name + "a")
        if self._obs_sem is None or self._action_sem is None:
            raise OSError(_last_error(), f"semaphore open failed for {self._name}")

    def wait_for_observation(self, timeout_s: float | None = None) -> Observation:
        """Wait for the engine to publish an observation.

        Bounded by default. An unbounded wait here is what turned a single lost
        wakeup into a permanent deadlock on 2026-09-05: the trainer sat in
        sem_wait at 0% CPU for three hours and every health metric -- throughput,
        win rate, engine count -- looked normal because nothing was crashing.
        Pass timeout_s=None only where blocking forever is genuinely correct.
        """
        if timeout_s is None:
            timeout_s = _DEFAULT_WAIT_TIMEOUT
        if self._observation_ready:
            # VecOvergrowthEnv's Windows batch-wait path already consumed this
            # semaphore. Keep all observation decoding and validation below in
            # one place so the optimization cannot change wire semantics.
            self._observation_ready = False
        elif not _sem_wait(self._obs_sem, timeout_s):
            raise ShmWaitTimeout(
                f"engine {self._name} published no observation within {timeout_s}s. "
                f"It is alive but silent, or has stopped stepping; the caller should "
                f"restart this worker rather than wait."
            )
        header = _unpack_header(self._mm[:_HEADER_SIZE])
        obs_bytes = self._mm[_HEADER_SIZE : _HEADER_SIZE + header["obs_floats"] * 4]
        # 2026-09-20: numpy at the source. struct.unpack + a pure-Python
        # any(isfinite) over 1356 floats cost ~14% of a training update (123M
        # math.isfinite calls per 100 updates in the profile). One frombuffer
        # and one vectorised isfinite do the same work in microseconds; the
        # list interface downstream consumers expect is preserved.
        arr = np.frombuffer(obs_bytes, dtype="<f4")
        # NaN/Inf sanitization (2026-08-17, found live during the first
        # cold-start smoke test): a rare non-finite raw observation value --
        # root cause not isolated under time pressure, but the timing
        # (both crashes happened only once the policy was already fighting
        # well, i.e. after real knockdowns/ragdolls started happening, not
        # during quiet early exploration) is consistent with some physics-
        # extreme quantity (a velocity/distance computation during a violent
        # hit) occasionally producing a non-finite float somewhere in
        # RLObservation::Extract(), rather than a Python-side numerical bug
        # -- reached the policy's continuous head raw (`torch.distributions.
        # Normal` rejects a NaN loc outright) and crashed two independent
        # multi-hundred-thousand-decision runs. This is the single choke
        # point every observation passes through (both wait_for_observation()
        # callers and reset(), which calls this internally) -- replacing a
        # non-finite value with 0.0 here is a defensive boundary check, not
        # a fix for wherever the value actually originates; if this fires
        # often in metrics.jsonl/dashboard, that IS the follow-up to
        # investigate, not something to ignore because it stopped crashing.
        if not np.isfinite(arr).all():
            global non_finite_observation_count
            non_finite_observation_count += 1
            arr = np.where(np.isfinite(arr), arr, 0.0).astype("<f4")
        # The old compatibility path converted the ndarray to a Python list,
        # then env.py immediately converted it back with np.asarray(). That
        # allocates thousands of Python float objects per decision and was
        # directly identified as an advisor-ranked safe lever. The copy is
        # essential: the frombuffer view aliases shared memory that the engine
        # overwrites on its next publish. Set OGRL_SHM_ARRAY_FASTPATH=0 only
        # for an explicit compatibility comparison.
        if os.environ.get("OGRL_SHM_ARRAY_FASTPATH", "0") != "0":
            values = arr.copy()
        else:
            values = arr.tolist()
        return Observation(step=header["step_counter"], done=bool(header["episode_done"]), values=values)

    def write_action(self, move_x: float, move_y: float, jump: bool, crouch: bool, attack: bool, grab: bool, drop: bool = False, walk: bool = False) -> None:
        """Publishes the next action and wakes the engine to continue stepping.
        Must be called exactly once per wait_for_observation() call."""
        action = struct.pack(
            f"<{_ACTION_FLOATS}f",
            float(move_x),
            float(move_y),
            1.0 if jump else 0.0,
            1.0 if crouch else 0.0,
            1.0 if attack else 0.0,
            1.0 if grab else 0.0,
            1.0 if drop else 0.0,
            1.0 if walk else 0.0,
        )
        offset = _HEADER_SIZE + self.obs_floats * 4
        self._mm[offset : offset + len(action)] = action
        _sem_post(self._action_sem)

    def reset(
        self,
        seed: int,
        soft: bool = False,
        difficulty: float | None = None,
        opponents: int = 1,
        weapons: float = 0.0,
        species: int = 0,
        armed_count: int = 0,
        weapon_type: int = 0,
        throw_aggression: float = 1.0,
    ) -> Observation:
        """Resets the training scenario in-process and returns the fresh
        post-reset observation, matching Gym's env.reset() -> obs contract.
        Call this instead of write_action() when the previous observation had
        done=True (or to force an early reset). Raises RuntimeError if the
        engine reports the reset failed (e.g. mid cutscene/campaign state) --
        the scenario is then still whatever it was before this call, and the
        caller should not assume a fresh episode.

        soft=False (default) uses Engine::ResetRLTrainingScenario, the
        original full ClearLoadedLevel+LoadLevel path (~575ms measured).
        soft=True uses Engine::SoftResetRLTrainingScenario (OGRL-20260817-028
        Sec1) -- reseed + Level::Message("post_reset"), no level reload.
        Both paths honor difficulty/opponents/weapons/species identically
        (rl_shm_transport.cpp sends the same set_rl_* messages before either
        one's post_reset). difficulty=None leaves rl_difficulty at whatever
        the level script last had (i.e. -1 / "use the script's own default"
        on the very first reset of a fresh engine) rather than forcing 0.0,
        which would silently zero the training difficulty for any caller that
        doesn't pass one explicitly."""
        header = _unpack_header(self._mm[:_HEADER_SIZE])
        header["reset_requested"] = 1
        header["reset_seed"] = seed & 0xFFFFFFFF
        header["reset_mode"] = RESET_SOFT if soft else RESET_HARD
        header["reset_difficulty"] = -1.0 if difficulty is None else float(difficulty)
        header["reset_opponents"] = int(opponents)
        header["reset_weapons"] = float(weapons)
        header["reset_species"] = int(species)
        header["reset_armed_count"] = int(armed_count)
        header["reset_weapon_type"] = int(weapon_type)
        header["reset_throw_aggression"] = float(throw_aggression)
        self._mm[:_HEADER_SIZE] = _pack_header(header)
        _sem_post(self._action_sem)

        obs = self.wait_for_observation()
        reset_ok = _unpack_header(self._mm[:_HEADER_SIZE])["reset_ok"]
        if not reset_ok:
            raise RuntimeError(f"engine reported reset(seed={seed}, soft={soft}) failed -- scenario is unchanged, not a fresh episode")
        return obs

    def request_shutdown(self) -> None:
        """Sets shutdown_requested and wakes the engine one last time so it
        exits gracefully instead of blocking forever on the next obs cycle."""
        header = _unpack_header(self._mm[:_HEADER_SIZE])
        header["shutdown_requested"] = 1
        self._mm[:_HEADER_SIZE] = _pack_header(header)
        _sem_post(self._action_sem)

    def close(self) -> None:
        if self._obs_sem is not None:
            _sem_close(self._obs_sem)
        if self._action_sem is not None:
            _sem_close(self._action_sem)
        if self._mm is not None:
            self._mm.close()
        if self._fd >= 0:
            _shm_close(self._fd)

    def __enter__(self) -> "ShmEnv":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
