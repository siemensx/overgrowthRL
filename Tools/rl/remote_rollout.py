#!/usr/bin/env python3
"""Distributed on-policy rollout collection: remote workers, one learner.

The learner owns the policy and both normalisers and performs every PPO update.
A worker owns engines only. Each update the learner broadcasts the current
policy and observation-normaliser state, every participant collects exactly
`n_steps` transitions for its own envs, and the learner concatenates them along
the ENV axis before computing GAE.

Why concatenating along the env axis is exact rather than an approximation:
`VecRolloutBuffer` stores (n_steps, n_envs, ...) and `compute_gae` runs the
recursion over time independently for each env column (see its docstring). A
remote worker's envs are therefore just additional columns; no trajectory is
split across machines and no term crosses columns.

Why this stays ON-POLICY: the learner does not update until every participant
has returned its full rollout, and each rollout was collected with exactly the
weights broadcast at the start of that update. There is no staleness, and no
V-trace or importance correction is needed. The cost is that the learner waits
for the slowest participant -- which is why env counts must be assigned
PROPORTIONALLY to measured speed. Equal splitting across a fast and a slow
machine is worse than running the fast machine alone: measured 2026-09-05, the
M4 sustains ~1900 sps and the trainer ~460, so an equal split would run at the
trainer's pace for both halves.

Rewards are sent RAW. The learner holds the reward normaliser (whose running
return is per-env) and normalises the combined batch itself, so normalisation
sees one consistent stream rather than diverging per machine.

Protocol: length-prefixed pickle over TCP. LAN only, no authentication -- do not
expose the port beyond a trusted network.
"""
from __future__ import annotations

import pickle
import socket
import struct
from typing import Any

HEADER = struct.Struct("!Q")


def send_msg(sock: socket.socket, obj: Any) -> None:
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(HEADER.pack(len(payload)))
    sock.sendall(payload)


def recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(1 << 20, n - len(buf)))
        if not chunk:
            raise ConnectionError(f"peer closed after {len(buf)} of {n} bytes")
        buf += chunk
    return bytes(buf)


def recv_msg(sock: socket.socket) -> Any:
    (n,) = HEADER.unpack(recv_exact(sock, HEADER.size))
    return pickle.loads(recv_exact(sock, n))


def configure_socket(sock: socket.socket) -> None:
    """TCP_NODELAY matters here: the messages are a few MB and latency-sensitive
    once per update, and Nagle's algorithm would add delay for no benefit."""
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
