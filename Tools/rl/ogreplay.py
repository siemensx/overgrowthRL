"""Versioned, crash-tolerant episode containers for Overgrowth RL.

The legacy ``tapes/*.jsonl`` files are policy-decision summaries.  They are
still useful for old runs, but they are not simulator checkpoints.  This
module is the new artifact boundary: a small JSON manifest plus typed,
checksummed, independently compressed binary chunks.

The codec is deliberately stdlib-only so training and the dashboard do not
acquire a packaging dependency.  zstandard is used when the optional Python
binding is present; zlib is a lossless fallback and is recorded in every
chunk header.  The engine-side recorder can later emit the same container
without changing the dashboard contract.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import time
import uuid
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

MAGIC = b"OGREPL01"
VERSION = 1
ENDIAN_MARKER = 0x0102
HEADER = struct.Struct(">8sHHIQ")
CHUNK_HEADER = struct.Struct(">4sHHQQQQII16s")
FOOTER = struct.Struct(">4sQQ32s")
FOOTER_TAG = b"FOOT"
CHUNK_TYPES = {"RESET", "TICK", "DECISION", "EVENT", "VISUAL", "INDEX"}
CHUNK_CODES = {name: name[:4].encode("ascii") for name in CHUNK_TYPES}
CODE_NAMES = {code: name for name, code in CHUNK_CODES.items()}


@dataclass(frozen=True)
class Float32Bits:
    """A float32 value whose original bits must survive a round trip."""

    bits: int

    @classmethod
    def from_float(cls, value: float) -> "Float32Bits":
        return cls(struct.unpack(">I", struct.pack(">f", value))[0])

    def value(self) -> float:
        return struct.unpack(">f", struct.pack(">I", self.bits & 0xFFFFFFFF))[0]


@dataclass(frozen=True)
class Float64Bits:
    """A float64 value whose original bits must survive a round trip."""

    bits: int

    @classmethod
    def from_float(cls, value: float) -> "Float64Bits":
        return cls(struct.unpack(">Q", struct.pack(">d", value))[0])

    def value(self) -> float:
        return struct.unpack(">d", struct.pack(">Q", self.bits & 0xFFFFFFFFFFFFFFFF))[0]


def _pack_len(length: int) -> bytes:
    if length < 0:
        raise ValueError("negative binary length")
    return struct.pack(">Q", length)


def canonical_pack(value: Any) -> bytes:
    """Encode JSON-shaped data with explicit types and IEEE float bits.

    Ordinary Python floats are encoded as float64.  Native engine bindings may
    use ``Float32Bits`` when a field is stored as a float32.  Dict keys are
    sorted by their encoded bytes, not insertion order.
    """
    if value is None:
        return b"n"
    if isinstance(value, bool):
        return b"b" + (b"\x01" if value else b"\x00")
    if isinstance(value, Float32Bits):
        return b"3" + struct.pack(">I", value.bits & 0xFFFFFFFF)
    if isinstance(value, Float64Bits):
        return b"6" + struct.pack(">Q", value.bits & 0xFFFFFFFFFFFFFFFF)
    if isinstance(value, float):
        return b"f" + struct.pack(">d", value)
    if isinstance(value, int) and not isinstance(value, bool):
        if -(1 << 63) <= value < (1 << 63):
            return b"i" + struct.pack(">q", value)
        if 0 <= value < (1 << 64):
            return b"u" + struct.pack(">Q", value)
        raise OverflowError("canonical integer is outside signed/unsigned 64-bit range")
    if isinstance(value, str):
        data = value.encode("utf-8")
        return b"s" + _pack_len(len(data)) + data
    if isinstance(value, (bytes, bytearray, memoryview)):
        data = bytes(value)
        return b"y" + _pack_len(len(data)) + data
    if isinstance(value, (list, tuple)):
        return b"l" + _pack_len(len(value)) + b"".join(canonical_pack(v) for v in value)
    if isinstance(value, Mapping):
        entries = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("canonical mapping keys must be strings")
            encoded_key = canonical_pack(key)
            entries.append((encoded_key, canonical_pack(item)))
        entries.sort(key=lambda pair: pair[0])
        return b"d" + _pack_len(len(entries)) + b"".join(k + v for k, v in entries)
    raise TypeError(f"unsupported canonical value: {type(value).__name__}")


def canonical_unpack(data: bytes, offset: int = 0) -> tuple[Any, int]:
    """Decode one value and return ``(value, next_offset)``."""
    if offset >= len(data):
        raise ValueError("truncated canonical value tag")
    tag = data[offset:offset + 1]
    offset += 1
    if tag == b"n":
        return None, offset
    if tag == b"b":
        if offset >= len(data):
            raise ValueError("truncated boolean")
        return data[offset] != 0, offset + 1
    if tag == b"3":
        end = offset + 4
        if end > len(data):
            raise ValueError("truncated float32")
        return Float32Bits(struct.unpack(">I", data[offset:end])[0]), end
    if tag == b"6":
        end = offset + 8
        if end > len(data):
            raise ValueError("truncated float64 bits")
        return Float64Bits(struct.unpack(">Q", data[offset:end])[0]), end
    if tag == b"f":
        end = offset + 8
        if end > len(data):
            raise ValueError("truncated float64")
        return struct.unpack(">d", data[offset:end])[0], end
    if tag in (b"i", b"u"):
        end = offset + 8
        if end > len(data):
            raise ValueError("truncated integer")
        return struct.unpack(">q" if tag == b"i" else ">Q", data[offset:end])[0], end
    if tag in (b"s", b"y"):
        end_len = offset + 8
        if end_len > len(data):
            raise ValueError("truncated byte-string length")
        length = struct.unpack(">Q", data[offset:end_len])[0]
        end = end_len + length
        if end > len(data):
            raise ValueError("truncated byte-string")
        raw = data[end_len:end]
        return (raw.decode("utf-8") if tag == b"s" else raw), end
    if tag == b"l":
        end_len = offset + 8
        if end_len > len(data):
            raise ValueError("truncated list length")
        count = struct.unpack(">Q", data[offset:end_len])[0]
        values = []
        cursor = end_len
        for _ in range(count):
            item, cursor = canonical_unpack(data, cursor)
            values.append(item)
        return values, cursor
    if tag == b"d":
        end_len = offset + 8
        if end_len > len(data):
            raise ValueError("truncated mapping length")
        count = struct.unpack(">Q", data[offset:end_len])[0]
        result = {}
        cursor = end_len
        for _ in range(count):
            key, cursor = canonical_unpack(data, cursor)
            item, cursor = canonical_unpack(data, cursor)
            if not isinstance(key, str):
                raise ValueError("canonical mapping key is not a string")
            result[key] = item
        return result, cursor
    raise ValueError(f"unknown canonical value tag: {tag!r}")


def canonical_json(value: Any) -> bytes:
    """Stable JSON for manifests and small API summaries."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def hash128(data: bytes) -> bytes:
    """Return a fast deterministic 128-bit digest.

    xxhash is preferred when it is already installed (the engine vendors the
    same family); blake2b-128 is the dependency-free fallback.
    """
    try:
        import xxhash  # type: ignore
        return xxhash.xxh3_128_digest(data)
    except (ImportError, AttributeError):
        return hashlib.blake2b(data, digest_size=16).digest()


def runtime_fingerprint(repo_root: str | Path, *, binary: str | Path | None = None, **extra: Any) -> dict[str, Any]:
    """Build an immutable, content-addressable identity for a run."""
    root = Path(repo_root).resolve()
    git_sha = None
    try:
        import subprocess
        git_sha = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    binary_path = Path(binary).resolve() if binary else None
    result: dict[str, Any] = {
        "schema": 1,
        "repo_root": str(root),
        "git_sha": git_sha,
        "python": __import__("platform").python_version(),
        "platform": __import__("platform").platform(),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if binary_path and binary_path.exists():
        digest = hashlib.sha256()
        with binary_path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        result["binary"] = {"path": str(binary_path), "sha256": digest.hexdigest(), "size": binary_path.stat().st_size}
    result.update(extra)
    identity = dict(result)
    identity.pop("generated_at", None)
    result["fingerprint"] = hashlib.sha256(canonical_json(identity)).hexdigest()
    return result


def _compress(raw: bytes) -> tuple[int, bytes]:
    try:
        import zstandard as zstd  # type: ignore
        return 2, zstd.ZstdCompressor(level=1).compress(raw)
    except ImportError:
        return 1, zlib.compress(raw, level=1)


def _decompress(method: int, data: bytes, expected_size: int) -> bytes:
    if method == 0:
        raw = data
    elif method == 1:
        raw = zlib.decompress(data)
    elif method == 2:
        import zstandard as zstd  # type: ignore
        raw = zstd.ZstdDecompressor().decompress(data, max_output_size=expected_size)
    else:
        raise ValueError(f"unsupported compression method {method}")
    if len(raw) != expected_size:
        raise ValueError(f"decompressed length {len(raw)} != expected {expected_size}")
    return raw


@dataclass(frozen=True)
class Chunk:
    kind: str
    schema_version: int
    first_index: int
    count: int
    records: Any
    checksum: int
    chain: bytes
    compression: int
    raw_bytes: int
    compressed_bytes: int


class ReplayWriter:
    """Write a replay atomically; completed chunks are always readable."""

    def __init__(self, path: str | Path, manifest: Mapping[str, Any]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.partial_path = self.path.with_suffix(self.path.suffix + ".partial")
        self.manifest = dict(manifest)
        self.manifest.setdefault("container", "ogreplay")
        self.manifest.setdefault("container_version", VERSION)
        self.manifest.setdefault("created_at", time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        self.manifest.setdefault("episode_id", str(uuid.uuid4()))
        self.manifest.setdefault("verification", "recorded_state")
        self.manifest.setdefault("authoritative_complete", False)
        self.manifest.setdefault("runtime_fingerprint", None)
        self._stream = self.partial_path.open("wb")
        manifest_bytes = canonical_json(self.manifest)
        self._stream.write(HEADER.pack(MAGIC, VERSION, ENDIAN_MARKER, 0, len(manifest_bytes)))
        self._stream.write(manifest_bytes)
        self._chain = bytes(16)
        self._chunks: list[dict[str, Any]] = []
        self._closed = False

    def add_chunk(self, kind: str, first_index: int, records: Any, *, schema_version: int = 1, count: int | None = None) -> None:
        if self._closed:
            raise RuntimeError("replay writer is closed")
        if kind not in CHUNK_TYPES:
            raise ValueError(f"invalid data chunk kind: {kind}")
        encoded = canonical_pack(records)
        compression, compressed = _compress(encoded)
        checksum = zlib.crc32(encoded) & 0xFFFFFFFF
        self._chain = hash128(self._chain + encoded)
        header = CHUNK_HEADER.pack(
            CHUNK_CODES[kind], schema_version, 0, int(first_index),
            int(len(records) if count is None and isinstance(records, (list, tuple)) else (count or 1)),
            len(encoded), len(compressed), checksum, compression, self._chain,
        )
        offset = self._stream.tell()
        self._stream.write(header)
        self._stream.write(compressed)
        self._stream.flush()
        self._chunks.append({
            "kind": kind, "schema_version": schema_version, "first_index": first_index,
            "count": int(len(records) if count is None and isinstance(records, (list, tuple)) else (count or 1)),
            "offset": offset, "raw_bytes": len(encoded), "compressed_bytes": len(compressed),
            "checksum": checksum, "chain": self._chain.hex(), "compression": compression,
        })

    def finalize(self, *, outcome: str | None = None, terminal: Mapping[str, Any] | None = None) -> Path:
        if self._closed:
            return self.path
        # Keep a readable in-container index as well as the footer copy. The
        # footer copy is what makes crash recovery possible; this chunk is
        # what lets a complete reader inspect the layout without special
        # casing footer metadata.
        index = {"chunks": list(self._chunks), "complete": True}
        self.add_chunk("INDEX", 0, index, count=len(self._chunks))
        self.manifest["chunk_count"] = len(self._chunks)
        self.manifest["complete"] = True
        if outcome is not None:
            self.manifest["outcome"] = outcome
        if terminal is not None:
            self.manifest["terminal"] = dict(terminal)
        # The manifest is immutable once written.  Persist the authoritative
        # index and terminal summary in the footer so a reader can recover
        # completed chunks from a crash-truncated .partial without trusting a
        # second copy of the manifest.
        footer_meta = canonical_pack({
            "chunks": self._chunks,
            "manifest_sha256": hashlib.sha256(canonical_json(self.manifest)).hexdigest(),
            "final": {"outcome": outcome, "terminal": dict(terminal) if terminal is not None else None},
        })
        self._stream.flush()
        body = self.partial_path.read_bytes()
        footer_without_digest = FOOTER.pack(FOOTER_TAG, len(footer_meta), len(self._chunks), bytes(32))
        digest = hashlib.sha256(body + footer_without_digest + footer_meta).digest()
        self._stream.write(FOOTER.pack(FOOTER_TAG, len(footer_meta), len(self._chunks), digest))
        self._stream.write(footer_meta)
        self._stream.flush()
        os.fsync(self._stream.fileno())
        self._stream.close()
        # Replace the initial manifest with a final manifest only by writing a
        # fresh completed file would be expensive and risks partial rewrites.
        # The footer is authoritative for completion; the manifest's initial
        # fields remain stable and the footer carries outcome/index metadata.
        os.replace(self.partial_path, self.path)
        self._closed = True
        return self.path

    def abort(self) -> None:
        if not self._closed:
            self._stream.flush()
            self._stream.close()
            self._closed = True

    def __enter__(self) -> "ReplayWriter":
        return self

    def __exit__(self, exc_type, _exc, _tb) -> None:
        if exc_type is None:
            self.finalize()
        else:
            self.abort()


class ReplayReader:
    """Read complete chunks from .ogreplay or recover a .partial file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data = self.path.read_bytes()
        if len(self.data) < HEADER.size:
            raise ValueError("replay header is truncated")
        magic, version, endian, _flags, manifest_len = HEADER.unpack_from(self.data, 0)
        if magic != MAGIC:
            raise ValueError(f"not an ogreplay container: {magic!r}")
        if version != VERSION:
            raise ValueError(f"unsupported ogreplay version {version}")
        if endian != ENDIAN_MARKER:
            raise ValueError("endianness mismatch")
        manifest_start = HEADER.size
        manifest_end = manifest_start + manifest_len
        if manifest_end > len(self.data):
            raise ValueError("replay manifest is truncated")
        self.manifest = json.loads(self.data[manifest_start:manifest_end])
        self._data_start = manifest_end
        self.chunks: list[Chunk] = []
        self.footer: dict[str, Any] | None = None
        self.complete = False
        self._parse_chunks()

    def _parse_chunks(self) -> None:
        cursor = self._data_start
        while cursor + 4 <= len(self.data):
            if self.data[cursor:cursor + 4] == FOOTER_TAG:
                if cursor + FOOTER.size > len(self.data):
                    break
                _tag, meta_len, chunk_count, body_digest = FOOTER.unpack_from(self.data, cursor)
                meta_start = cursor + FOOTER.size
                meta_end = meta_start + meta_len
                if meta_end > len(self.data):
                    break
                meta, used = canonical_unpack(self.data[meta_start:meta_end])
                if used != meta_len:
                    raise ValueError("footer metadata has trailing bytes")
                footer_without_digest = FOOTER.pack(FOOTER_TAG, meta_len, chunk_count, bytes(32))
                digest_actual = hashlib.sha256(self.data[:cursor] + footer_without_digest + self.data[meta_start:meta_end]).digest()
                self.footer = {"chunk_count": chunk_count, "body_sha256": body_digest.hex(), "metadata": meta,
                               "body_digest_match": digest_actual == body_digest}
                self.complete = bool(self.footer["body_digest_match"] and chunk_count == len(self.chunks))
                break
            if cursor + CHUNK_HEADER.size > len(self.data):
                break
            raw_header = self.data[cursor:cursor + CHUNK_HEADER.size]
            code, schema, _flags, first, count, raw_len, compressed_len, checksum, compression, chain = CHUNK_HEADER.unpack(raw_header)
            end = cursor + CHUNK_HEADER.size + compressed_len
            if end > len(self.data):
                break
            kind = CODE_NAMES.get(code)
            if kind is None:
                raise ValueError(f"unknown replay chunk code {code!r}")
            compressed = self.data[cursor + CHUNK_HEADER.size:end]
            raw = _decompress(compression, compressed, raw_len)
            if zlib.crc32(raw) & 0xFFFFFFFF != checksum:
                raise ValueError(f"chunk checksum mismatch at offset {cursor}")
            records, used = canonical_unpack(raw)
            if used != len(raw):
                raise ValueError(f"chunk {kind} has trailing bytes")
            self.chunks.append(Chunk(kind, schema, first, count, records, checksum, chain, compression, raw_len, compressed_len))
            cursor = end

    @property
    def incomplete(self) -> bool:
        return not self.complete

    def records(self, kind: str) -> list[Any]:
        result: list[Any] = []
        for chunk in self.chunks:
            if chunk.kind != kind:
                continue
            if isinstance(chunk.records, list):
                result.extend(chunk.records)
            else:
                result.append(chunk.records)
        return result

    def summary(self) -> dict[str, Any]:
        manifest = dict(self.manifest)
        verification = manifest.get("verification", "recorded_state")
        if manifest.get("legacy"):
            status = "LEGACY / UNVERIFIABLE"
        elif verification == "reference_pixels_verified":
            status = "REFERENCE PIXELS VERIFIED"
        elif verification == "exact_simulation_verified":
            status = "EXACT SIMULATION VERIFIED"
        elif manifest.get("divergence"):
            status = f"DIVERGED AT TICK {manifest['divergence'].get('tick', '?')}"
        elif verification == "native_state_trace":
            status = "NATIVE STATE TRACE / UNVERIFIED"
        else:
            status = "RECORDED STATE PLAYBACK"
        manifest.update({
            "name": self.path.stem.replace(".ogreplay", ""),
            "path": str(self.path),
            "complete": self.complete,
            "incomplete": self.incomplete,
            "status": status,
            "chunk_count": len(self.chunks),
            "file_bytes": len(self.data),
            "tick_count": sum(c.count for c in self.chunks if c.kind == "TICK"),
            "decision_count": sum(c.count for c in self.chunks if c.kind == "DECISION"),
        })
        final = (self.footer or {}).get("metadata", {}).get("final", {})
        if isinstance(final, dict):
            if final.get("outcome") is not None:
                manifest["outcome"] = final["outcome"]
            if final.get("terminal") is not None:
                manifest["terminal"] = final["terminal"]
        return manifest


def load_replay_summary(path: str | Path) -> dict[str, Any]:
    try:
        return ReplayReader(path).summary()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {"name": Path(path).stem, "status": "CORRUPT / UNREADABLE", "error": str(exc), "path": str(path)}


def write_policy_trace(path: str | Path, *, meta: Mapping[str, Any], decisions: Iterable[Mapping[str, Any]], visual_states: Iterable[Mapping[str, Any]] = (), events: Iterable[Mapping[str, Any]] = (), native_ticks: Iterable[Mapping[str, Any]] = ()) -> Path:
    """Convenience writer used by the Python trainer's bounded recorder."""
    manifest = dict(meta)
    manifest.setdefault("trace_kind", "policy_plus_observation")
    manifest.setdefault("state_schema", "observation-v1-partial")
    manifest.setdefault("authoritative_complete", False)
    manifest.setdefault("verification", "recorded_state")
    decisions = list(decisions)
    visual_states = list(visual_states)
    events = list(events)
    native_ticks = list(native_ticks)
    with ReplayWriter(path, manifest) as writer:
        writer.add_chunk("RESET", 0, [{"episode": manifest.get("episode_id"), "seed": manifest.get("seed"), "scenario": manifest.get("scenario")}])
        if decisions:
            writer.add_chunk("DECISION", 0, decisions)
        if visual_states:
            writer.add_chunk("VISUAL", 0, visual_states)
        if native_ticks:
            writer.add_chunk("TICK", 0, native_ticks)
        if events:
            writer.add_chunk("EVENT", 0, events)
        writer.finalize(outcome=manifest.get("outcome"), terminal=manifest.get("terminal"))
    return Path(path)
