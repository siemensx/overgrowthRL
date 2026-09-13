#!/usr/bin/env python3
"""Record a rendered ``watch.py`` session to an MP4 on macOS.

The engine's existing tape/native-digest artifacts explain state and actions,
but they do not show the rendered scene.  This small wrapper captures the
screen with FFmpeg's AVFoundation input while ``watch.py`` owns the game
window.  It is deliberately a wrapper rather than an engine change.  The
default assisted camera is diagnostic-only: legacy controlled-character
target selection can read camera state, so these recordings are not
canonical evaluation runs.

Usage::

    python3 Tools/rl/record_watch.py --out /tmp/watch.mp4 -- \
        --checkpoint Tools/rl/ppo/checkpoints/run21_win_latest.pt \
        --level arenas/t_train_101.xml --frame-stack 4 --act-period 4 \
        --opponents 3 --armed-count 0 --episodes 5 --device cpu

The AVFoundation screen index is machine-specific.  On this Mac it is 4
(``Capture screen 0``); use ``ffmpeg -f avfoundation -list_devices true -i ''``
to discover it on another machine.  macOS Screen Recording permission for the
terminal running this command is required.
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="output MP4 path")
    parser.add_argument("--screen", default="4", help="AVFoundation screen video index (default: 4)")
    parser.add_argument("--fps", type=int, default=30, help="capture frame rate (default: 30)")
    parser.add_argument("--warmup-seconds", type=float, default=1.0,
                        help="capture lead-in before launching watch.py")
    parser.add_argument("--tail-seconds", type=float, default=1.5,
                        help="capture lead-out after watch.py exits")
    parser.add_argument("--ffmpeg", default=None, help="FFmpeg executable; default: PATH lookup")
    parser.add_argument("watch_args", nargs=argparse.REMAINDER,
                        help="arguments for watch.py after a literal --")
    args = parser.parse_args()
    if args.watch_args[:1] == ["--"]:
        args.watch_args = args.watch_args[1:]
    if not args.watch_args:
        parser.error("watch.py arguments are required after --")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.warmup_seconds < 0 or args.tail_seconds < 0:
        parser.error("warmup and tail seconds cannot be negative")
    return args


def main() -> int:
    args = parse_args()
    ffmpeg = args.ffmpeg or shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("FFmpeg was not found; install it or pass --ffmpeg /path/to/ffmpeg")
    repo_root = Path(__file__).resolve().parents[2]
    output = Path(args.out).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    watch_script = repo_root / "Tools" / "rl" / "ppo" / "watch.py"

    capture_command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "avfoundation", "-framerate", str(args.fps),
        "-capture_cursor", "0", "-pixel_format", "nv12", "-i", f"{args.screen}:none",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
    ]
    watch_args = list(args.watch_args)
    if "--auto-camera" not in watch_args:
        watch_args.append("--auto-camera")
    watch_command = [sys.executable, str(watch_script), *watch_args]
    print("recording:", " ".join(capture_command), flush=True)
    print("watching:  ", " ".join(watch_command), flush=True)

    capture = subprocess.Popen(capture_command, stdin=subprocess.PIPE,
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                               text=False)
    watch = None
    watch_returncode = 1
    interrupted = False
    try:
        time.sleep(args.warmup_seconds)
        if capture.poll() is not None:
            error = capture.stderr.read().decode(errors="replace").strip()
            raise RuntimeError(f"FFmpeg exited before watch.py started: {error}")
        watch = subprocess.Popen(watch_command, cwd=repo_root)
        watch_returncode = watch.wait()
    except KeyboardInterrupt:
        interrupted = True
        if watch is not None and watch.poll() is None:
            watch.send_signal(signal.SIGINT)
            watch.wait(timeout=10)
    finally:
        if watch is not None and watch.poll() is None:
            watch.terminate()
            watch.wait(timeout=10)
        if capture.poll() is None:
            time.sleep(args.tail_seconds)
            try:
                # Closing stdin after the q command matters on macOS: leaving
                # the pipe open can keep FFmpeg in its capture loop even after
                # it has received the interactive quit command.
                capture.stdin.write(b"q\n")
                capture.stdin.flush()
                capture.stdin.close()
            except (BrokenPipeError, OSError, ValueError):
                pass
        try:
            capture.wait(timeout=30)
        except subprocess.TimeoutExpired:
            capture.terminate()
            try:
                capture.wait(timeout=10)
            except subprocess.TimeoutExpired:
                capture.kill()
                capture.wait(timeout=10)
            raise RuntimeError("FFmpeg did not finalize within 30 seconds")
        capture_error = capture.stderr.read() if capture.stderr is not None else b""

    if capture.returncode != 0:
        detail = capture_error.decode(errors="replace").strip()
        raise RuntimeError(f"FFmpeg failed with exit code {capture.returncode}: {detail}")
    if not output.exists() or output.stat().st_size == 0:
        raise RuntimeError(f"FFmpeg reported success but produced no movie: {output}")
    print(f"saved {output} ({output.stat().st_size:,} bytes)", flush=True)
    if interrupted:
        return 130
    return watch_returncode


if __name__ == "__main__":
    raise SystemExit(main())
