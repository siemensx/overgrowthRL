#!/usr/bin/env python3
"""Record rendered ``watch.py`` fights as window-only, per-episode MP4s.

The engine's tape/native-digest artifacts explain state and actions, but they
do not show the rendered scene.  This wrapper starts ``watch.py``, finds its
Overgrowth window through macOS CoreGraphics, captures the display with
FFmpeg, crops to that window, and splits the result at the episode boundaries
that ``watch.py`` writes.  The default output is therefore one movie per
fight, not a recording of the whole desktop or one movie for the whole
command.

The assisted camera and wider FOV are render-only diagnostic aids.  Legacy
controlled-character target selection can read camera state, so these videos
are not canonical evaluation runs.

Usage::

    python3 Tools/rl/record_watch.py --out /tmp/run21_latest_5ep.mp4 -- \
        --checkpoint Tools/rl/ppo/checkpoints/run21_win_latest.pt \
        --level arenas/t_train_101.xml --frame-stack 4 --act-period 4 \
        --opponents 3 --armed-count 0 --difficulty 1.0 \
        --seed 900000 --episodes 5 --device cpu

This creates ``run21_latest_5ep_ep00.mp4`` through ``ep04.mp4`` and an
``run21_latest_5ep_episodes.json`` summary beside them.  Add
``--keep-master`` only when the full cropped command recording is useful.

The AVFoundation screen index is machine-specific.  On this Mac it is 4
(``Capture screen 0``); use ``ffmpeg -f avfoundation -list_devices true -i ''``
to discover it on another machine.  macOS Screen Recording permission for the
terminal running this command is required.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", required=True,
        help="output filename stem; per-fight files add _epNN.mp4")
    parser.add_argument(
        "--screen", default="4",
        help="AVFoundation screen video index (default: 4)")
    parser.add_argument(
        "--fps", type=int, default=30,
        help="capture frame rate (default: 30)")
    parser.add_argument(
        "--window-title", default="Overgrowth",
        help="case-insensitive owner name to crop (default: Overgrowth)")
    parser.add_argument(
        "--window-rect", default=None,
        help="manual capture rectangle x,y,width,height in display pixels")
    parser.add_argument(
        "--window-timeout", type=float, default=30.0,
        help="seconds to wait for the Overgrowth window (default: 30)")
    parser.add_argument(
        "--full-screen", action="store_true",
        help="explicitly capture the whole selected display; not recommended")
    parser.add_argument(
        "--warmup-seconds", type=float, default=0.0,
        help="delay after the window is found before capture begins")
    parser.add_argument(
        "--tail-seconds", type=float, default=1.5,
        help="capture lead-out after watch.py exits (default: 1.5)")
    parser.add_argument(
        "--episode-padding", type=float, default=0.35,
        help="seconds added before/after each fight when splitting (default: 0.35)")
    parser.add_argument(
        "--keep-master", action="store_true",
        help="also retain the full cropped recording at --out")
    parser.add_argument(
        "--ffmpeg", default=None,
        help="FFmpeg executable; default: PATH lookup")
    parser.add_argument(
        "watch_args", nargs=argparse.REMAINDER,
        help="arguments for watch.py after a literal --")
    args = parser.parse_args()
    if args.watch_args[:1] == ["--"]:
        args.watch_args = args.watch_args[1:]
    if not args.watch_args:
        parser.error("watch.py arguments are required after --")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.window_timeout <= 0:
        parser.error("--window-timeout must be positive")
    if args.warmup_seconds < 0 or args.tail_seconds < 0 or args.episode_padding < 0:
        parser.error("warmup, tail, and episode padding cannot be negative")
    if args.window_rect is not None:
        try:
            rect = tuple(int(part.strip()) for part in args.window_rect.split(","))
        except ValueError:
            rect = ()
        if len(rect) != 4 or rect[2] <= 0 or rect[3] <= 0:
            parser.error("--window-rect must be x,y,width,height with positive width and height")
    return args


def parse_window_rect(value: str) -> tuple[int, int, int, int]:
    x, y, width, height = (int(part.strip()) for part in value.split(","))
    return x, y, width, height


def find_window_rect(owner: str) -> tuple[int, int, int, int] | None:
    """Return the frontmost matching layer-0 window in display pixels.

    CoreGraphics reports window bounds in display points.  The selected
    AVFoundation display receives pixel coordinates, so the small Swift probe
    applies the main-display scale before returning the rectangle.
    """
    swift_code = r'''
import CoreGraphics
import Foundation

let target = TARGET_OWNER.lowercased()
let main = CGMainDisplayID()
let displayBounds = CGDisplayBounds(main)
let mode = CGDisplayCopyDisplayMode(main)
let scaleX = mode.map { CGFloat($0.pixelWidth) / CGFloat($0.width) } ?? 1.0
let scaleY = mode.map { CGFloat($0.pixelHeight) / CGFloat($0.height) } ?? 1.0
let options = CGWindowListOption(arrayLiteral: .optionOnScreenOnly, .excludeDesktopElements)
let windows = CGWindowListCopyWindowInfo(options, kCGNullWindowID) as? [[String: Any]] ?? []

for info in windows {
    guard let owner = info[kCGWindowOwnerName as String] as? String,
          owner.lowercased().contains(target) else { continue }
    let layer = info[kCGWindowLayer as String] as? Int ?? 0
    guard layer == 0 else { continue }
    guard let bounds = info[kCGWindowBounds as String] as? NSDictionary,
          let rect = CGRect(dictionaryRepresentation: bounds) else { continue }
    let x = (rect.origin.x - displayBounds.origin.x) * scaleX
    let y = (rect.origin.y - displayBounds.origin.y) * scaleY
    let width = rect.size.width * scaleX
    let height = rect.size.height * scaleY
    guard width >= 200.0 && height >= 100.0 else { continue }
    let number = info[kCGWindowNumber as String] as? UInt32 ?? 0
    print("\(number) \(Int(x.rounded())) \(Int(y.rounded())) \(Int(width.rounded())) \(Int(height.rounded()))")
}
'''.replace("TARGET_OWNER", json.dumps(owner))
    try:
        result = subprocess.run(
            ["swift", "-e", swift_code],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        match = re.fullmatch(r"\s*\d+\s+(-?\d+)\s+(-?\d+)\s+(\d+)\s+(\d+)\s*", line)
        if match:
            x, y, width, height = (int(value) for value in match.groups())
            return x, y, width, height
    return None


def even_capture_rect(rect: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x, y, width, height = rect
    x = max(0, x)
    y = max(0, y)
    width = max(2, width - (width % 2))
    height = max(2, height - (height % 2))
    return x - (x % 2), y - (y % 2), width, height


def wait_for_window(
    owner: str, timeout: float, watch: subprocess.Popen,
) -> tuple[int, int, int, int]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if watch.poll() is not None:
            raise RuntimeError("watch.py exited before the Overgrowth window appeared")
        rect = find_window_rect(owner)
        if rect is not None:
            return even_capture_rect(rect)
        time.sleep(0.25)
    raise RuntimeError(
        f"could not find a visible '{owner}' window within {timeout:g}s; "
        "use --window-rect x,y,width,height only after checking the window, "
        "or explicitly pass --full-screen"
    )


def stop_process(process: subprocess.Popen | None, sig: int = signal.SIGTERM) -> None:
    if process is None or process.poll() is not None:
        return
    process.send_signal(sig)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def stop_capture(capture: subprocess.Popen, tail_seconds: float) -> bytes:
    if capture.poll() is None:
        if tail_seconds:
            time.sleep(tail_seconds)
        try:
            # Closing stdin after the q command matters on macOS: leaving the
            # pipe open can keep FFmpeg in its capture loop after quit.
            if capture.stdin is not None:
                capture.stdin.write(b"q\n")
                capture.stdin.flush()
                capture.stdin.close()
        except (BrokenPipeError, OSError, ValueError):
            pass
    try:
        capture.wait(timeout=30)
    except subprocess.TimeoutExpired:
        stop_process(capture)
        raise RuntimeError("FFmpeg did not finalize within 30 seconds")
    return capture.stderr.read() if capture.stderr is not None else b""


def read_episode_events(path: Path) -> list[dict]:
    by_episode: dict[int, dict] = {}
    if not path.exists():
        return []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        episode = int(record["episode"])
        current = by_episode.setdefault(episode, {})
        current[record["event"]] = record
    episodes = []
    for episode, record in sorted(by_episode.items()):
        if "start" in record and "end" in record:
            episodes.append({
                "episode": episode,
                "seed": int(record["start"]["seed"]),
                "outcome": record["end"].get("outcome"),
                "start_wall_time": float(record["start"]["wall_time"]),
                "end_wall_time": float(record["end"]["wall_time"]),
                "steps": record["end"].get("steps"),
            })
    return episodes


def ffprobe_duration(ffprobe: str, movie: Path) -> float:
    result = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(movie)],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def episode_output_path(output: Path, episode: int) -> Path:
    suffix = output.suffix or ".mp4"
    stem = output.stem if output.suffix else output.name
    return output.with_name(f"{stem}_ep{episode:02d}{suffix}")


def split_episode(
    ffmpeg: str, master: Path, output: Path, start: float, end: float,
) -> None:
    duration = max(0.05, end - start)
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{max(0.0, start):.3f}", "-i", str(master),
        "-t", f"{duration:.3f}", "-an",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg could not split episode to {output}: {result.stderr.strip()}")


def main() -> int:
    args = parse_args()
    ffmpeg = args.ffmpeg or shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("FFmpeg was not found; install it or pass --ffmpeg /path/to/ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise SystemExit("ffprobe was not found; install FFmpeg's command-line tools")

    repo_root = Path(__file__).resolve().parents[2]
    output = Path(args.out).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    watch_script = repo_root / "Tools" / "rl" / "ppo" / "watch.py"
    events_path = output.with_name(f".{output.stem}_episode_events.jsonl")
    summary_path = output.with_name(f"{output.stem}_episodes.json")
    fd, master_name = tempfile.mkstemp(prefix=f".{output.stem}_master_", suffix=".mp4", dir=output.parent)
    os.close(fd)
    master = Path(master_name)

    watch_args = list(args.watch_args)
    if "--auto-camera" not in watch_args:
        watch_args.append("--auto-camera")
    # The recorder owns this path so that episode splitting cannot accidentally
    # consume a stale event file from an earlier command.
    watch_args.extend(["--episode-events", str(events_path)])
    watch_command = [sys.executable, str(watch_script), *watch_args]

    capture_filter: list[str] = []
    rect = None
    if args.window_rect is not None:
        rect = even_capture_rect(parse_window_rect(args.window_rect))
    if not args.full_screen and rect is not None:
        x, y, width, height = rect
        capture_filter = ["-vf", f"crop={width}:{height}:{x}:{y}"]

    watch: subprocess.Popen | None = None
    capture: subprocess.Popen | None = None
    capture_started_wall = 0.0
    watch_returncode = 1
    interrupted = False
    capture_error = b""
    try:
        print("watching:  ", " ".join(watch_command), flush=True)
        watch = subprocess.Popen(watch_command, cwd=repo_root)
        if not args.full_screen and rect is None:
            rect = wait_for_window(args.window_title, args.window_timeout, watch)
        if rect is not None:
            print(f"window crop: x={rect[0]} y={rect[1]} width={rect[2]} height={rect[3]}", flush=True)
            capture_filter = ["-vf", f"crop={rect[2]}:{rect[3]}:{rect[0]}:{rect[1]}"]
        if args.warmup_seconds:
            time.sleep(args.warmup_seconds)

        capture_command = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "avfoundation", "-framerate", str(args.fps),
            "-capture_cursor", "0", "-pixel_format", "nv12",
            "-i", f"{args.screen}:none", *capture_filter,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(master),
        ]
        print("recording: ", " ".join(capture_command), flush=True)
        capture_started_wall = time.time()
        capture = subprocess.Popen(
            capture_command, stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=False,
        )
        watch_returncode = watch.wait()
    except KeyboardInterrupt:
        interrupted = True
        stop_process(watch, signal.SIGINT)
    finally:
        if watch is not None and watch.poll() is None:
            stop_process(watch)
        if capture is not None:
            capture_error = stop_capture(capture, args.tail_seconds)

    try:
        if capture is None:
            raise RuntimeError("capture did not start")
        if capture.returncode != 0:
            detail = capture_error.decode(errors="replace").strip()
            raise RuntimeError(f"FFmpeg failed with exit code {capture.returncode}: {detail}")
        if not master.exists() or master.stat().st_size == 0:
            raise RuntimeError("FFmpeg reported success but produced no movie")

        master_duration = ffprobe_duration(ffprobe, master)
        episodes = read_episode_events(events_path)
        if not episodes:
            raise RuntimeError(
                "watch.py produced no complete episode boundaries; refusing to emit a misleading "
                "single-command movie"
            )
        for episode in episodes:
            start = max(0.0, episode["start_wall_time"] - capture_started_wall - args.episode_padding)
            end = min(master_duration, episode["end_wall_time"] - capture_started_wall + args.episode_padding)
            episode["video_start_seconds"] = start
            episode["video_end_seconds"] = end
            episode_path = episode_output_path(output, episode["episode"])
            split_episode(ffmpeg, master, episode_path, start, end)
            episode["video_path"] = str(episode_path)
            print(f"saved {episode_path} ({episode_path.stat().st_size:,} bytes) {episode['outcome']}", flush=True)

        summary = {
            "source": "Tools/rl/record_watch.py",
            "window_title": args.window_title,
            "window_rect": rect,
            "screen": args.screen,
            "fps": args.fps,
            "master_duration_seconds": master_duration,
            "master_retained": args.keep_master,
            "episodes": episodes,
        }
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"episode summary: {summary_path}", flush=True)
        if args.keep_master:
            shutil.copy2(master, output)
            print(f"saved cropped master {output} ({output.stat().st_size:,} bytes)", flush=True)
    finally:
        master.unlink(missing_ok=True)
        events_path.unlink(missing_ok=True)

    if interrupted:
        return 130
    return watch_returncode


if __name__ == "__main__":
    raise SystemExit(main())
