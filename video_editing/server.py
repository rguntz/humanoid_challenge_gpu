"""Local dashboard to cut a long recording into episodes.

Serves a small web page (index.html) where you scrub through a video in
videos/task/, mark the start and end of each episode, and export them. Each
episode is cut frame-accurately (re-encoded with ffmpeg, libx264 CRF 18, upright
1080x1920) into videos/task/dataset/<video>_epNNN.mp4. The episode list is saved
as you edit, in videos/task/dataset/<video>_episodes.json, so you can stop and
come back later.

Usage:
    uv run video_editing/server.py          # then open http://localhost:8765
"""
import argparse
import json
import mimetypes
import re
import subprocess
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import cv2
import imageio_ffmpeg

ROOT = Path(__file__).parent.parent
HERE = Path(__file__).parent
VIDEO_DIR = ROOT / "videos" / "task"
DATASET_DIR = VIDEO_DIR / "dataset"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
CHUNK = 8 << 20  # bytes per answer to an open-ended video range request

_info_cache = {}
_export = {"running": False, "video": None, "done": 0, "total": 0, "current": None, "log": []}


def video_info(path):
    """Duration, fps and frame count (cached; OpenCV reads the container header)."""
    key = (path, path.stat().st_mtime)
    if key not in _info_cache:
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        _info_cache[key] = {"name": path.name, "fps": fps, "frames": frames,
                            "duration": frames / fps if fps else 0, "size_mb": round(path.stat().st_size / 1e6)}
    return _info_cache[key]


def episodes_path(video_name):
    return DATASET_DIR / f"{Path(video_name).stem}_episodes.json"


def load_episodes(video_name):
    p = episodes_path(video_name)
    if p.exists():
        return json.loads(p.read_text())
    return {"video": video_name, "next_id": 1, "episodes": []}


def save_episodes(video_name, data):
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    data["video"] = video_name
    episodes_path(video_name).write_text(json.dumps(data, indent=2))


def clip_name(video_name, ep_id):
    return f"{Path(video_name).stem}_ep{ep_id:03d}.mp4"


def export_worker(video_name, overwrite):
    """Cut every complete episode; skip the ones already exported with the same bounds."""
    src = VIDEO_DIR / video_name
    info = video_info(src)
    data = load_episodes(video_name)
    todo = []
    for ep in data["episodes"]:
        if ep.get("start_frame") is None or ep.get("end_frame") is None or ep["end_frame"] <= ep["start_frame"]:
            continue  # incomplete or invalid: not exported
        out = DATASET_DIR / clip_name(video_name, ep["id"])
        done = ep.get("exported") or {}
        same = done.get("start_frame") == ep["start_frame"] and done.get("end_frame") == ep["end_frame"]
        if out.exists() and same and not overwrite:
            continue
        todo.append((ep, out))
    _export.update(running=True, video=video_name, done=0, total=len(todo), current=None, log=[])
    for ep, out in todo:
        _export["current"] = out.name
        start = ep["start_frame"] / info["fps"]
        n_frames = ep["end_frame"] - ep["start_frame"] + 1
        # -ss before -i with re-encoding is frame-accurate; ffmpeg applies the
        # phone's rotation, so the clip is stored upright without a rotation tag.
        cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{start:.4f}", "-i", str(src),
               "-frames:v", str(n_frames), "-an", "-c:v", "libx264", "-crf", "18", "-preset", "fast",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            ep["exported"] = {"file": out.name, "start_frame": ep["start_frame"], "end_frame": ep["end_frame"]}
            _export["log"].append(f"{out.name}: {n_frames} frames ({n_frames / info['fps']:.1f} s)")
        else:
            _export["log"].append(f"{out.name}: FAILED {r.stderr.strip()[-300:]}")
        # Re-read before saving so edits made in the page during the export are kept.
        latest = load_episodes(video_name)
        for e in latest["episodes"]:
            if e["id"] == ep["id"] and "exported" in ep:
                e["exported"] = ep["exported"]
        save_episodes(video_name, latest)
        _export["done"] += 1
    _export.update(running=False, current=None)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # keep the terminal quiet

    def send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_json(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        if url.path == "/":
            body = (HERE / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif url.path == "/api/videos":
            videos = [video_info(p) for p in sorted(VIDEO_DIR.glob("*.mp4"))]
            self.send_json(videos)
        elif url.path == "/api/episodes":
            self.send_json(load_episodes(q["video"][0]))
        elif url.path == "/api/export_status":
            self.send_json(_export)
        elif url.path.startswith("/video/"):
            self.send_video(VIDEO_DIR / Path(unquote(url.path[len("/video/"):])).name)
        else:
            self.send_error(404)

    def do_POST(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        if url.path == "/api/episodes":
            save_episodes(q["video"][0], self.read_json())
            self.send_json({"ok": True})
        elif url.path == "/api/export":
            if _export["running"]:
                self.send_json({"ok": False, "error": "an export is already running"}, 409)
                return
            body = self.read_json()
            threading.Thread(target=export_worker, args=(q["video"][0], bool(body.get("overwrite"))),
                             daemon=True).start()
            self.send_json({"ok": True})
        else:
            self.send_error(404)

    def send_video(self, path):
        """Serve the file with HTTP Range support, which the browser needs to seek."""
        if not path.exists():
            self.send_error(404)
            return
        size = path.stat().st_size
        start, end = 0, size - 1
        m = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        if m:
            if m.group(1):
                start = int(m.group(1))
                # Open-ended request ("bytes=N-"): answer with a chunk, not the rest
                # of a 1.7 GB file; the browser asks for the next part when needed.
                end = int(m.group(2)) if m.group(2) else min(size - 1, start + CHUNK - 1)
            else:  # suffix range: last N bytes
                start = size - int(m.group(2))
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        try:
            with open(path, "rb") as f:
                f.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    chunk = f.read(min(1 << 20, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except OSError:
            # The browser drops video requests it no longer needs (e.g. when seeking);
            # on macOS this can surface as EPIPE, ECONNRESET or ENOBUFS (errno 55).
            pass


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://localhost:{args.port}"
    print(f"Episode cutter running at {url}  (Ctrl+C to stop)")
    print(f"Videos from {VIDEO_DIR}, clips to {DATASET_DIR}")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
