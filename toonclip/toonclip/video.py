"""Video plumbing: YouTube download, frame extraction, writing with audio."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np


def ffmpeg_exe() -> str:
    """System ffmpeg if installed, otherwise the copy bundled with imageio-ffmpeg."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def download(url: str, out_dir: str | Path, max_height: int = 1080) -> str:
    """Download a video with yt-dlp and return the local file path.

    Only download videos you own or have permission to reuse.
    """
    import yt_dlp

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    opts = {
        "format": f"bv*[height<={max_height}]+ba/b[height<={max_height}]/b",
        "merge_output_format": "mp4",
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "ffmpeg_location": ffmpeg_exe(),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        path = Path(ydl.prepare_filename(info)).with_suffix(".mp4")
    if not path.exists():
        raise RuntimeError("Download finished but the video file was not found.")
    return str(path)


def probe(src: str) -> dict:
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not 1 <= fps <= 240:
        fps = 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    info = {
        "fps": fps,
        "frames": n,
        "duration": n / fps if n else 0.0,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


def fit_size(width: int, height: int, short_side: int) -> tuple[int, int]:
    """Working size for the model: short side ~= `short_side`, both sides /64."""
    s = short_side / min(width, height)
    w = max(int(round(width * s / 64)) * 64, 64)
    h = max(int(round(height * s / 64)) * 64, 64)
    return w, h


def _cover(frame, w: int, h: int):
    """Resize + centre-crop so the frame exactly fills w x h (no stretching)."""
    fh, fw = frame.shape[:2]
    s = max(w / fw, h / fh)
    nw, nh = max(int(round(fw * s)), w), max(int(round(fh * s)), h)
    frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    x, y = (nw - w) // 2, (nh - h) // 2
    return frame[y:y + h, x:x + w]


def read_frames(src: str, start: float, end: float | None, fps_out: float, size: tuple[int, int]):
    """Sample the clip at `fps_out` and return (frames RGB uint8, actual start, actual duration)."""
    info = probe(src)
    fps = info["fps"]
    cap = cv2.VideoCapture(str(src))
    first = int(round(max(start, 0.0) * fps))
    if first:
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    last = int(round(end * fps)) if end else (info["frames"] or 10**9)
    if info["frames"]:
        last = min(last, info["frames"])
    step = fps / fps_out
    frames, i, want = [], 0, 0.0
    while first + i < last:
        ok, frame = cap.read()
        if not ok:
            break
        if i >= want - 1e-6:
            frames.append(cv2.cvtColor(_cover(frame, *size), cv2.COLOR_BGR2RGB))
            want += step
        i += 1
    cap.release()
    if not frames:
        raise ValueError("No frames in that time range. Check the start/end times.")
    return frames, first / fps, i / fps


def write_video(frames, dst: str, fps: float, audio_src: str | None = None,
                audio_start: float = 0.0, duration: float | None = None,
                out_height: int | None = None) -> str:
    """Encode RGB frames to mp4; optionally copy the audio from `audio_src`."""
    h, w = frames[0].shape[:2]
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{fps:.6f}", "-i", "-"]
    if audio_src:
        cmd += ["-ss", f"{audio_start:.4f}"]
        if duration:
            cmd += ["-t", f"{duration:.4f}"]
        cmd += ["-i", str(audio_src), "-map", "0:v:0", "-map", "1:a:0?", "-c:a", "aac", "-b:a", "192k"]
    if out_height and out_height != h:
        ow = int(round(w * out_height / h / 2)) * 2
        cmd += ["-vf", f"scale={ow}:{out_height // 2 * 2}:flags=lanczos"]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(dst)]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for f in frames:
            enc.stdin.write(np.ascontiguousarray(f).tobytes())
        enc.stdin.close()
    except BrokenPipeError:
        pass
    err = enc.stderr.read().decode(errors="replace")
    if enc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed:\n{err.strip()}")
    return str(dst)
