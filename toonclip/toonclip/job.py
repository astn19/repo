"""One conversion from start to finish: source -> frames -> AI redraw -> mp4 with audio."""
from __future__ import annotations

import time
from pathlib import Path

from . import video
from .engine import ROOT, Engine, Settings

OUT_DIR = ROOT / "output"
DL_DIR = ROOT / "downloads"
MAX_SECONDS = 90


def resolve_source(source: str, log=print) -> str:
    """Accept a local file path or a video URL."""
    source = (source or "").strip().strip('"')
    if not source:
        raise ValueError("Give me a video file or a YouTube link.")
    if source.lower().startswith(("http://", "https://")):
        log("Downloading video...")
        return video.download(source, DL_DIR)
    if not Path(source).exists():
        raise ValueError(f"File not found: {source}")
    return source


def convert(engine: Engine, source: str, settings: Settings, start: float = 0.0,
            end: float | None = None, draw_fps: float = 12.0, short_side: int = 512,
            out_height: int | None = 1080, out_path: str | None = None,
            progress=None, log=print) -> str:
    src = resolve_source(source, log)
    info = video.probe(src)
    end = end if end and end > start else None
    length = (end or info["duration"] or 0) - start
    if length > MAX_SECONDS:
        raise ValueError(f"That is {length:.0f} s. Pick a section of {MAX_SECONDS} s or less "
                         "with the start/end boxes (long clips are best done in pieces).")
    size = video.fit_size(info["width"], info["height"], int(short_side))
    draw_fps = min(float(draw_fps), info["fps"])
    frames, t0, dur = video.read_frames(src, start, end, draw_fps, size)
    log(f"{len(frames)} drawings at {size[0]}x{size[1]} ({dur:.1f} s of video at {draw_fps:g} fps)")

    t = time.time()
    drawn = engine.stylize(frames, settings, progress=progress, log=log)
    log(f"Drawing took {time.time() - t:.0f} s")

    if not out_path:
        OUT_DIR.mkdir(exist_ok=True)
        safe = "".join(c if c.isalnum() else "_" for c in settings.style).strip("_")
        out_path = str(OUT_DIR / f"{Path(src).stem[:40]}_{safe}_{int(time.time())}.mp4")
    # real fps of the sampled frames so picture and sound stay in sync
    fps = len(drawn) / dur if dur > 0 else draw_fps
    video.write_video(drawn, out_path, fps, audio_src=src, audio_start=t0, duration=dur,
                      out_height=out_height)
    log(f"Saved: {out_path}")
    return out_path
