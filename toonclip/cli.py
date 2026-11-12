"""Command line use.

  python cli.py doctor                       check GPU, ffmpeg and model downloads
  python cli.py styles                       list styles
  python cli.py render VIDEO_OR_URL --style "Comic Book" --start 12 --end 30
"""
from __future__ import annotations

import argparse
import sys


def doctor() -> int:
    ok = True
    try:
        import torch

        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            gb = p.total_memory / 2**30
            print(f"[ok]   GPU: {p.name}, {gb:.0f} GB")
            if gb < 11:
                print("[warn] Under 12 GB: start with --low-vram and keep resolution at 512.")
        else:
            ok = False
            print("[FAIL] PyTorch cannot see an NVIDIA GPU. Re-run the installer, or update your "
                  "NVIDIA driver. (It will still run on CPU, but takes hours.)")
    except ImportError:
        print("[FAIL] PyTorch is not installed. Run the installer first.")
        return 1
    try:
        from toonclip import video

        print(f"[ok]   ffmpeg: {video.ffmpeg_exe()}")
    except Exception as e:
        ok = False
        print(f"[FAIL] ffmpeg: {e}")
    try:
        from huggingface_hub import model_info

        from toonclip import engine as E

        repos = {E.MOTION, E.MOTION_FAST, E.CN_DEPTH, E.CN_EDGE, E.VAE, *E.DEPTH_MODELS}
        for name, st in E.load_styles().items():
            good = None
            for m in st["models"]:
                if m.lower().endswith((".safetensors", ".ckpt")):
                    import os

                    good = m if os.path.exists(m) else None
                else:
                    try:
                        model_info(m)
                        good = m
                    except Exception:
                        print(f"[warn] style '{name}': model {m} not reachable")
                if good:
                    break
            if good:
                print(f"[ok]   style '{name}' -> {good}")
            else:
                ok = False
                print(f"[FAIL] style '{name}' has no working model. Edit styles.json.")
        for r in sorted(repos):
            try:
                model_info(r)
                print(f"[ok]   {r}")
            except Exception as e:
                tag = "[warn]" if r in E.DEPTH_MODELS or r == E.VAE else "[FAIL]"
                ok = ok and tag == "[warn]"
                print(f"{tag} {r}: {type(e).__name__}")
    except Exception as e:
        ok = False
        print(f"[FAIL] model check: {e}")
    print("\nAll good." if ok else "\nSome checks failed; see above.")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Turn a video clip into animation on your own GPU.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor")
    sub.add_parser("styles")
    r = sub.add_parser("render")
    r.add_argument("source", help="video file or YouTube link")
    r.add_argument("-o", "--out")
    r.add_argument("--style", default="Anime")
    r.add_argument("--prompt", default="", help='what is in the clip, e.g. "a golfer swinging a club"')
    r.add_argument("--start", type=float, default=0)
    r.add_argument("--end", type=float)
    r.add_argument("--restyle", type=float, default=0.65)
    r.add_argument("--structure", type=float, default=0.8)
    r.add_argument("--fps", type=float, default=12)
    r.add_argument("--size", type=int, default=512, help="working short side: 512, 576, 640, 768")
    r.add_argument("--out-height", type=int, default=1080)
    r.add_argument("--steps", type=int, default=20)
    r.add_argument("--seed", type=int, default=7)
    r.add_argument("--fast", action="store_true")
    r.add_argument("--long-window", action="store_true")
    r.add_argument("--model", default="", help="custom SD1.5 model (repo id or .safetensors)")
    r.add_argument("--low-vram", action="store_true")
    a = ap.parse_args()

    if a.cmd == "doctor":
        return doctor()
    from toonclip.engine import Engine, Settings, load_styles

    if a.cmd == "styles":
        print("\n".join(load_styles()))
        return 0
    from toonclip.job import convert

    s = Settings(style=a.style, prompt=a.prompt, restyle=a.restyle, structure=a.structure,
                 steps=a.steps, seed=a.seed, fast=a.fast, long_window=a.long_window,
                 custom_model=a.model)
    convert(Engine(low_vram=a.low_vram), a.source, s, a.start, a.end, a.fps, a.size,
            a.out_height, a.out,
            progress=lambda f, m: print(f"\r{m}  {f * 100:5.1f}%", end="", flush=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
