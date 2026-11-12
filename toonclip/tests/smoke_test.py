"""Plumbing test: runs the real engine code with tiny random stand-in models on CPU.

It proves the pipeline wiring, windowing, cross-fades, audio and file output work.
It says nothing about picture quality (random models draw noise).

    python tests/smoke_test.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import diffusers  # noqa: E402
from diffusers import AutoencoderKL, ControlNetModel, DDIMScheduler, MotionAdapter, UNet2DConditionModel  # noqa: E402
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer  # noqa: E402

from toonclip import engine as E  # noqa: E402
from toonclip import job, video  # noqa: E402

TMP = Path(tempfile.mkdtemp())


def tiny_tokenizer():
    chars = list("abcdefghijklmnopqrstuvwxyz0123456789,.")
    toks = chars + [c + "</w>" for c in chars] + ["<|startoftext|>", "<|endoftext|>"]
    (TMP / "vocab.json").write_text(json.dumps({t: i for i, t in enumerate(toks)}))
    (TMP / "merges.txt").write_text("#version: 0.2\n")
    return CLIPTokenizer(str(TMP / "vocab.json"), str(TMP / "merges.txt"), model_max_length=77)


class TinyBase:
    def __init__(self):
        torch.manual_seed(0)
        self.unet = UNet2DConditionModel(
            block_out_channels=(8, 8), layers_per_block=2, sample_size=8, in_channels=4, out_channels=4,
            down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
            cross_attention_dim=8, norm_num_groups=2)
        self.vae = tiny_vae()
        self.tokenizer = tiny_tokenizer()
        self.text_encoder = CLIPTextModel(CLIPTextConfig(
            bos_token_id=76, eos_token_id=77, pad_token_id=77, hidden_size=8, intermediate_size=16,
            num_attention_heads=2, num_hidden_layers=2, vocab_size=78, max_position_embeddings=77))
        self.scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="linear")


def tiny_vae(*a, **k):
    return AutoencoderKL(block_out_channels=(8, 8), in_channels=3, out_channels=3,
                         down_block_types=["DownEncoderBlock2D"] * 2,
                         up_block_types=["UpDecoderBlock2D"] * 2, latent_channels=4, norm_num_groups=2)


def tiny_controlnet(*a, **k):
    return ControlNetModel(block_out_channels=(8, 8), layers_per_block=2, in_channels=4,
                           down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
                           cross_attention_dim=8, conditioning_embedding_out_channels=(8, 8),
                           norm_num_groups=2)


def tiny_adapter(*a, **k):
    return MotionAdapter(block_out_channels=(8, 8), motion_layers_per_block=2,
                         motion_norm_num_groups=2, motion_num_attention_heads=2)


def patch():
    E.Engine._load_base = lambda self, model: TinyBase()
    diffusers.MotionAdapter.from_pretrained = staticmethod(tiny_adapter)
    diffusers.ControlNetModel.from_pretrained = staticmethod(tiny_controlnet)
    diffusers.AutoencoderKL.from_pretrained = staticmethod(tiny_vae)
    E.Engine._depth_maps = lambda self, frames: [np.repeat(f.mean(2, keepdims=True).astype(np.uint8), 3, 2) for f in frames]


def make_clip(path, seconds=4, fps=30, size=(160, 96)):
    w, h = size
    p = subprocess.Popen(
        [video.ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-f", "lavfi", "-i",
         f"sine=frequency=440:duration={seconds}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", str(path)], stdin=subprocess.PIPE)
    for i in range(seconds * fps):
        f = np.zeros((h, w, 3), np.uint8)
        f[:, :, 1] = 90
        x = int(i / (seconds * fps) * (w - 20))
        f[30:60, x:x + 20] = (255, 60, 0)
        p.stdin.write(f.tobytes())
    p.stdin.close()
    assert p.wait() == 0


def streams(path):
    out = subprocess.run([video.ffmpeg_exe(), "-i", str(path)], capture_output=True, text=True).stderr
    return out


def main():
    patch()
    clip = TMP / "clip.mp4"
    make_clip(clip)
    eng = E.Engine()

    # 1) engine: clip shorter than one window, exactly one, and several windows
    for n in (5, 16, 41):
        frames = [np.full((64, 64, 3), i * 5 % 255, np.uint8) for i in range(n)]
        seen = []
        out = eng.stylize(frames, E.Settings(steps=4, restyle=0.5), progress=lambda f, m: seen.append(f), log=lambda m: None)
        assert len(out) == n and all(o.shape == (64, 64, 3) and o.dtype == np.uint8 for o in out), n
        assert seen and abs(seen[-1] - 1.0) < 1e-6 and seen == sorted(seen), seen[-3:]
        print(f"ok  stylize {n} frames")

    # 2) long-window (FreeNoise) path
    frames = [np.full((64, 64, 3), i * 3 % 255, np.uint8) for i in range(40)]
    out = eng.stylize(frames, E.Settings(steps=4, restyle=0.5, long_window=True), log=lambda m: None)
    assert len(out) == 40
    print("ok  long window")
    out = eng.stylize(frames[:20], E.Settings(steps=4, restyle=0.5), log=lambda m: None)
    assert len(out) == 20
    print("ok  back to normal window")

    # 3) every style in styles.json resolves and renders end to end, with audio
    for name in E.load_styles():
        path = job.convert(eng, str(clip), E.Settings(style=name, steps=4, prompt="a red box"),
                           start=1.0, end=3.0, draw_fps=12, short_side=64, out_height=128,
                           out_path=str(TMP / "out.mp4"), log=lambda m: None)
        info = video.probe(path)
        s = streams(path)
        assert abs(info["duration"] - 2.0) < 0.2, info
        assert info["height"] == 128 and "Audio: aac" in s, (info, s[-400:])
    print(f"ok  full conversion for {len(E.load_styles())} styles: {info}")
    print("\nALL PASSED")


if __name__ == "__main__":
    main()
