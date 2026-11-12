"""The AI engine: redraws a clip in a chosen art style on your own GPU.

How it works (same recipe the commercial video-to-video tools started from):

  * A Stable Diffusion 1.5 art model (anime, comic, ...) redraws the frames.
  * AnimateDiff's motion module makes it draw 16 frames *together*, so the
    result moves like animation instead of flickering frame by frame.
  * Two ControlNets pin the drawing to the real footage: a depth map (body and
    scene shape) and an edge map (outlines, club, ball, limbs).
  * Long clips are done in overlapping windows; each new window starts from the
    last drawn frames of the previous one, and the overlap is cross-faded.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
STYLES_FILE = ROOT / "styles.json"

MOTION = "guoyww/animatediff-motion-adapter-v1-5-3"
MOTION_FAST = "wangfuyun/AnimateLCM"
FAST_LORA = "AnimateLCM_sd15_t2v_lora.safetensors"
CN_DEPTH = "lllyasviel/control_v11f1p_sd15_depth"
CN_EDGE = "lllyasviel/control_v11p_sd15_canny"
VAE = "stabilityai/sd-vae-ft-mse"
DEPTH_MODELS = ["depth-anything/Depth-Anything-V2-Small-hf", "Intel/dpt-hybrid-midas"]

NEGATIVE = ("worst quality, low quality, blurry, jpeg artifacts, deformed, bad anatomy, "
            "extra limbs, extra fingers, missing limbs, fused fingers, text, watermark, logo")

WINDOW = 16  # frames AnimateDiff draws together (what the motion module was trained on)


def load_styles() -> dict:
    data = json.loads(STYLES_FILE.read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if not k.startswith("_")}


@dataclass
class Settings:
    style: str = "Anime"
    prompt: str = ""            # what is in the clip, e.g. "a golfer swinging a club"
    restyle: float = 0.65       # 0.3 = close to the footage ... 0.9 = heavily redrawn
    structure: float = 0.8      # how tightly the drawing follows the real shapes/motion
    steps: int = 20
    guidance: float = 7.0
    seed: int = 7
    fast: bool = False          # AnimateLCM: ~3x faster, slightly softer detail
    long_window: bool = False   # 32-frame windows (smoother, needs ~16 GB+ VRAM)
    custom_model: str = ""      # overrides the style's model (repo id or .safetensors path)


class Engine:
    def __init__(self, low_vram: bool = False, device: str | None = None):
        import torch

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.low_vram = low_vram
        self.pipe = None
        self._key = None
        self._depth = None

    # ------------------------------------------------------------- loading

    def _load_base(self, model: str):
        from diffusers import StableDiffusionPipeline

        kw = dict(torch_dtype=self.dtype, safety_checker=None, requires_safety_checker=False)
        if model.lower().endswith((".safetensors", ".ckpt")):
            return StableDiffusionPipeline.from_single_file(model, **kw)
        return StableDiffusionPipeline.from_pretrained(model, **kw)

    def load(self, models: list[str], fast: bool, log=print):
        """Build (or reuse) the pipeline for the first model in `models` that loads."""
        from diffusers import (AnimateDiffVideoToVideoControlNetPipeline, AutoencoderKL,
                               ControlNetModel, DDIMScheduler, LCMScheduler, MotionAdapter)

        key = (tuple(models), fast)
        if self.pipe is not None and self._key == key:
            return self.pipe
        self.pipe = None
        self._key = None
        if self.device == "cuda":
            self.torch.cuda.empty_cache()

        base, errors = None, []
        for m in models:
            try:
                log(f"Loading art model: {m}")
                base = self._load_base(m)
                break
            except Exception as e:  # try the next fallback
                errors.append(f"  {m}: {type(e).__name__}: {str(e)[:200]}")
        if base is None:
            raise RuntimeError("None of the style's models could be loaded:\n" + "\n".join(errors))

        log("Loading motion module and ControlNets")
        adapter = MotionAdapter.from_pretrained(MOTION_FAST if fast else MOTION, torch_dtype=self.dtype)
        controlnets = [ControlNetModel.from_pretrained(r, torch_dtype=self.dtype) for r in (CN_DEPTH, CN_EDGE)]
        try:  # a better colour decoder than most checkpoints ship with
            vae = AutoencoderKL.from_pretrained(VAE, torch_dtype=self.dtype)
        except Exception:
            vae = base.vae

        sched_kw = dict(beta_schedule="linear", clip_sample=False,
                        timestep_spacing="linspace", steps_offset=1)
        if fast:
            scheduler = LCMScheduler.from_config(base.scheduler.config, beta_schedule="linear")
        else:
            scheduler = DDIMScheduler.from_config(base.scheduler.config, **sched_kw)

        pipe = AnimateDiffVideoToVideoControlNetPipeline(
            vae=vae, text_encoder=base.text_encoder, tokenizer=base.tokenizer, unet=base.unet,
            motion_adapter=adapter, controlnet=controlnets, scheduler=scheduler,
        )
        if fast:
            pipe.load_lora_weights(MOTION_FAST, weight_name=FAST_LORA, adapter_name="lcm")
            pipe.set_adapters(["lcm"], [0.8])
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
        if self.low_vram and self.device == "cuda":
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(self.device)
        pipe.set_progress_bar_config(disable=True)
        self.pipe, self._key = pipe, key
        return pipe

    # ---------------------------------------------------- guidance from footage

    def _depth_maps(self, frames):
        from PIL import Image

        if self._depth is None:
            from transformers import pipeline

            err = None
            for repo in DEPTH_MODELS:
                try:
                    self._depth = pipeline("depth-estimation", model=repo,
                                           device=0 if self.device == "cuda" else -1)
                    break
                except Exception as e:
                    err = e
            if self._depth is None:
                raise RuntimeError(f"Could not load a depth model: {err}")
        h, w = frames[0].shape[:2]
        out = []
        for i in range(0, len(frames), 8):
            res = self._depth([Image.fromarray(f) for f in frames[i:i + 8]])
            for r in res:
                d = np.asarray(r["depth"].convert("L").resize((w, h)))
                out.append(np.repeat(d[..., None], 3, axis=2))
        return out

    @staticmethod
    def _edge_maps(frames):
        out = []
        for f in frames:
            # smooth away grass/crowd texture first so only real outlines remain
            g = cv2.bilateralFilter(f, 9, 60, 7)
            g = cv2.cvtColor(g, cv2.COLOR_RGB2GRAY)
            e = cv2.Canny(g, 60, 140)
            out.append(np.repeat(e[..., None], 3, axis=2))
        return out

    # ------------------------------------------------------------- rendering

    def stylize(self, frames: list, s: Settings, progress=None, log=print) -> list:
        """frames: list of RGB uint8 arrays (all the same size). Returns the same, redrawn."""
        from PIL import Image

        torch = self.torch
        styles = load_styles()
        if s.style not in styles:
            raise ValueError(f"Unknown style {s.style!r}. Available: {', '.join(styles)}")
        st = styles[s.style]
        models = [s.custom_model.strip()] if s.custom_model.strip() else list(st["models"])
        pipe = self.load(models, s.fast, log)

        # the clip description goes first so it is never cut off by the 77-token limit
        prompt = ", ".join(p for p in (s.prompt.strip(), st["prompt"]) if p)
        negative = ", ".join(p for p in (st.get("negative", ""), NEGATIVE) if p)
        steps = max(4, min(int(s.steps), 8)) if s.fast else int(s.steps)
        guidance = min(float(s.guidance), 2.0) if s.fast else float(s.guidance)
        # the scheduler only runs steps*restyle of the steps; keep at least 2
        restyle = float(np.clip(s.restyle, max(0.1, 2.0 / steps), 1.0))
        cn_scale = [float(s.structure), float(s.structure) * 0.75]

        window = WINDOW
        if s.long_window and hasattr(pipe, "enable_free_noise"):
            window = 32
            pipe.enable_free_noise(context_length=16, context_stride=4)
        elif hasattr(pipe, "disable_free_noise"):
            pipe.disable_free_noise()
        overlap = 4
        n = len(frames)
        h, w = frames[0].shape[:2]

        log("Reading depth and outlines from the footage")
        depth = self._depth_maps(frames)
        edges = self._edge_maps(frames)

        # window start positions; the last window is shifted back so it is full length
        if n <= window:
            starts = [0]
        else:
            starts = list(range(0, n - window, window - overlap)) + [n - window]
            starts = sorted(set(starts))

        result: list = [None] * n
        for wi, a in enumerate(starts):
            b = min(a + window, n)
            src = [f for f in frames[a:b]]
            # continuity: start from frames we already drew, not from raw footage
            for j in range(a, b):
                if result[j] is not None:
                    src[j - a] = result[j]
            pad = window - len(src) if n < window else 0  # very short clips: repeat last frame
            idx = list(range(a, b)) + [b - 1] * pad
            src = src + [src[-1]] * pad

            def cb(_pipe, step, _t, kwargs, wi=wi):
                if progress:
                    done_steps = max(int(steps * restyle), 1)
                    progress((wi + (step + 1) / done_steps) / len(starts),
                             f"Drawing part {wi + 1} of {len(starts)}")
                return kwargs

            gen = torch.Generator("cpu").manual_seed(int(s.seed))
            try:
                with torch.inference_mode():
                    out = pipe(
                        video=[Image.fromarray(f) for f in src],
                        conditioning_frames=[[Image.fromarray(depth[i]) for i in idx],
                                             [Image.fromarray(edges[i]) for i in idx]],
                        prompt=prompt, negative_prompt=negative,
                        height=h, width=w,
                        strength=restyle, num_inference_steps=steps, guidance_scale=guidance,
                        controlnet_conditioning_scale=cn_scale,
                        generator=gen, output_type="np",
                        callback_on_step_end=cb,
                    ).frames[0]
            except torch.OutOfMemoryError as e:
                raise RuntimeError(
                    "Your GPU ran out of memory. Lower the working resolution, turn off "
                    "'Long window', or start the app with --low-vram."
                ) from e
            drawn = [(np.clip(f, 0, 1) * 255).round().astype(np.uint8) for f in out]

            m = sum(1 for j in range(a, b) if result[j] is not None)  # overlap length
            for k, j in enumerate(range(a, b)):
                if result[j] is None:
                    result[j] = drawn[k]
                else:  # cross-fade from the previous window into this one
                    t = (k + 1) / (m + 1)
                    result[j] = cv2.addWeighted(drawn[k], t, result[j], 1 - t, 0)
        return result
