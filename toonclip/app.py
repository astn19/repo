"""ToonClip: the point-and-click app. Run it, then open the link it prints."""
from __future__ import annotations

import argparse

import gradio as gr

from toonclip.engine import Engine, Settings, load_styles
from toonclip.job import convert

ENGINE: Engine | None = None
LOW_VRAM = False


def engine() -> Engine:
    global ENGINE
    if ENGINE is None:
        ENGINE = Engine(low_vram=LOW_VRAM)
    return ENGINE


def run(test, file, url, start, end, style, prompt, restyle, structure, fps, size, out_h,
        steps, seed, fast, long_window, custom_model, progress=gr.Progress()):
    source = (url or "").strip() or file
    if not source:
        raise gr.Error("Upload a video or paste a YouTube link first.")
    start = float(start or 0)
    end = float(end) if end else None
    if test:  # only the first 16 drawings, to check the look before a long render
        end = start + 16 / float(fps)
    s = Settings(style=style, prompt=prompt or "", restyle=restyle, structure=structure,
                 steps=int(steps), seed=int(seed), fast=fast, long_window=long_window,
                 custom_model=custom_model or "")
    log: list[str] = []
    try:
        path = convert(engine(), source, s, start, end, float(fps), int(size),
                       None if out_h == "Same as drawing" else int(out_h),
                       progress=lambda f, m: progress(f, desc=m), log=log.append)
    except Exception as e:
        raise gr.Error(str(e))
    return path, "\n".join(log)


def build() -> gr.Blocks:
    styles = list(load_styles())
    with gr.Blocks(title="ToonClip") as ui:
        gr.Markdown("# ToonClip\nTurn a video clip into animation, drawn by AI on your own GPU.")
        with gr.Row():
            with gr.Column():
                file = gr.Video(label="Video file", sources=["upload"])
                url = gr.Textbox(label="...or a YouTube link", placeholder="https://www.youtube.com/watch?v=...")
                with gr.Row():
                    start = gr.Number(label="Start (seconds)", value=0, precision=1)
                    end = gr.Number(label="End (seconds, 0 = to the end)", value=0, precision=1)
                style = gr.Dropdown(styles, value=styles[0], label="Style")
                prompt = gr.Textbox(label="What is in the clip? (helps a lot)",
                                    placeholder="a golfer swinging a club on a green golf course")
                restyle = gr.Slider(0.3, 0.9, value=0.65, step=0.05, label="Restyle amount",
                                    info="Lower = closer to the real footage. Higher = more redrawn.")
                structure = gr.Slider(0.3, 1.3, value=0.8, step=0.05, label="Follow the footage",
                                      info="Higher keeps bodies and motion exact. Lower is looser and more cartoony.")
                with gr.Accordion("More settings", open=False):
                    fps = gr.Dropdown([8, 12, 15, 24], value=12, label="Drawings per second",
                                      info="12 is the classic animation look. 24 is smoother and takes twice as long.")
                    size = gr.Dropdown([512, 576, 640, 768], value=512, label="Drawing resolution",
                                       info="512 fits 12 GB cards. 768 keeps small figures and faces cleaner (16 GB+).")
                    out_h = gr.Dropdown(["720", "1080", "Same as drawing"], value="1080", label="Output height")
                    fast = gr.Checkbox(False, label="Fast mode (about 3x quicker, slightly softer)")
                    long_window = gr.Checkbox(False, label="Long window (smoother over time, needs 16 GB+)")
                    steps = gr.Slider(10, 40, value=20, step=1, label="Steps (quality vs time)")
                    seed = gr.Number(value=7, precision=0, label="Seed (change for a different take)")
                    custom_model = gr.Textbox(label="Custom art model (optional)",
                                              placeholder="Hugging Face id or path to a .safetensors file")
                with gr.Row():
                    test_btn = gr.Button("Quick test (first 16 drawings)")
                    go_btn = gr.Button("Convert clip", variant="primary")
            with gr.Column():
                out = gr.Video(label="Result")
                status = gr.Textbox(label="Log", lines=6)
        inputs = [file, url, start, end, style, prompt, restyle, structure, fps, size, out_h,
                  steps, seed, fast, long_window, custom_model]
        test_btn.click(lambda *a, progress=gr.Progress(): run(True, *a, progress=progress), inputs, [out, status])
        go_btn.click(lambda *a, progress=gr.Progress(): run(False, *a, progress=progress), inputs, [out, status])
    return ui


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--low-vram", action="store_true", help="for GPUs under 12 GB (slower)")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    LOW_VRAM = args.low_vram
    build().queue().launch(server_port=args.port, inbrowser=True)
