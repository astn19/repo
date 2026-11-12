# ToonClip

Turn a video clip (a file or a YouTube link) into animation: anime, comic book,
3D cartoon and more. An AI model redraws the footage on your own NVIDIA GPU, so
there are no credits or subscriptions. It is a personal take on what DomoAI's
video-to-video tool does.

## What you need

- Windows or Linux PC with an NVIDIA graphics card, 12 GB of video memory or
  more (8 GB works with `--low-vram`, slowly)
- Python 3.10 to 3.13 (python.org; tick "Add Python to PATH" on Windows)
- About 15 GB of free disk space (the AI models download on first use)

## Setup (once)

1. Unzip the folder anywhere.
2. Windows: double-click `install.bat`. Linux: run `./install.sh`.
3. The installer ends with a check list. Every line should say `[ok]`.

## Use

1. Windows: double-click `run.bat`. Linux: `./run.sh`. A page opens in your browser.
2. Upload a video or paste a YouTube link, and set the start and end seconds.
3. Pick a style and type what is in the clip ("a golfer swinging a club on a
   green golf course"). That description matters more than any other setting.
4. Press **Quick test** to see the first 16 drawings. Adjust, then press
   **Convert clip**. Results are saved in the `output` folder.

The first run of each style downloads its model (2 to 4 GB), so it is slow once.

Only convert videos you own or have permission to reuse.

## The two sliders that matter

| Slider | Lower | Higher |
|---|---|---|
| Restyle amount | Looks like the real footage with a light art pass | Fully redrawn, more "cartoon", less faithful |
| Follow the footage | Looser, more stylised shapes | Bodies, club and ball stay exactly where they were |

For sports and action start at 0.65 / 0.8. If limbs smear during a fast swing,
raise **Follow the footage** to about 1.0. If it still looks too real, raise
**Restyle amount** to 0.75.

## Getting better results

- Small figures draw badly at 512. Use resolution 768 if your card has 16 GB+,
  or crop the video closer to the player before converting.
- 12 drawings per second is the classic animation look and renders twice as
  fast as 24.
- Fast mode is about three times quicker with slightly softer detail. Good for
  finding settings; switch it off for the final render.
- Long clips are drawn in overlapping pieces and cross-faded. Expect small style
  shifts about once a second; "Long window" reduces them on 16 GB+ cards.
- Rough speed on a 12 GB card at 512: around 1 to 3 seconds per drawing, so a
  30 second clip at 12 drawings per second takes roughly 10 to 20 minutes.

## Adding styles

Styles live in `styles.json`. Each one is an art model plus the words that
describe the look. Any Stable Diffusion 1.5 checkpoint works, including
`.safetensors` files downloaded from model-sharing sites: copy a block, give it
a name, and put the file path in `models`. You can also paste a model into
"Custom art model" in the app to try it without editing the file.

## Command line

```
python cli.py doctor
python cli.py styles
python cli.py render "https://www.youtube.com/watch?v=..." --style "Comic Book" --start 12 --end 30 --prompt "a golfer swinging a club"
```

## How it works

1. `yt-dlp` fetches the video; frames are sampled at the drawing rate.
2. A depth map and an outline map are computed for every frame.
3. A Stable Diffusion 1.5 art model redraws the frames, 16 at a time, with the
   AnimateDiff motion module keeping them consistent and two ControlNets
   (depth and outlines) pinning the drawing to the real motion.
4. The pieces are cross-faded, upscaled, and the original audio is put back.

## Troubleshooting

- **"GPU ran out of memory"**: lower the resolution, turn off Long window, or
  start with `run.bat --low-vram`.
- **A style fails to load**: run `python cli.py doctor`. It lists which model
  could not be reached; replace it in `styles.json`.
- **PyTorch cannot see the GPU**: update the NVIDIA driver, then get the current
  install command from pytorch.org and run it inside the `venv`.
- **YouTube download fails**: run `venv\Scripts\python -m pip install -U yt-dlp`
  (YouTube changes often and yt-dlp updates to match).

## Licences

The art models are community models with their own licences (several are
non-commercial). Check a model's page before using output commercially.
