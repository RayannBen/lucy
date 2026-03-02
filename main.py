# %%
"""
Lucy-Edit (Diffusers) - WSL2 friendly script (VRAM/RAM safe)
Changes vs previous:
- height/width rounded to multiples of 16 (required by pipeline)
- num_frames constraint enforced: (num_frames - 1) % 4 == 0  -> 5, 9, 13, ...
- Adds num_inference_steps (progress bar 50/50 was steps, not frames)
- Enables VAE slicing/tiling (often fixes VAE decode OOM)
- Optional: PYTORCH_ALLOC_CONF suggestion for fragmentation (set in shell)
- Loads video once and resizes consistently
"""

from __future__ import annotations

import gc
import subprocess
from dataclasses import dataclass
from typing import List, Tuple

import torch
from PIL import Image

from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler, LucyEditPipeline
from diffusers.utils import export_to_video, load_video


# -----------------------
# Config
# -----------------------


@dataclass
class Config:
    video_path: str = "assets/lunettes_test_5.mp4"
    out_path: str = "output.mp4"
    out_1080p_path: str = "output_1080p.mp4"

    model_id: str = "decart-ai/Lucy-Edit-Dev"

    prompt: str = (
        "Remove the reflections from the sunglasses lenses. "
        "The lenses should appear as clean, dark tinted glass with no visible reflections. "
        "Keep everything else in the scene unchanged."
    )
    negative_prompt: str = ""

    # IMPORTANT: for this pipeline, (num_frames - 1) must be divisible by 4 -> 5, 9, 13, ...
    num_frames: int = 5
    guidance_scale: float = 4.0

    # Denoising steps (this is what showed 50/50). Lower = faster/less VRAM.
    num_inference_steps: int = 500

    # Working resolution for inference (must be divisible by 16)
    # 960x544 is a good "540p-ish" compromise for 1080p sources
    work_width: int = 896
    work_height: int = 512

    # Performance / memory
    use_cpu_offload: bool = True  # True if VRAM is bottleneck (uses more RAM)
    enable_attention_slicing: bool = True
    enable_vae_slicing: bool = True
    enable_vae_tiling: bool = True
    allow_tf32: bool = True

    # Export
    fps: int = 24
    upscale_to_1080p: bool = True  # requires ffmpeg in PATH


CFG = Config()


# -----------------------
# Utilities
# -----------------------


def round_to_16(x: int) -> int:
    """Diffusers Lucy-Edit requires height/width divisible by 16."""
    return int(round(x / 16)) * 16


def valid_num_frames(n: int) -> int:
    """Pipeline wants (num_frames - 1) divisible by 4. Force nearest valid."""
    if n < 1:
        return 1
    # nearest k where k-1 divisible by 4 => k in {1,5,9,13,...}
    # pick nearest by rounding (n-1)/4 then back
    k = int(round((n - 1) / 4)) * 4 + 1
    return max(1, k)


def load_video_frames(
    path: str, num_frames: int, size: Tuple[int, int]
) -> List[Image.Image]:
    """Load first N frames and resize to (width, height)."""
    width, height = size
    width, height = round_to_16(width), round_to_16(height)

    frames = load_video(path)[:num_frames]
    return [f.resize((width, height), resample=Image.BICUBIC) for f in frames]


def upscale_with_ffmpeg(in_path: str, out_path: str, width: int, height: int) -> None:
    """Upscale using ffmpeg Lanczos (fast, decent)."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        in_path,
        "-vf",
        f"scale={width}:{height}:flags=lanczos",
        "-c:a",
        "copy",
        out_path,
    ]
    subprocess.run(cmd, check=True)


# -----------------------
# Main
# -----------------------


def main(cfg: Config) -> None:
    if cfg.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Enforce constraints
    nf = valid_num_frames(cfg.num_frames)
    w = round_to_16(cfg.work_width)
    h = round_to_16(cfg.work_height)

    if nf != cfg.num_frames:
        print(
            f"[warn] Adjusted num_frames from {cfg.num_frames} -> {nf} to satisfy (num_frames-1)%4==0."
        )
    if (w, h) != (cfg.work_width, cfg.work_height):
        print(
            f"[warn] Adjusted size from {cfg.work_width}x{cfg.work_height} -> {w}x{h} (must be /16)."
        )

    print(
        f"[info] inference size: {w}x{h} | frames: {nf} | steps: {cfg.num_inference_steps}"
    )

    # 1) Load video once (at working resolution)
    video = load_video_frames(cfg.video_path, num_frames=nf, size=(w, h))

    # 2) Load pipeline
    vae = AutoencoderKLWan.from_pretrained(
        cfg.model_id,
        subfolder="vae",
        torch_dtype=torch.float32,
    )
    pipe = LucyEditPipeline.from_pretrained(
        cfg.model_id,
        vae=vae,
        torch_dtype=torch.float16,
    )

    # Force the correct scheduler (model config ships UniPC which crashes at high step counts)
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)

    # 3) Memory optimizations
    if cfg.enable_attention_slicing and hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing("max")

    # VAE decode is a common VRAM OOM point -> these two help a lot when supported
    if cfg.enable_vae_slicing and hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if cfg.enable_vae_tiling and hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()

    if cfg.use_cpu_offload and hasattr(pipe, "enable_sequential_cpu_offload"):
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to(device)

    # 4) Inference — let the pipeline handle prompt encoding & device placement
    gc.collect()
    torch.cuda.empty_cache()

    with torch.inference_mode():
        frames = pipe(
            prompt=cfg.prompt,
            negative_prompt=cfg.negative_prompt or None,
            video=video,
            height=h,
            width=w,
            num_frames=nf,
            guidance_scale=cfg.guidance_scale,
            num_inference_steps=cfg.num_inference_steps,
        ).frames[0]

    # 5) Export
    export_to_video(frames, cfg.out_path, fps=cfg.fps)
    print(f"[ok] wrote: {cfg.out_path}")

    # 6) Optional upscale to 1080p
    if cfg.upscale_to_1080p:
        try:
            upscale_with_ffmpeg(cfg.out_path, cfg.out_1080p_path, 1920, 1080)
            print(f"[ok] wrote: {cfg.out_1080p_path}")
        except FileNotFoundError:
            print(
                "[warn] ffmpeg not found; skipping 1080p upscale. Install with: sudo apt install -y ffmpeg"
            )
        except subprocess.CalledProcessError as e:
            print(f"[warn] ffmpeg upscale failed: {e}")


if __name__ == "__main__":
    main(CFG)

# %%
