#!/usr/bin/env python
import argparse, os, math, time, random
from pathlib import Path
import torch
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    DPMSolverMultistepScheduler,
)

def is_sdxl(model_id_or_path: str) -> bool:
    name = model_id_or_path.lower()
    return "sdxl" in name or "stable-diffusion-xl" in name

def seed_everything(seed: int | None):
    if seed is None:
        return None
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    return seed

def build_pipe(base, dtype):
    if is_sdxl(base):
        pipe = StableDiffusionXLPipeline.from_pretrained(
            base, torch_dtype=dtype, use_safetensors=True
        )
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            base, torch_dtype=dtype, safety_checker=None, feature_extractor=None, use_safetensors=True
        )
    # A good default sampler
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    return pipe

def apply_loras(pipe, loras: list[str], scales: list[float]):
    """
    loras: list of .safetensors paths
    scales: list of multipliers (e.g., 0.6)
    Newer diffusers supports cumulative loading; we set weights after each load.
    """
    assert len(loras) == len(scales)
    for path, scale in zip(loras, scales):
        pipe.load_lora_weights(path)
        # set lora scale for UNet; if SDXL, also for text encoders
        pipe.fuse_lora = False  # keep unfused so we can unload later if desired
        try:
            pipe.set_adapters(adapter_names=None, adapter_weights=[scale])  # diffusers>=0.31
        except Exception:
            # fallback for older diffusers
            try:
                pipe.set_lora_scale(scale)
            except Exception:
                pass
    return pipe

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="Base model id or path (SD1.5 or SDXL)")
    p.add_argument("--lora", nargs="+", required=True, help="One or more LoRA .safetensors")
    p.add_argument("--lora-scale", nargs="+", type=float, default=None, help="Scale(s) per LoRA (e.g. 0.6 0.8)")
    p.add_argument("--prompt", required=True)
    p.add_argument("--neg", default="")
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--cfg", type=float, default=5.5)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--images", type=int, default=1)
    p.add_argument("--outdir", default="outputs")
    args = p.parse_args()

    # dtype & device
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    seed_everything(args.seed)

    pipe = build_pipe(args.base, dtype=dtype).to("cuda" if torch.cuda.is_available() else "cpu")
    pipe.enable_attention_slicing()
    if torch.cuda.is_available():
        pipe.enable_model_cpu_offload()  # okay with accelerate; helps VRAM on SDXL

    # LoRA scales
    if args.lora_scale is None:
        scales = [0.6] * len(args.lora)
    else:
        if len(args.lora_scale) == 1 and len(args.lora) > 1:
            scales = [args.lora_scale[0]] * len(args.lora)
        else:
            assert len(args.lora_scale) == len(args.lora), "Provide one scale per LoRA or a single value."
            scales = args.lora_scale

    pipe = apply_loras(pipe, args.lora, scales)

    os.makedirs(args.outdir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    results = []

    # Prepare kwargs for SDXL vs SD1.5
    common = dict(num_inference_steps=args.steps, guidance_scale=args.cfg)
    if is_sdxl(args.base):
        kwargs = dict(
            prompt=args.prompt,
            negative_prompt=args.neg,
            width=args.width,
            height=args.height,
        )
    else:
        kwargs = dict(
            prompt=args.prompt,
            negative_prompt=args.neg,
            width=args.width,
            height=args.height,
        )

    for i in range(args.images):
        image = pipe(**kwargs, **common).images[0]
        fn = Path(args.outdir) / f"sample_{ts}_i{i}_seed{args.seed or 'rnd'}.png"
        image.save(fn)
        results.append(str(fn))

    print("\n".join(results))

if __name__ == "__main__":
    main()
