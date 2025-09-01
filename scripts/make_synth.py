#!/usr/bin/env python
import argparse, random, json, math
from pathlib import Path
from typing import List, Tuple, Dict, Any
from PIL import Image, ImageOps
import torch
from diffusers import (
    StableDiffusionImg2ImgPipeline,
    StableDiffusionXLImg2ImgPipeline,
    DPMSolverMultistepScheduler,
)

DEFAULT_PROMPTS = [
    "anime style, {char}, soft rim light, clean lineart, comic panel",
    "stylized anime portrait, {char}, halftone shading, crisp inks",
    "dynamic half body, {char}, dramatic lighting, cinematic, high detail lineart",
    "full body, {char}, action pose, smoky background, comic shading",
    "close-up, {char}, subtle blush, studio lighting, high fidelity anime",
]
DEFAULT_NEG = "lowres, blurry, jpeg artifacts, extra limbs, extra fingers, text, watermark, mutated, disfigured"
DEFAULT_DENOISE = [0.35, 0.45, 0.55, 0.65]

def is_sdxl(model_id: str) -> bool:
    m = model_id.lower()
    return "sdxl" in m or "stable-diffusion-xl" in m

def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def _flatten_prompts(prompts_raw: Any) -> List[str]:
    if isinstance(prompts_raw, dict):
        out: List[str] = []
        for _, lst in prompts_raw.items():
            if isinstance(lst, list):
                out += [s for s in lst if isinstance(s, str) and s.strip()]
        return out
    if isinstance(prompts_raw, list):
        return [s for s in prompts_raw if isinstance(s, str) and s.strip()]
    return []

def _norm_counts(weights: Dict[str, float], total: int) -> Dict[str, int]:
    # proportional rounding that sums exactly to total
    keys = list(weights.keys())
    raw = {k: weights[k] * total for k in keys}
    floored = {k: int(math.floor(raw[k])) for k in keys}
    remainder = total - sum(floored.values())
    # distribute the remainder to buckets with largest fractional parts
    fracs = sorted(keys, key=lambda k: raw[k] - floored[k], reverse=True)
    for i in range(remainder):
        floored[fracs[i % len(fracs)]] += 1
    return floored

def load_recipe(character: str, prompts_file: str | None) -> Dict[str, Any]:
    """
    Supports:
      - TXT: one prompt per line (flat mode)
      - JSON flat: {"prompts":[...], "negatives":[...], "denoise_strengths":[...], "cfg":6.0, "steps":28}
      - JSON buckets: see advanced schema (global + buckets)
    Returns a dict with a normalized plan:
      {
        "global": { "neg": str, "cfg": float, "steps": int },
        "buckets": [
          {"name":"portrait","count":100,"size":[768,1024],"denoise":[...],"prompts":[...],"neg":str},
          ...
        ],
        "total": 300
      }
    """
    # Default: flat recipe using built-ins
    if not prompts_file:
        return {
            "global": {"neg": DEFAULT_NEG, "cfg": None, "steps": None},
            "buckets": [{
                "name": "default",
                "count": None,  # to be filled by CLI --n
                "size": None,   # to be filled by CLI W,H
                "denoise": DEFAULT_DENOISE,
                "prompts": DEFAULT_PROMPTS,
                "neg": None
            }],
            "total": None
        }

    p = Path(prompts_file)
    if p.suffix.lower() == ".txt":
        lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        prompts = lines or DEFAULT_PROMPTS
        return {
            "global": {"neg": DEFAULT_NEG, "cfg": None, "steps": None},
            "buckets": [{
                "name": "default",
                "count": None,
                "size": None,
                "denoise": DEFAULT_DENOISE,
                "prompts": prompts,
                "neg": None
            }],
            "total": None
        }

    if p.suffix.lower() == ".json":
        data = _read_json(p)
        # advanced?
        if "buckets" in data:
            g = data.get("global", {})
            total = int(g.get("count")) if isinstance(g.get("count"), int) else None
            g_steps = int(g["steps"]) if isinstance(g.get("steps"), int) else None
            g_cfg = float(g["cfg"]) if isinstance(g.get("cfg"), (int, float)) else None

            # global negatives
            gn = g.get("negatives", DEFAULT_NEG)
            if isinstance(gn, list):
                g_neg = ", ".join([s for s in gn if isinstance(s, str) and s.strip()]) or DEFAULT_NEG
            elif isinstance(gn, str):
                g_neg = gn.strip() or DEFAULT_NEG
            else:
                g_neg = DEFAULT_NEG

            g_dnz = g.get("denoise_strengths", DEFAULT_DENOISE)
            if not (isinstance(g_dnz, list) and g_dnz):
                g_dnz = DEFAULT_DENOISE

            # gather buckets
            buckets = []
            weights = {}
            pending_count = 0
            for name, spec in data["buckets"].items():
                spec = spec or {}
                bp = _flatten_prompts(spec.get("prompts", []))
                if not bp:
                    continue
                # per-bucket negatives merged with global
                bn = spec.get("negatives", None)
                if isinstance(bn, list):
                    bneg = g_neg + ", " + ", ".join([s for s in bn if isinstance(s, str) and s.strip()]) if bn else g_neg
                elif isinstance(bn, str) and bn.strip():
                    bneg = g_neg + ", " + bn.strip()
                else:
                    bneg = g_neg
                # sizes & denoise
                bsize = spec.get("size", None)  # [w,h]
                bdnz = spec.get("denoise_strengths", g_dnz)
                if not (isinstance(bdnz, list) and bdnz):
                    bdnz = g_dnz
                # counts
                if isinstance(spec.get("count"), int):
                    bcount = int(spec["count"])
                    pending_count += bcount
                    bweight = None
                else:
                    bcount = None
                    if isinstance(spec.get("weight"), (int, float)):
                        weights[name] = float(spec["weight"])
                buckets.append({
                    "name": name, "count": bcount, "weight": weights.get(name),
                    "size": bsize, "denoise": bdnz, "prompts": bp, "neg": bneg
                })

            # normalize counts from weights if needed
            if total is None:
                total = pending_count if pending_count > 0 else None
            if total is not None:
                # Fill missing counts proportionally by weights; if no weights, split evenly
                missing = [b for b in buckets if b["count"] is None]
                if missing:
                    if weights and sum(weights.values()) > 0:
                        counts = _norm_counts(weights, total - pending_count)
                    else:
                        # even split
                        even = (total - pending_count) // max(len(missing), 1)
                        counts = {b["name"]: even for b in missing}
                        # fix rounding leftovers
                        leftover = (total - pending_count) - even * len(missing)
                        for i in range(leftover):
                            counts[missing[i]["name"]] += 1
                    for b in buckets:
                        if b["count"] is None:
                            b["count"] = counts.get(b["name"], 0)
            return {
                "global": {"neg": g_neg, "cfg": g_cfg, "steps": g_steps},
                "buckets": buckets,
                "total": total
            }

        # flat json
        prompts = _flatten_prompts(data.get("prompts", []))
        if not prompts:
            prompts = DEFAULT_PROMPTS
        neg_raw = data.get("negatives", DEFAULT_NEG)
        if isinstance(neg_raw, list):
            neg = ", ".join([s for s in neg_raw if isinstance(s, str) and s.strip()]) or DEFAULT_NEG
        elif isinstance(neg_raw, str):
            neg = neg_raw.strip() or DEFAULT_NEG
        else:
            neg = DEFAULT_NEG
        dnz = data.get("denoise_strengths", DEFAULT_DENOISE)
        if not (isinstance(dnz, list) and dnz):
            dnz = DEFAULT_DENOISE
        g_cfg = float(data["cfg"]) if isinstance(data.get("cfg"), (int, float)) else None
        g_steps = int(data["steps"]) if isinstance(data.get("steps"), int) else None
        return {
            "global": {"neg": neg, "cfg": g_cfg, "steps": g_steps},
            "buckets": [{
                "name": "default", "count": None, "size": None,
                "denoise": dnz, "prompts": prompts, "neg": None
            }],
            "total": None
        }

    raise ValueError("prompts_file must be .txt or .json")

def load_pipe(base, dtype):
    if is_sdxl(base):
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(base, torch_dtype=dtype, use_safetensors=True)
    else:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(base, torch_dtype=dtype, safety_checker=None, feature_extractor=None, use_safetensors=True)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    if torch.cuda.is_available():
        try: pipe.enable_xformers_memory_efficient_attention()
        except Exception: pass
        pipe.enable_model_cpu_offload()
    return pipe

def prepare_init(img_path, width, height):
    im = Image.open(img_path).convert("RGB")
    im = ImageOps.exif_transpose(im)
    im = im.resize((width, height), Image.LANCZOS)
    return im

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--character", required=True, help="short key, e.g. blake, weiss")
    ap.add_argument("--base", default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--anchor", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--prompts-file", default=None, help=".txt or .json (advanced)")
    ap.add_argument("--neg", default=None, help="Override negative prompt (string)")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--w", type=int, default=768)
    ap.add_argument("--h", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--cfg", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    # Recipe
    recipe = load_recipe(args.character, args.prompts_file)
    # CLI precedence over file-level defaults
    steps = args.steps
    cfg = args.cfg
    if recipe["global"]["steps"] is not None and args.steps == 28:
        steps = recipe["global"]["steps"]
    if recipe["global"]["cfg"] is not None and abs(args.cfg - 6.0) < 1e-9:
        cfg = recipe["global"]["cfg"]

    global_neg = recipe["global"]["neg"]
    if args.neg is not None:
        global_neg = args.neg

    total_from_file = recipe["total"]
    total = args.n if total_from_file is None else total_from_file

    # Seed
    if args.seed is not None:
        torch.manual_seed(args.seed); random.seed(args.seed)

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipe = load_pipe(args.base, dtype=dtype)

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    # Loop buckets
    bucket_index = 0
    for b in recipe["buckets"]:
        bname = b["name"]
        bcount = b["count"]
        if bcount is None:
            # If counts were not set in JSON and not weight-derived, split evenly across buckets.
            bcount = total // len(recipe["buckets"])
            if bucket_index < (total % len(recipe["buckets"])):
                bcount += 1
        bucket_index += 1

        # sizes: prefer per-bucket, else CLI
        if b["size"] and isinstance(b["size"], list) and len(b["size"]) == 2:
            w, h = int(b["size"][0]), int(b["size"][1])
        else:
            w, h = args.w, args.h

        init = prepare_init(args.anchor, w, h)
        denoise_grid = b["denoise"] if (isinstance(b["denoise"], list) and b["denoise"]) else DEFAULT_DENOISE

        # negatives: bucket neg overrides (already merged), else global
        neg = b["neg"] if b["neg"] else global_neg

        for i in range(bcount):
            prompt = random.choice(b["prompts"]).replace("{char}", args.character)
            denoise = random.choice(denoise_grid)
            gen = torch.Generator(device=pipe.device).manual_seed(random.randint(0, 10_000_000))
            image = pipe(
                prompt=prompt,
                negative_prompt=neg,
                image=init,
                strength=denoise,
                guidance_scale=cfg,
                num_inference_steps=steps,
                generator=gen
            ).images[0]

            stem = f"{args.character}_{bname}_{i:05d}"
            img_path = out / f"{stem}.png"
            cap_path = out / f"{stem}.txt"
            image.save(img_path)
            cap_path.write_text(prompt, encoding="utf-8")

    print(f"Done. Wrote ~{total} images to {out}")

if __name__ == "__main__":
    main()
