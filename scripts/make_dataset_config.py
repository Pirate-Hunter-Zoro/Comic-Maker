#!/usr/bin/env python
"""
Generate a kohya dataset .config for a character.
Writes: projects/<universe>/configs/<character>_dataset.config
"""
import json, argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True, help="Repo root, e.g. /home/.../Comic-Maker")
    ap.add_argument("--universe", required=True, help="e.g. rwby_post_ever_after")
    ap.add_argument("--character", required=True, help="e.g. blake")
    ap.add_argument("--image-dir", required=True,
                    help="Relative to project root, e.g. projects/<universe>/data/datasets/blake_synth")
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--min-bucket", type=int, default=512)
    ap.add_argument("--max-bucket", type=int, default=1536)
    ap.add_argument("--caption-extension", default=".txt")
    args = ap.parse_args()

    project = Path(args.project_root)
    cfg_dir = project / "projects" / args.universe / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg_dir / f"{args.character}_dataset.config"

    config = {
        "datasets": [{
            "resolution": args.resolution,
            "min_bucket_reso": args.min_bucket,
            "max_bucket_reso": args.max_bucket,
            "caption_extension": args.caption_extension,
            "data": [{
                "image_dir": args.image_dir,
                "class_tokens": "",
                "caption_separator": "\n",
                "shuffle_caption": False
            }]
        }]
    }

    out_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
