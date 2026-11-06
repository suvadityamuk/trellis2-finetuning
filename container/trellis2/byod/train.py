"""trellis2-train: LoRA fine-tune a TRELLIS.2 geometry flow on trellis2-prep output.

    trellis2-train --config ss --latents <user>/<latents-dataset> --push-to <user>/<lora-repo>

--config is a preset (`ss`, `shape`), a local JSON file, an `hf://` path or an http(s) URL;
`--set key=value` overrides single fields. Every `i_save` steps the adapter is written to
--work and uploaded to --push-to under --push-path; metrics stream to Trackio. Re-running
the same command resumes from the newest checkpoint already in --push-to.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request

from huggingface_hub import HfApi, HfFileSystem, hf_hub_download, snapshot_download

TRELLIS = "/workspace/TRELLIS.2"
PRESETS = {"ss": "configs/lora/ss_flow_lora_gso.json",
           "shape": "configs/lora/slat_flow_img2shape_lora_gso.json"}
LATENTS = {"ss": "ss_latents/ss_enc_conv3d_16l8_fp16_64",
           "shape": "shape_latents/shape_enc_next_dc_f16c32_fp16_512"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="trellis2-train", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True,
                    help="'ss' | 'shape' | path.json | hf://[datasets/]user/repo/path.json | https://...")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override config fields, e.g. max_steps=500 i_save=100 lr=5e-5")
    ap.add_argument("--latents", required=True, help="trellis2-prep dataset repo id, or a local directory")
    ap.add_argument("--latent-dir", default=None,
                    help="latent folder inside --latents (default: the one matching the flow)")
    ap.add_argument("--push-to", default=None, help="model repo id that receives the checkpoints")
    ap.add_argument("--push-path", default=None, help="folder inside --push-to (default: --run-name)")
    ap.add_argument("--private", action="store_true", help="create --push-to as a private repo")
    ap.add_argument("--run-name", default=None, help="run / folder name (default: ss_lora or shape_lora)")
    ap.add_argument("--trackio-project", default="trellis2-lora")
    ap.add_argument("--trackio-space", default=os.environ.get("TRACKIO_SPACE_ID"),
                    help="Space id the dashboard syncs to (omit for a local dashboard)")
    ap.add_argument("--trackio-private", action="store_true", help="create the Trackio Space as private")
    ap.add_argument("--no-holdout", action="store_true", help="train on the held-out objects too")
    ap.add_argument("--work", default="/workspace/work/train", help="local working directory")
    return ap.parse_args()


def load_config(spec: str, overrides: list[str]) -> dict:
    if spec in PRESETS:
        text = open(os.path.join(TRELLIS, PRESETS[spec])).read()
    elif spec.startswith("hf://"):
        text = HfFileSystem().read_text(spec)
    elif spec.startswith(("http://", "https://")):
        text = urllib.request.urlopen(spec).read().decode()
    else:
        text = open(spec).read()
    cfg = json.loads(text)
    for item in overrides:
        key, _, value = item.partition("=")
        try:
            cfg[key] = json.loads(value)
        except json.JSONDecodeError:
            cfg[key] = value
    return cfg


def fetch_latents(spec: str, latent_dir: str, work: str) -> str:
    if os.path.isdir(spec):
        return os.path.abspath(spec)
    dst = os.path.join(work, "data")
    snapshot_download(spec, repo_type="dataset", local_dir=dst,
                      allow_patterns=["metadata.csv", "holdout.txt", f"{latent_dir}/*", "renders_cond/**"])
    return dst


def resume_from_hub(api: HfApi, repo: str, folder: str, out_dir: str) -> None:
    """Pull the newest adapter-<step>.safetensors in repo/folder so finetune_lora resumes from it."""
    if not api.repo_exists(repo):
        return
    steps = [(int(m.group(1)), f) for f in api.list_repo_files(repo)
             if (m := re.fullmatch(rf"{re.escape(folder)}/adapter-(\d+)\.safetensors", f))]
    if steps:
        step, path = max(steps)
        shutil.copy(hf_hub_download(repo, path), os.path.join(out_dir, f"adapter-{step}.safetensors"))
        print(f"-- resuming from {repo}/{path}")


def write_card(api: HfApi, repo: str, folder: str, cfg: dict, latents: str) -> None:
    flow = "shape" if cfg["sparse"] else "sparse-structure"
    card = f"""---
base_model: microsoft/TRELLIS.2-4B
library_name: peft
tags:
- trellis2
- lora
- image-to-3d
---

# TRELLIS.2 {flow} flow LoRA

LoRA adapter for `{cfg['ckpt']}`, trained with `trellis2-train` on
[`{latents}`](https://huggingface.co/datasets/{latents}).

| | |
|---|---|
| checkpoints | `{folder}/adapter-<step>.safetensors` (every {cfg['i_save']} steps) |
| config | `{folder}/config.json` |
| rank / alpha | {cfg['rank']} / {cfg['alpha']} |
| targets | {', '.join(f'`{t}`' for t in cfg['lora_targets'])} |
| steps | {cfg['max_steps']} at batch {cfg['batch_size']} x {cfg['grad_accum']}, lr {cfg['lr']} |

Load it by injecting a `peft.LoraConfig(r=rank, lora_alpha=alpha, target_modules=targets)`
into the flow with `inject_adapter_in_model`, then `set_peft_model_state_dict` with the
safetensors file.
"""
    if "README.md" not in api.list_repo_files(repo):
        api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=repo)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.set)
    flow = "shape" if cfg.get("sparse") else "ss"
    run_name = args.run_name or f"{flow}_lora"
    push_path = args.push_path if args.push_path is not None else run_name
    latent_dir = args.latent_dir or LATENTS[flow]

    out_dir = os.path.join(args.work, run_name)
    os.makedirs(out_dir, exist_ok=True)
    cfg_path = os.path.join(out_dir, "config.json")
    json.dump(cfg, open(cfg_path, "w"), indent=2)

    data = fetch_latents(args.latents, latent_dir, args.work)
    latent_key = "shape_latent" if flow == "shape" else "ss_latent"
    roots = json.dumps({"data": {latent_key: os.path.join(data, latent_dir),
                                 "render_cond": os.path.join(data, "renders_cond"), "meta": data}})

    api = HfApi()
    if args.push_to:
        api.create_repo(args.push_to, private=args.private, exist_ok=True)
        resume_from_hub(api, args.push_to, push_path, out_dir)
        api.upload_file(path_or_fileobj=cfg_path, path_in_repo=f"{push_path}/config.json".lstrip("/"),
                        repo_id=args.push_to, commit_message=f"{run_name}: config")
        if not os.path.isdir(args.latents):
            write_card(api, args.push_to, push_path, cfg, args.latents)
    if args.trackio_space:
        os.environ["TRACKIO_SPACE_ID"] = args.trackio_space
    if args.trackio_private:
        os.environ["TRACKIO_PRIVATE"] = "1"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    cmd = ["accelerate", "launch", "--num_processes", "1", "--mixed_precision", "bf16", "finetune_lora.py",
           "--config", cfg_path, "--data_dir", roots, "--output_dir", out_dir,
           "--trackio_project", args.trackio_project, "--run_name", run_name]
    holdout = os.path.join(data, "holdout.txt")
    if not args.no_holdout and os.path.exists(holdout):
        cmd += ["--holdout", holdout]
    if args.push_to:
        cmd += ["--push_to", args.push_to, "--push_path", push_path]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=TRELLIS, check=True)
    print("TRAIN DONE", f"https://huggingface.co/{args.push_to}" if args.push_to else out_dir)


if __name__ == "__main__":
    sys.exit(main())
