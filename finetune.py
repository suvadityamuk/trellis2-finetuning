"""LoRA fine-tune one TRELLIS.2 geometry flow, in one script.

    python finetune.py --flow ss    --latents <user>/<latents-dataset>
    python finetune.py --flow shape --latents ./data --push-to <user>/<lora-repo>

TRELLIS.2 generates geometry with two rectified-flow transformers: a dense
**sparse-structure** flow (coarse 64^3 occupancy) and a sparse **shape** flow (fine
detail). This script freezes one of them, injects PEFT LoRA adapters, and trains only
those on latents produced by ``preprocess.py`` (or the published
``suvadityamuk/gso-footwear-trellis2-latents``). DINOv3 encodes the conditioning renders
and stays frozen, as does everything else in the pipeline.

What it handles for you:
  * latents from a local folder or a Hub dataset repo (only the needed folders download)
  * the held-out split in ``holdout.txt`` is excluded from training
  * ``adapter-<step>.safetensors`` every ``i_save`` steps, optionally uploaded to
    ``--push-to``; re-running resumes from the newest one (local or on the Hub)
  * loss / smoothed loss / grad norm / lr to Trackio (a Space with ``--trackio-space``,
    otherwise a local dashboard), with an alert on a non-finite loss

Hyperparameters come from a preset (``--flow ss|shape``, the values used for the
footwear run) or ``--config`` (a JSON file, ``hf://`` path or URL), and single fields can
be overridden with ``--set key=value``.

Runs inside the ``cr.hf.co/suvadityamuk/trellis2-finetuning`` image, where TRELLIS.2 and
its CUDA extensions live at /workspace/TRELLIS.2 (override with ``--trellis-root``).
Needs one NVIDIA GPU with 24 GB or more and an HF token with access to DINOv3 (gated).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import urllib.request
from dataclasses import asdict, dataclass, field

# --------------------------------------------------------------------------- presets
# The two configs used for the footwear run (container/trellis2/configs/lora/*.json).
# Schedule, sigma_min and p_uncond are copied from the stock TRELLIS.2 training configs.
SHAPE_NORMALIZATION = {
    "mean": [0.781296, 0.018091, -0.495192, -0.558457, 1.060530, 0.093252, 1.518149, -0.933218,
             -0.732996, 2.604095, -0.118341, -2.143904, 0.495076, -2.179512, -2.130751, -0.996944,
             0.261421, -2.217463, 1.260067, -0.150213, 3.790713, 1.481266, -1.046058, -1.523667,
             -0.059621, 2.220780, 1.621212, 0.877230, 0.567247, -3.175944, -3.186688, 1.578665],
    "std": [5.972266, 4.706852, 5.445010, 5.209927, 5.320220, 4.547237, 5.020802, 5.444004,
            5.226681, 5.683095, 4.831436, 5.286469, 5.652043, 5.367606, 5.525084, 4.730578,
            4.805265, 5.124013, 5.530808, 5.619001, 5.103930, 5.417670, 5.269677, 5.547194,
            5.634698, 5.235274, 6.110351, 5.511298, 6.237273, 4.879207, 5.347008, 5.405691],
}
PRESETS = {
    "ss": {
        "ckpt": "microsoft/TRELLIS.2-4B/ckpts/ss_flow_img_dit_1_3B_64_bf16",
        "dataset": "ImageConditionedSparseStructureLatent",
        "dataset_args": {"min_aesthetic_score": 4.5, "image_size": 512},
        "sparse": False,
        "t_schedule": {"name": "logitNormal", "args": {"mean": 1.0, "std": 1.0}},
        "rank": 8, "alpha": 16,
        "lora_targets": ["to_qkv", "to_out", "to_q", "to_kv", "mlp.mlp.0", "mlp.mlp.2"],
    },
    "shape": {
        "ckpt": "microsoft/TRELLIS.2-4B/ckpts/slat_flow_img2shape_dit_1_3B_512_bf16",
        "dataset": "ImageConditionedSLatShape",
        "dataset_args": {"resolution": 512, "image_size": 512, "min_aesthetic_score": 4.5,
                         "max_tokens": 8192, "normalization": SHAPE_NORMALIZATION},
        "sparse": True,
        "t_schedule": {"name": "uniform", "args": {}},
        # The sparse MLP is SparseLinear over SparseTensors, which PEFT's Linear LoRA cannot
        # wrap, so the shape flow adapts attention only and compensates with a higher rank.
        "rank": 16, "alpha": 32,
        "lora_targets": ["to_qkv", "to_out", "to_q", "to_kv"],
    },
}
# Latent folder inside the preprocessed dataset that each flow trains on.
LATENT_DIRS = {"ss": "ss_latents/ss_enc_conv3d_16l8_fp16_64",
               "shape": "shape_latents/shape_enc_next_dc_f16c32_fp16_512"}


@dataclass
class Config:
    """Typed view of a LoRA hyperparameter set; unknown keys are rejected."""

    ckpt: str                       # pretrained flow, e.g. "microsoft/TRELLIS.2-4B/ckpts/ss_flow_..."
    dataset: str                    # a trellis2.datasets class name
    dataset_args: dict              # its kwargs (the dataset roots are built from --latents)
    sparse: bool                    # True for the shape flow, whose latents are SparseTensors
    t_schedule: dict                # {"name": "uniform"} or {"name": "logitNormal", "args": {...}}
    image_cond_model: str = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    image_size: int = 512
    sigma_min: float = 1e-5
    p_uncond: float = 0.1           # classifier-free-guidance dropout of the image condition
    rank: int = 8
    alpha: int = 16
    lora_targets: list = field(default_factory=lambda: ["to_qkv", "to_out", "to_q", "to_kv"])
    use_checkpoint: bool = True     # gradient-checkpoint the DiT blocks; a 1.3B flow needs it
    lr: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    batch_size: int = 4             # micro-batch; effective batch = batch_size * grad_accum
    grad_accum: int = 4
    num_workers: int = 4
    max_steps: int = 3000           # optimizer steps
    i_log: int = 10
    i_save: int = 500
    seed: int = 0

    @property
    def flow(self) -> str:
        return "shape" if self.sparse else "ss"


def load_config(flow: str | None, spec: str | None, overrides: list[str]) -> Config:
    """Preset or JSON (local / hf:// / http), then ``key=value`` overrides parsed as JSON."""
    if spec:
        if spec.startswith("hf://"):
            from huggingface_hub import HfFileSystem
            raw = json.loads(HfFileSystem().read_text(spec))
        elif spec.startswith(("http://", "https://")):
            raw = json.loads(urllib.request.urlopen(spec).read().decode())
        else:
            raw = json.load(open(spec))
    else:
        raw = json.loads(json.dumps(PRESETS[flow]))   # deep copy
    for item in overrides:
        key, _, value = item.partition("=")
        try:
            raw[key] = json.loads(value)
        except json.JSONDecodeError:
            raw[key] = value
    return Config(**raw)


# ------------------------------------------------------------------------------- data
def fetch_latents(spec: str, latent_dir: str, work: str) -> str:
    """Local folder as-is; a Hub dataset repo downloads only what training reads."""
    if os.path.isdir(spec):
        return os.path.abspath(spec)
    from huggingface_hub import snapshot_download
    dst = os.path.join(work, "data")
    snapshot_download(spec, repo_type="dataset", local_dir=dst,
                      allow_patterns=["metadata.csv", "holdout.txt", f"{latent_dir}/*", "renders_cond/**"])
    return dst


def build_loader(cfg: Config, data: str, latent_dir: str, holdout: set[str]):
    """TRELLIS.2 dataset yields {'x_0', 'cond'}; the sparse one batches x_0 into one SparseTensor."""
    from torch.utils.data import DataLoader
    from trellis2 import datasets

    latent_key = "shape_latent" if cfg.sparse else "ss_latent"
    # Each root maps a role to a folder holding a metadata.csv; the dataset joins them on sha256.
    roots = json.dumps({"data": {latent_key: os.path.join(data, latent_dir),
                                 "render_cond": os.path.join(data, "renders_cond"), "meta": data}})
    ds = getattr(datasets, cfg.dataset)(roots, **cfg.dataset_args)
    if holdout:                                           # held-out objects never reach training
        kept = [(root, sha) for (root, sha) in ds.instances if sha not in holdout]
        print(f"[holdout] excluding {len(ds.instances) - len(kept)} objects; {len(kept)} remain for training")
        ds.instances = kept
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                      num_workers=cfg.num_workers, pin_memory=True,
                      collate_fn=getattr(ds, "collate_fn", None))  # None -> default stacking (dense stage)


# ----------------------------------------------------------------------- model + LoRA
def build_denoiser(cfg: Config):
    """Pretrained flow, frozen, with trainable fp32 LoRA adapters injected in place."""
    from peft import LoraConfig, inject_adapter_in_model
    from trellis2 import models

    denoiser = models.from_pretrained(cfg.ckpt)           # <ckpt>.json + <ckpt>.safetensors from the Hub
    for block in denoiser.blocks:
        block.use_checkpoint = cfg.use_checkpoint
    lora = LoraConfig(r=cfg.rank, lora_alpha=cfg.alpha, lora_dropout=0.0,
                      bias="none", target_modules=cfg.lora_targets)
    inject_adapter_in_model(lora, denoiser)               # in place: keeps the forward the sampler calls
    for name, p in denoiser.named_parameters():
        p.requires_grad_("lora_" in name)
        if "lora_" in name:
            p.data = p.data.float()                       # fp32 master weights for the adapters
    return denoiser


# --------------------------------------------------------------- flow-matching objective
def sample_t(batch_size: int, schedule: dict, device):
    """Uniform timesteps for the shape flow, logit-normal for sparse-structure."""
    import torch
    if schedule["name"] == "uniform":
        t = torch.rand(batch_size)
    else:  # logitNormal
        a = schedule["args"]
        t = torch.sigmoid(torch.randn(batch_size) * a["std"] + a["mean"])
    return t.to(device).float()


def flow_loss(denoiser, x0, cond, cfg: Config):
    """Velocity MSE at x_t = (1-t) x0 + (sigma_min + (1-sigma_min) t) noise."""
    import torch
    import torch.nn.functional as F
    t = sample_t(x0.shape[0], cfg.t_schedule, x0.device)
    noise = x0.replace(torch.randn_like(x0.feats)) if cfg.sparse else torch.randn_like(x0)
    tv = t.view(-1, *([1] * (len(x0.shape) - 1)))
    x_t = (1 - tv) * x0 + (cfg.sigma_min + (1 - cfg.sigma_min) * tv) * noise
    pred = denoiser(x_t, t * 1000, cond)                  # timestep scaled by 1000, as in the stock trainer
    target = (1 - cfg.sigma_min) * noise - x0
    return F.mse_loss(pred.feats, target.feats) if cfg.sparse else F.mse_loss(pred, target)


# -------------------------------------------------------------------------- checkpoints
def latest_checkpoint(out_dir: str) -> tuple[int, str | None]:
    """Highest-step adapter on disk, or (0, None)."""
    best_step, best_path = 0, None
    for path in glob.glob(os.path.join(out_dir, "adapter-*.safetensors")):
        m = re.search(r"adapter-(\d+)\.safetensors$", path)
        if m and int(m.group(1)) >= best_step:
            best_step, best_path = int(m.group(1)), path
    return best_step, best_path


def pull_latest_from_hub(repo: str, folder: str, out_dir: str) -> None:
    """Copy the newest <folder>/adapter-<step>.safetensors from the Hub so training resumes from it."""
    import shutil
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    if not api.repo_exists(repo):
        return
    pattern = rf"{re.escape(folder)}/adapter-(\d+)\.safetensors" if folder else r"adapter-(\d+)\.safetensors"
    steps = [(int(m.group(1)), f) for f in api.list_repo_files(repo) if (m := re.fullmatch(pattern, f))]
    if steps and max(steps)[0] > latest_checkpoint(out_dir)[0]:
        step, path = max(steps)
        shutil.copy(hf_hub_download(repo, path), os.path.join(out_dir, f"adapter-{step}.safetensors"))
        print(f"[resume] pulled {repo}/{path}")


def save_adapter(acc, denoiser, out_dir: str, step: int, push_to: str | None, push_path: str) -> None:
    """Write the LoRA weights only (~37 MB, not the base model) and mirror them to the Hub."""
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file
    name = f"adapter-{step}.safetensors"
    path = os.path.join(out_dir, name)
    save_file(get_peft_model_state_dict(acc.unwrap_model(denoiser)), path)
    print(f"[save] {path}", flush=True)
    if push_to:
        from huggingface_hub import HfApi
        try:
            HfApi().upload_file(path_or_fileobj=path, repo_id=push_to,
                                path_in_repo=f"{push_path}/{name}".lstrip("/"),
                                commit_message=f"{os.path.basename(out_dir)}: step {step}")
            print(f"[hub] {name} -> {push_to}/{push_path}")
        except Exception as e:                            # a Hub hiccup must never sink training
            print(f"[hub] upload failed (non-fatal): {e}")


def write_model_card(repo: str, folder: str, cfg: Config, latents: str) -> None:
    from huggingface_hub import HfApi
    api = HfApi()
    if "README.md" in api.list_repo_files(repo):
        return
    source = f"[`{latents}`](https://huggingface.co/datasets/{latents})" if not os.path.isdir(latents) else "local data"
    card = f"""---
base_model: microsoft/TRELLIS.2-4B
library_name: peft
tags:
- trellis2
- lora
- image-to-3d
---

# TRELLIS.2 {"shape" if cfg.sparse else "sparse-structure"} flow LoRA

LoRA adapter for `{cfg.ckpt}`, trained with `finetune.py` on {source}.

| | |
|---|---|
| checkpoints | `{folder}/adapter-<step>.safetensors` (every {cfg.i_save} steps) |
| config | `{folder}/config.json` |
| rank / alpha | {cfg.rank} / {cfg.alpha} |
| targets | {', '.join(f'`{t}`' for t in cfg.lora_targets)} |
| steps | {cfg.max_steps} at batch {cfg.batch_size} x {cfg.grad_accum}, lr {cfg.lr} |

Load it by injecting a `peft.LoraConfig(r=rank, lora_alpha=alpha, target_modules=targets)`
into the flow with `inject_adapter_in_model`, then `set_peft_model_state_dict` with the
safetensors file.
"""
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=repo)


# ---------------------------------------------------------------------------- tracking
class NoTracker:
    """Stands in for trackio with --no-trackio."""
    class AlertLevel:
        ERROR = "error"

    def init(self, **_): pass
    def log(self, *_, **__): pass
    def alert(self, **_): pass
    def finish(self): pass


# ----------------------------------------------------------------------------- training
def train(cfg: Config, args: argparse.Namespace) -> None:
    import torch
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    from trellis2.trainers.flow_matching.mixins.image_conditioned import DinoV3FeatureExtractor
    from trellis2.utils.data_utils import cycle, recursive_to_device

    tracker = NoTracker() if args.no_trackio else __import__("trackio")
    run_name = args.run_name or f"{cfg.flow}_lora"
    push_path = args.push_path if args.push_path is not None else run_name
    latent_dir = args.latent_dir or LATENT_DIRS[cfg.flow]
    out_dir = os.path.join(args.output, run_name)
    os.makedirs(out_dir, exist_ok=True)
    json.dump(asdict(cfg), open(os.path.join(out_dir, "config.json"), "w"), indent=2)

    # Data and Hub bookkeeping first, so a bad repo id fails before minutes of model loading.
    data = fetch_latents(args.latents, latent_dir, args.output)
    holdout_file = os.path.join(data, "holdout.txt")
    holdout = set() if args.no_holdout or not os.path.exists(holdout_file) else set(open(holdout_file).read().split())
    if args.push_to:
        from huggingface_hub import HfApi
        HfApi().create_repo(args.push_to, private=args.private, exist_ok=True)
        pull_latest_from_hub(args.push_to, push_path, out_dir)
        HfApi().upload_file(path_or_fileobj=os.path.join(out_dir, "config.json"),
                            path_in_repo=f"{push_path}/config.json".lstrip("/"),
                            repo_id=args.push_to, commit_message=f"{run_name}: config")
        write_model_card(args.push_to, push_path, cfg, args.latents)

    set_seed(cfg.seed)
    acc = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=cfg.grad_accum)
    denoiser = build_denoiser(cfg)
    start_step, ckpt = latest_checkpoint(out_dir)
    if ckpt:
        try:                                              # LoRA weights only; optimizer momentum restarts
            set_peft_model_state_dict(denoiser, load_file(ckpt))
            print(f"[resume] loaded {ckpt}; continuing from step {start_step}/{cfg.max_steps}")
        except Exception as e:                            # corrupt / incompatible -> start clean
            print(f"[resume] could not load {ckpt} ({e}); starting from step 0")
            start_step = 0
    if start_step >= cfg.max_steps:
        print(f"[done] {out_dir} already has adapter-{start_step}; nothing to train")
        return

    dino = DinoV3FeatureExtractor(cfg.image_cond_model, image_size=cfg.image_size)  # frozen, gated
    dino.cuda()
    loader = build_loader(cfg, data, latent_dir, holdout)

    lora_params = [p for p in denoiser.parameters() if p.requires_grad]
    print(f"[model] {sum(p.numel() for p in lora_params) / 1e6:.1f}M trainable LoRA parameters")
    opt = torch.optim.AdamW(lora_params, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
    denoiser, opt = acc.prepare(denoiser, opt)           # not the loader: SparseTensor batches move by hand

    @torch.no_grad()
    def encode_cond(images):
        cond = dino(images)                               # (B, N, 1024) DINOv3 patch tokens
        cond[torch.rand(cond.shape[0], device=cond.device) < cfg.p_uncond] = 0  # CFG dropout
        return cond

    if args.trackio_space:
        print(f"[trackio] https://huggingface.co/spaces/{args.trackio_space}")
    tracker.init(project=args.trackio_project, name=run_name, config=asdict(cfg),
                 space_id=args.trackio_space, private=args.trackio_private)

    denoiser.train()
    step, ema_loss = start_step, None
    for batch in cycle(loader):
        x0 = recursive_to_device(batch["x_0"], acc.device)
        cond = encode_cond(batch["cond"])
        with acc.accumulate(denoiser):
            with acc.autocast():                          # bf16 forward
                loss = flow_loss(denoiser, x0, cond, cfg)
            acc.backward(loss)
            grad_norm = acc.clip_grad_norm_(lora_params, cfg.grad_clip) if acc.sync_gradients else None
            opt.step()
            opt.zero_grad()
        if not acc.sync_gradients:                        # mid-accumulation micro-step
            continue
        step += 1
        loss_val = loss.item()
        ema_loss = loss_val if ema_loss is None else 0.98 * ema_loss + 0.02 * loss_val
        if not torch.isfinite(loss):
            tracker.alert(title="Non-finite loss", level=tracker.AlertLevel.ERROR,
                          text=f"loss={loss_val} at step {step}")
        if step % cfg.i_log == 0 or step == start_step + 1:
            print(f"step {step}/{cfg.max_steps}  loss {loss_val:.4f}  ema {ema_loss:.4f}  "
                  f"grad_norm {float(grad_norm):.3f}", flush=True)
        if step % cfg.i_log == 0:
            tracker.log({"loss": loss_val, "loss_ema": ema_loss, "grad_norm": float(grad_norm),
                         "lr": opt.param_groups[0]["lr"]}, step=step)
        if step % cfg.i_save == 0 and step < cfg.max_steps:
            save_adapter(acc, denoiser, out_dir, step, args.push_to, push_path)
        if step >= cfg.max_steps:
            break

    save_adapter(acc, denoiser, out_dir, cfg.max_steps, args.push_to, push_path)
    tracker.finish()
    print("TRAIN DONE", f"https://huggingface.co/{args.push_to}" if args.push_to else out_dir)


# --------------------------------------------------------------------------------- CLI
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    which = ap.add_mutually_exclusive_group(required=True)
    which.add_argument("--flow", choices=sorted(PRESETS), help="train with the footwear preset for this flow")
    which.add_argument("--config", help="hyperparameter JSON: local path, hf://[datasets/]user/repo/file.json or URL")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override config fields (values parsed as JSON), e.g. max_steps=500 lr=5e-5")
    ap.add_argument("--latents", required=True,
                    help="preprocessed dataset: local directory or Hub dataset repo id")
    ap.add_argument("--latent-dir", default=None, help="latent folder inside --latents (default: matches the flow)")
    ap.add_argument("--output", default="/workspace/work/results",
                    help="adapters go to <output>/<run-name>/ (Hub latents download to <output>/data)")
    ap.add_argument("--run-name", default=None, help="folder and Trackio run name (default: ss_lora / shape_lora)")
    ap.add_argument("--no-holdout", action="store_true", help="also train on the objects in holdout.txt")
    ap.add_argument("--push-to", default=None, help="model repo id that receives every checkpoint (created if missing)")
    ap.add_argument("--push-path", default=None, help="folder inside --push-to (default: --run-name)")
    ap.add_argument("--private", action="store_true", help="create --push-to as private")
    ap.add_argument("--trackio-project", default="trellis2-lora")
    ap.add_argument("--trackio-space", default=os.environ.get("TRACKIO_SPACE_ID"),
                    help="Space the dashboard syncs to (default $TRACKIO_SPACE_ID; unset logs locally)")
    ap.add_argument("--trackio-private", action="store_true", help="create the Trackio Space as private")
    ap.add_argument("--no-trackio", action="store_true", help="print metrics only")
    ap.add_argument("--trellis-root", default=os.environ.get("TRELLIS2_ROOT", "/workspace/TRELLIS.2"),
                    help="TRELLIS.2 checkout with our patch applied (default $TRELLIS2_ROOT or /workspace/TRELLIS.2)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.flow, args.config, args.set)
    print(json.dumps(asdict(cfg) | {"dataset_args": "..."}, indent=2))
    if os.path.isdir(args.latents):
        args.latents = os.path.abspath(args.latents)
    args.output = os.path.abspath(args.output)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    sys.path.insert(0, args.trellis_root)
    os.chdir(args.trellis_root)                           # TRELLIS.2 resolves some paths relative to its root
    train(cfg, args)


if __name__ == "__main__":
    main()
