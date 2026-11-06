"""LoRA fine-tuning for the TRELLIS.2 geometry flows (sparse-structure and shape).

One script, two stages. The config's ``sparse`` flag switches between the dense
sparse-structure DiT (``SparseStructureFlowModel``) and the SparseTensor shape DiT
(``SLatFlowModel``). TRELLIS.2 supplies the pretrained model, the dataset + collate,
and the frozen DINOv3 encoder; PEFT supplies the LoRA adapters; Accelerate supplies
the bf16 + gradient-accumulation loop. Nothing else from the stock trainer is used.

    accelerate launch --num_processes 1 --mixed_precision bf16 finetune_lora.py \
        --config configs/lora/ss_flow_lora_gso.json \
        --data_dir "$DATA_ROOTS_JSON" --output_dir results/ss_lora
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from dataclasses import asdict, dataclass, field

import torch
import torch.nn.functional as F
import trackio
from accelerate import Accelerator
from accelerate.utils import set_seed
from huggingface_hub import HfApi
from peft import (LoraConfig, get_peft_model_state_dict, inject_adapter_in_model,
                  set_peft_model_state_dict)
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader

from trellis2 import datasets, models
from trellis2.trainers.flow_matching.mixins.image_conditioned import DinoV3FeatureExtractor
from trellis2.utils.data_utils import cycle, recursive_to_device


# --------------------------------------------------------------------------- config
@dataclass
class Config:
    """Typed view of a LoRA hyperparameter JSON (schedule/sigma_min/p_uncond copied from the stock config)."""

    ckpt: str                       # e.g. "microsoft/TRELLIS.2-4B/ckpts/ss_flow_img_dit_1_3B_64_bf16"
    dataset: str                    # a trellis2.datasets class name
    dataset_args: dict              # its kwargs; the dataset roots are passed separately as --data_dir
    sparse: bool                    # True for the shape flow, whose latents are SparseTensors
    t_schedule: dict                # {"name": "uniform"} or {"name": "logitNormal", "args": {...}}
    image_cond_model: str = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    image_size: int = 512
    sigma_min: float = 1e-5
    p_uncond: float = 0.1
    rank: int = 8
    alpha: int = 16
    # Attention projections are plain nn.Linear (safe for PEFT) in both stages. The sparse MLP
    # is SparseLinear over SparseTensors, which PEFT's Linear LoRA cannot wrap, so the shape
    # config targets attention only and compensates with a higher rank.
    lora_targets: list = field(default_factory=lambda: ["to_qkv", "to_out", "to_q", "to_kv"])
    use_checkpoint: bool = True     # gradient-checkpoint the DiT blocks; a 1.3B flow OOMs an 80GB A100 without it
    lr: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    batch_size: int = 4             # per-step micro-batch; effective batch = batch_size * grad_accum
    grad_accum: int = 4
    num_workers: int = 4
    max_steps: int = 3000
    i_log: int = 10
    i_save: int = 500               # frequent, small adapter checkpoints -> fine-grained resume after a timeout
    seed: int = 0


# --------------------------------------------------------------------- model + LoRA
def build_denoiser(cfg: Config) -> torch.nn.Module:
    """Load the pretrained flow, enable checkpointing, and inject trainable LoRA adapters."""
    denoiser = models.from_pretrained(cfg.ckpt)           # builds from <ckpt>.json, loads <ckpt>.safetensors
    for block in denoiser.blocks:                         # DiT blocks checkpoint with use_reentrant=False
        block.use_checkpoint = cfg.use_checkpoint
    lora = LoraConfig(r=cfg.rank, lora_alpha=cfg.alpha, lora_dropout=0.0,
                      bias="none", target_modules=cfg.lora_targets)
    inject_adapter_in_model(lora, denoiser)               # in place: keeps the custom forward the sampler needs
    for name, p in denoiser.named_parameters():           # train LoRA only, and keep it as an fp32 master weight
        p.requires_grad_("lora_" in name)
        if "lora_" in name:
            p.data = p.data.float()
    return denoiser


# ------------------------------------------------------------ flow-matching objective
def sample_t(batch_size: int, schedule: dict, device: torch.device) -> torch.Tensor:
    """Rectified-flow timestep sampler: uniform for the shape flow, logit-normal for sparse-structure."""
    if schedule["name"] == "uniform":
        t = torch.rand(batch_size)
    else:  # logitNormal
        a = schedule["args"]
        t = torch.sigmoid(torch.randn(batch_size) * a["std"] + a["mean"])
    return t.to(device).float()


def flow_loss(denoiser: torch.nn.Module, x0, cond: torch.Tensor, cfg: Config) -> torch.Tensor:
    """Velocity MSE of x_t = (1-t) x0 + (sigma_min + (1-sigma_min) t) noise; SparseTensors share the math."""
    t = sample_t(x0.shape[0], cfg.t_schedule, x0.device)
    noise = x0.replace(torch.randn_like(x0.feats)) if cfg.sparse else torch.randn_like(x0)
    tv = t.view(-1, *([1] * (len(x0.shape) - 1)))
    x_t = (1 - tv) * x0 + (cfg.sigma_min + (1 - cfg.sigma_min) * tv) * noise
    pred = denoiser(x_t, t * 1000, cond)                  # timestep scaled by 1000, exactly as the stock trainer
    target = (1 - cfg.sigma_min) * noise - x0
    return F.mse_loss(pred.feats, target.feats) if cfg.sparse else F.mse_loss(pred, target)


# ---------------------------------------------------------------------------- data
def build_loader(cfg: Config, data_dir: str, holdout: set[str] | None = None) -> DataLoader:
    """TRELLIS.2 dataset yields {'x_0', 'cond'}; the sparse dataset batches x_0 into one SparseTensor via collate_fn."""
    ds = getattr(datasets, cfg.dataset)(data_dir, **cfg.dataset_args)
    if holdout:                                           # drop eval objects so they never leak into training
        kept = [(root, sha) for (root, sha) in ds.instances if sha not in holdout]
        print(f"[holdout] excluding {len(ds.instances) - len(kept)} held-out objects; {len(kept)} remain for training")
        ds.instances = kept
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                      num_workers=cfg.num_workers, pin_memory=True,
                      collate_fn=getattr(ds, "collate_fn", None))  # None -> default stacking for the dense stage


# ------------------------------------------------------------------------- training
def latest_checkpoint(out_dir: str) -> tuple[int, str | None]:
    """Highest-step adapter already on disk (0, None if none) — the point a re-run resumes from."""
    best_step, best_path = 0, None
    for path in glob.glob(os.path.join(out_dir, "adapter-*.safetensors")):
        m = re.search(r"adapter-(\d+)\.safetensors$", path)
        if m and int(m.group(1)) >= best_step:
            best_step, best_path = int(m.group(1)), path
    return best_step, best_path


def save_adapter(acc: Accelerator, denoiser: torch.nn.Module, out_dir: str, step: int,
                 push_to: str | None = None, push_path: str = "") -> None:
    """Write the few-MB PEFT adapter (not the base model), and mirror it to the Hub now.

    Adapters live under the cached $ROOT tree, but the job's between-stage cache push only
    runs after a whole flow finishes. Uploading each checkpoint here makes mid-flow progress
    durable, so if the job hits its wall-clock cap a re-run resumes from the last saved step
    (via ``latest_checkpoint``) instead of retraining the flow from scratch.
    ``push_to`` is a model repo (``<push_path>/adapter-<step>.safetensors``); ``CACHE_REPO``
    is the dataset-repo mirror used by ``jobs/run_full.sh``.
    """
    name = f"adapter-{step}.safetensors"
    path = os.path.join(out_dir, name)
    save_file(get_peft_model_state_dict(acc.unwrap_model(denoiser)), path)
    flow = os.path.basename(out_dir.rstrip("/"))           # e.g. ss_lora -> adapters/ss_lora/adapter-<step>.safetensors
    targets = [(push_to, "model", f"{push_path}/{name}".lstrip("/"))] if push_to else []
    if os.environ.get("CACHE_REPO"):
        targets.append((os.environ["CACHE_REPO"], "dataset", f"adapters/{flow}/{name}"))
    for repo, repo_type, path_in_repo in targets:
        try:
            HfApi().upload_file(path_or_fileobj=path, repo_id=repo, repo_type=repo_type,
                                path_in_repo=path_in_repo, commit_message=f"{flow}: step {step}")
            print(f"[hub] {name} -> {repo}/{path_in_repo}")
        except Exception as e:                             # a hub hiccup must never sink training
            print(f"[hub] adapter upload to {repo} failed (non-fatal): {e}")


def train(cfg: Config, data_dir: str, out_dir: str, holdout_path: str | None = None,
          push_to: str | None = None, push_path: str = "",
          trackio_project: str = "trellis2-geoflow-lora", run_name: str | None = None) -> None:
    set_seed(cfg.seed)
    os.makedirs(out_dir, exist_ok=True)
    acc = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=cfg.grad_accum)

    denoiser = build_denoiser(cfg)
    start_step, ckpt = latest_checkpoint(out_dir)          # restored from cache by a prior, timed-out job?
    if ckpt:
        try:                                               # reload only the LoRA weights; optimizer momentum restarts
            set_peft_model_state_dict(denoiser, load_file(ckpt))
            print(f"[resume] loaded {ckpt}; continuing from step {start_step}/{cfg.max_steps}")
        except Exception as e:                             # corrupt/incompatible checkpoint -> start clean rather than crash
            print(f"[resume] could not load {ckpt} ({e}); starting from step 0")
            start_step = 0
    dino = DinoV3FeatureExtractor(cfg.image_cond_model, image_size=cfg.image_size)  # frozen, no_grad, gated
    dino.cuda()
    holdout = set(open(holdout_path).read().split()) if holdout_path and os.path.exists(holdout_path) else set()
    loader = build_loader(cfg, data_dir, holdout)

    lora_params = [p for p in denoiser.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(lora_params, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
    denoiser, opt = acc.prepare(denoiser, opt)            # not the loader: SparseTensor batches are moved by hand

    @torch.no_grad()
    def encode_cond(images: torch.Tensor) -> torch.Tensor:
        cond = dino(images)                               # (B, N, 1024) DINOv3 patch tokens
        cond[torch.rand(cond.shape[0], device=cond.device) < cfg.p_uncond] = 0  # CFG dropout to the zero cond
        return cond

    if acc.is_main_process:
        # space_id -> a hosted HF Space so runs persist after the (ephemeral) job exits;
        # unset falls back to a local dashboard. Spaces are public by default; set
        # TRACKIO_PRIVATE=1 to keep the metrics private.
        trackio.init(project=trackio_project, name=run_name or os.path.basename(out_dir.rstrip("/")),
                     config=asdict(cfg), space_id=os.environ.get("TRACKIO_SPACE_ID"),
                     private=os.environ.get("TRACKIO_PRIVATE", "0") == "1")

    denoiser.train()
    opt_step, ema_loss = start_step, None                 # pick up where a resumed checkpoint left off
    if opt_step >= cfg.max_steps:                          # already complete on disk: just re-emit and exit
        if acc.is_main_process:
            trackio.finish()
        return
    for batch in cycle(loader):
        x0 = recursive_to_device(batch["x_0"], acc.device)
        cond = encode_cond(batch["cond"])
        with acc.accumulate(denoiser):
            with acc.autocast():                          # bf16 forward; mse runs fp32 under autocast's loss policy
                loss = flow_loss(denoiser, x0, cond, cfg)
            acc.backward(loss)
            grad_norm = acc.clip_grad_norm_(lora_params, cfg.grad_clip) if acc.sync_gradients else None
            opt.step()
            opt.zero_grad()
        if not acc.sync_gradients:                        # micro-step of an accumulation group: nothing to log yet
            continue
        opt_step += 1
        loss_val = loss.item()
        ema_loss = loss_val if ema_loss is None else 0.98 * ema_loss + 0.02 * loss_val  # per-step MSE is noisy; smooth it
        if acc.is_main_process:
            if not torch.isfinite(loss):                  # diverged / NaN: surface on the dashboard, not just stdout
                trackio.alert(title="Non-finite loss", level=trackio.AlertLevel.ERROR,
                              text=f"loss={loss_val} at step {opt_step}")
            if opt_step % cfg.i_log == 0:
                trackio.log({"loss": loss_val, "loss_ema": ema_loss, "grad_norm": float(grad_norm),
                             "lr": opt.param_groups[0]["lr"]}, step=opt_step)
            if opt_step % cfg.i_save == 0 and opt_step < cfg.max_steps:
                save_adapter(acc, denoiser, out_dir, opt_step, push_to, push_path)
        if opt_step >= cfg.max_steps:
            break

    if acc.is_main_process:
        save_adapter(acc, denoiser, out_dir, cfg.max_steps, push_to, push_path)
        trackio.finish()


def main() -> None:
    ap = argparse.ArgumentParser(description="LoRA fine-tune a TRELLIS.2 geometry flow.")
    ap.add_argument("--config", required=True, help="LoRA hyperparameter JSON")
    ap.add_argument("--data_dir", required=True, help="dataset roots (JSON string) passed to the dataset")
    ap.add_argument("--output_dir", required=True, help="directory the adapters are written to")
    ap.add_argument("--holdout", default=None, help="file of sha256 (one per line) to exclude from training")
    ap.add_argument("--push_to", default=None, help="model repo id each checkpoint is uploaded to")
    ap.add_argument("--push_path", default="", help="folder inside --push_to for the checkpoints")
    ap.add_argument("--trackio_project", default="trellis2-geoflow-lora")
    ap.add_argument("--run_name", default=None, help="Trackio run name (default: output_dir basename)")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = Config(**json.load(f))
    train(cfg, args.data_dir, args.output_dir, args.holdout, args.push_to, args.push_path,
          args.trackio_project, args.run_name)


if __name__ == "__main__":
    main()
