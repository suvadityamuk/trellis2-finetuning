# TRELLIS.2 starter image: skip the setup, write the fine-tuning

Getting [TRELLIS.2](https://huggingface.co/microsoft/TRELLIS.2-4B) to *run* is most of
the work of fine-tuning it: CUDA extensions (flash-attn, FlexGEMM, CuMesh, nvdiffrast,
o-voxel), two headless Blenders, and a multi-stage preprocessing pipeline. This image
has all of it done. Pair it with the preprocessed dataset below and the only thing
left to write is your training code.

## What's inside

| | |
|---|---|
| Base | `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel` (torch 2.6, CUDA 12.4, `nvcc`) |
| CUDA extensions | flash-attn 2.7.3, FlexGEMM, CuMesh, nvdiffrast, nvdiffrec, o-voxel |
| Fine-tuning stack | PEFT, Accelerate, Trackio, safetensors, transformers 5.18 (DINOv3) |
| Blender | 3.0.1 and 4.5.1 at `/tmp/blender-*-linux-x64/blender`, plus `xvfb` |
| Code | TRELLIS.2 at `/workspace/TRELLIS.2` (the working directory), including a reference LoRA trainer, `finetune_lora.py` |
| GPUs | Ampere (A100, A10, A40, RTX 30xx), Ada (L4, L40S, RTX 40xx), Hopper (H100) |

Model weights are **not** baked in. TRELLIS.2-4B (MIT) downloads on first use into
`/root/.cache/huggingface`; mount a volume there to keep it between runs. DINOv3, the
image encoder, is gated under Meta's license: request access at
[facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)
and pass an `HF_TOKEN` from that account.

## Quickstart

```bash
docker run --gpus all -it -e HF_TOKEN=hf_xxx \
  -v hf-cache:/root/.cache/huggingface cr.hf.co/suvadityamuk/trellis2-finetuning:latest

# inside the container: pull the preprocessed footwear training data (~1.9 GB)
hf download suvadityamuk/gso-footwear-trellis2-latents --repo-type dataset --local-dir /workspace/data/gso
```

That dataset holds, for 188 Google Scanned Objects shoes, everything the two geometry
flows train on: sparse-structure latents (64³), shape latents (512), 16 conditioning
renders per object, a held-out split, and ground-truth meshes. No Blender needed.

## Train with the reference script

```bash
DATA=/workspace/data/gso
# sparse-structure flow
accelerate launch --num_processes 1 --mixed_precision bf16 finetune_lora.py \
  --config configs/lora/ss_flow_lora_gso.json \
  --data_dir "{\"gso\": {\"ss_latent\": \"$DATA/ss_latents/ss_enc_conv3d_16l8_fp16_64\", \"render_cond\": \"$DATA/renders_cond\", \"meta\": \"$DATA\"}}" \
  --output_dir results/ss_lora --holdout $DATA/holdout.txt
# shape flow
accelerate launch --num_processes 1 --mixed_precision bf16 finetune_lora.py \
  --config configs/lora/slat_flow_img2shape_lora_gso.json \
  --data_dir "{\"gso\": {\"shape_latent\": \"$DATA/shape_latents/shape_enc_next_dc_f16c32_fp16_512\", \"render_cond\": \"$DATA/renders_cond\", \"meta\": \"$DATA\"}}" \
  --output_dir results/shape_lora --holdout $DATA/holdout.txt
```

The configs target an 80 GB GPU (batch 4 × grad-accum 4). On a 24 GB card, lower
`batch_size` and raise `grad_accum` in the JSON. Set `TRACKIO_SPACE_ID=<user>/<space>`
to log to a hosted Trackio dashboard.

## Or write your own loop

`finetune_lora.py` exposes the building blocks, so a custom loop is short. Save it
next to the trainer (e.g. `/workspace/TRELLIS.2/my_loop.py`) and `python my_loop.py`:

```python
import json, torch
from finetune_lora import Config, build_denoiser, build_loader, flow_loss
from trellis2.trainers.flow_matching.mixins.image_conditioned import DinoV3FeatureExtractor
from trellis2.utils.data_utils import cycle, recursive_to_device

cfg = Config(**json.load(open("configs/lora/ss_flow_lora_gso.json")))
cfg.batch_size = 2
data = json.dumps({"gso": {"ss_latent": "/workspace/data/gso/ss_latents/ss_enc_conv3d_16l8_fp16_64",
                           "render_cond": "/workspace/data/gso/renders_cond",
                           "meta": "/workspace/data/gso"}})

denoiser = build_denoiser(cfg).cuda()        # frozen TRELLIS.2 flow + trainable LoRA
dino = DinoV3FeatureExtractor(cfg.image_cond_model, image_size=cfg.image_size)
dino.cuda()
opt = torch.optim.AdamW([p for p in denoiser.parameters() if p.requires_grad], lr=cfg.lr)

for step, batch in zip(range(100), cycle(build_loader(cfg, data))):
    x0 = recursive_to_device(batch["x_0"], "cuda")
    cond = dino(batch["cond"])               # DINOv3 patch tokens (no grad)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = flow_loss(denoiser, x0, cond, cfg)
    loss.backward()
    opt.step()
    opt.zero_grad()
    print(step, loss.item())
```

Swap in your own objective, adapter, or data and keep everything else.

## Bringing your own objects

The preprocessing toolkit is in `/workspace/TRELLIS.2/data_toolkit` and its Blender
paths are preconfigured. `jobs/run_full.sh` runs the whole chain (mesh dump, dual
grids, latent encoding, conditioning renders); the rendering step needs
`xvfb-run -a`.

## Licenses

TRELLIS.2 code and weights: MIT. DINOv3: Meta's DINOv3 license (gated, not included).
Blender: GPL. Google Scanned Objects data: CC-BY-4.0.
