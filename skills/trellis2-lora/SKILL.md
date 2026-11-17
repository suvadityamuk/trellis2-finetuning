---
name: trellis2-lora
description: Fine-tune the geometry of TRELLIS.2 (Microsoft's image-to-3D model) with LoRA on a user's own 3D meshes, and generate meshes from images with the resulting adapters. Runs on Hugging Face Jobs (or local Docker with an NVIDIA GPU) using the prebuilt `trellis2-prep` and `trellis2-train` images; data, checkpoints and metrics all live on the Hub, with Trackio dashboards. Use when the user mentions TRELLIS.2 or TRELLIS, image-to-3D fine-tuning, training a 3D generation model on a category of objects (shoes, chairs, furniture, products), LoRA for 3D geometry, preprocessing meshes into TRELLIS.2 latents, or generating a GLB from an image with a fine-tuned TRELLIS.2 adapter.
---

# Fine-tune TRELLIS.2 geometry with LoRA

TRELLIS.2-4B turns one image into a textured mesh. This workflow trains LoRA adapters on
its two **geometry** flows (sparse structure `ss`, and `shape`) from a folder of 3D meshes.
Textures always come from the base model. Source and full docs:
https://github.com/suvadityamuk/trellis2-finetuning (trained shoe adapters:
https://huggingface.co/suvadityamuk/trellis-2-finetune-gso).

```
meshes (Hub dataset) → trellis2-prep → latents (Hub dataset) → trellis2-train --config ss    → LoRA (Hub model)
                                                              → trellis2-train --config shape ↗   + Trackio Space
```

| image | command |
|---|---|
| `cr.hf.co/suvadityamuk/trellis2-finetuning-prep:latest` | `trellis2-prep` |
| `cr.hf.co/suvadityamuk/trellis2-finetuning-train:latest` | `trellis2-train` |
| `cr.hf.co/suvadityamuk/trellis2-finetuning:latest` | starter image: TRELLIS.2 at `/workspace/TRELLIS.2`, all CUDA extensions, PEFT, Trackio. Use it for inference |

Nothing here runs without a CUDA GPU (24 GB+). On a laptop, always launch on HF Jobs.

## Rules for agents

1. **Confirm cost before every GPU job.** Tell the user the flavor and `--timeout` you
   plan to use and get a yes. Do not quote prices from memory; run `hf jobs hardware`.
2. **Dry-run first** on any new dataset, into **separate repos** (both commands resume from
   whatever their output repo already holds, so a dry run must not share one with the real
   run): prep with `--max-objects 8 --push-to <user>/my-meshes-trellis2-dryrun` on `l4x1`,
   then train from that repo with `--set max_steps=20 i_save=10 batch_size=1` and
   `--push-to <user>/my-trellis2-lora-dryrun` on `l4x1`. Scale up only after both succeed.
3. **Private by default.** Pass `--private` on prep and train, and `--trackio-private`
   (Trackio Spaces are public otherwise). Drop them only if the user asks for public.
4. **Always `--push-to`.** Job containers are deleted when they end; anything not pushed is lost.
5. Launch with `-d` and give the user the job URL. Never block a turn on `hf jobs logs -f`;
   check with `hf jobs inspect <id>` or a bounded `hf jobs logs <id> | tail -50`.

## Prerequisites (check before launching)

- `hf auth whoami` works and the token has **write** access.
- HF Jobs needs a Pro, Team or Enterprise account.
- **Training only:** the token's account must have accepted the gated
  [DINOv3 license](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m).
  Verify: `hf download facebook/dinov3-vitl16-pretrain-lvd1689m config.json` must succeed.
  If not, send the user to that page; the training job fails without it.
- Inputs are **3D meshes**, not images: `.glb .gltf .obj(+.mtl/textures) .fbx .ply .stl .usd`,
  any folder layout, **one object per file**. Merge multi-part scenes into one file first.
- For scale: the reference run used 179 shoes of one category.

## Step 0: meshes on the Hub

```bash
hf repos create <user>/my-meshes --repo-type dataset --private
hf upload <user>/my-meshes ./my_meshes . --repo-type dataset
```

If the meshes already live in a model repo, add `--meshes-repo-type model` in step 1.

## Step 1: preprocess (one GPU job)

```bash
hf jobs run -d --flavor a100-large --timeout 6h --secrets HF_TOKEN \
  cr.hf.co/suvadityamuk/trellis2-finetuning-prep:latest \
  trellis2-prep --meshes <user>/my-meshes --push-to <user>/my-meshes-trellis2 --private \
    --holdout 0.05 --views 16 --workers 8
```

- Dry run: `--flavor l4x1 --timeout 1h`, `--max-objects 8`, and a `-dryrun` `--push-to` repo.
- `--meshes-include 'chairs/**'` limits which files are used.
- **Never set `--workers` above 8**: Blender startups crash.
- Eight stages (`index mesh voxel shape ss render split export`). The repo is updated after
  each one, and re-running the identical command resumes. Scale: 12 objects ≈ 16 min on L4.
- Success check: the output repo has `ss_latents/`, `shape_latents/`, `renders_cond/`,
  `metadata.csv`, `holdout.txt`, and a README with per-stage counts. Compare those counts to
  the number of input meshes; objects that failed a stage are dropped, not fatal.

## Step 2: train (two independent jobs; launch both in parallel)

```bash
for FLOW in ss shape; do
hf jobs run -d --flavor a100-large --timeout 12h --secrets HF_TOKEN \
  cr.hf.co/suvadityamuk/trellis2-finetuning-train:latest \
  trellis2-train --config $FLOW \
    --latents <user>/my-meshes-trellis2 \
    --push-to <user>/my-trellis2-lora --private \
    --trackio-space <user>/trellis2-trackio --trackio-private \
    --set max_steps=3000 i_save=500
done
```

- Dry run: `--flavor l4x1 --timeout 1h`, `--latents` and `--push-to` pointing at the
  `-dryrun` repos, and `--set max_steps=20 i_save=10 batch_size=1`. The trainer drops
  incomplete batches, so a tiny dataset with `batch_size=4` can yield zero batches.
- `--config` takes `ss`, `shape`, a local JSON, `hf://<repo>/<path>`, or a URL. Use
  `--set key=value` (JSON-parsed) for one-off changes. See [references/configs.md](references/configs.md)
  for fields and tuning advice.
- Every `i_save` steps, `<run-name>/adapter-<step>.safetensors` (~37 MB) plus `config.json`
  lands in `--push-to`. Re-running the same command resumes from the newest checkpoint.
- Held-out objects are excluded automatically; `--no-holdout` trains on everything.
- `--subset <dataset repo | dir | file>` trains on part of `--latents` (a `subset.txt` of sha256s, optional own
  `holdout.txt`). GSO subsets exported from the GSO subset builder Space already have both.
- On a 24 GB GPU (`l4x1`, A10, 3090/4090) add `batch_size=2 grad_accum=8` to `--set`
  (same effective batch of 16; smoke-tested on L4).
- Reference speed on one A100, 3000 steps: 7.2 h for `ss`, 3.6 h for `shape`. If
  `--timeout` hits, just re-run; it resumes.

## Monitoring

```bash
hf jobs ps                      # running jobs
hf jobs inspect <id>            # status
hf jobs logs <id> | tail -50    # recent output
hf jobs cancel <id>
```

Live loss, grad norm and LR are at `https://huggingface.co/spaces/<user>/trellis2-trackio`.
A non-finite loss raises a Trackio alert; stop the job and lower `lr`.

## Step 3: generate meshes with the adapters

See [references/inference.md](references/inference.md). It uses `scripts/generate.py`, which
loads both adapters into the base pipeline and writes base-vs-LoRA GLBs to a Hub repo.

## Common failures

| symptom | fix |
|---|---|
| 401/403 on `facebook/dinov3-...` in a train job | accept the DINOv3 license with the token's account |
| prep crashes in `render` with many workers | `--workers 8` or fewer |
| an object becomes many tiny objects | it was split across files; merge to one file per object |
| near-duplicate variants land in both train and holdout | name them `<model>_in_<variant>` (or `_Color_`, `_Size_`), or dedupe first |
| CUDA OOM in training on a 24 GB GPU | `--set batch_size=2 grad_accum=8`, or use `a100-large` |
| job killed at timeout | re-run the identical command; both commands resume |

## Running locally instead of HF Jobs

Same commands, different launcher (NVIDIA GPU, Docker with the NVIDIA runtime):

```bash
docker run --gpus all --rm --shm-size 16g -e HF_TOKEN \
  -v hf-cache:/root/.cache/huggingface -v "$PWD/work":/workspace/work \
  cr.hf.co/suvadityamuk/trellis2-finetuning-prep:latest \
  trellis2-prep --meshes <user>/my-meshes --push-to <user>/my-meshes-trellis2 --private
```

For local meshes add `-v "$PWD/my_meshes":/meshes:ro` and pass `--meshes /meshes`.

The GitHub repo also has `preprocess.py` and `finetune.py`, single-file equivalents of the
two commands that run in the starter image with the repo mounted. Prefer the images unless
the user wants to modify the training code.

## Limits

Geometry only; textures come from the base model. Adapters are trained for the 512
pipeline. The holdout split's de-duplication rules were written for GSO-style names.
