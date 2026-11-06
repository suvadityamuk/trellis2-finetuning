# Fine-tuning TRELLIS.2 geometry with LoRA: code release

Code for LoRA fine-tuning the two geometry flows of
[TRELLIS.2](https://github.com/microsoft/TRELLIS.2) (the sparse-structure flow and the
shape flow) on your own 3D meshes. We used it to adapt TRELLIS.2 to 188 shoes from
Google Scanned Objects. On 9 held-out shoes, Chamfer distance fell from 0.174 to 0.033
and F-score@0.02 rose from 0.28 to 0.80.

```
release/
├── preprocess.py            one script: folder of meshes  -> training latents
├── finetune.py              one script: training latents  -> LoRA adapters
├── container/               everything inside the published Docker images
│   ├── Dockerfile           the full image (CUDA extensions, Blender, PEFT, Trackio)
│   ├── prep.Dockerfile      -prep image  (default command: trellis2-prep)
│   ├── train.Dockerfile     -train image (default command: trellis2-train)
│   ├── packages.txt         apt packages (headless GL / X for Blender)
│   ├── STARTER.md           README shipped inside the image
│   ├── patches/             our changes to upstream TRELLIS.2 files
│   └── trellis2/            our files, laid out at their paths in the TRELLIS.2 tree
└── LICENSE                  TRELLIS.2's MIT license (Microsoft)
```

## Quick start

You need an NVIDIA GPU with 24 GB or more (see the batch-size note below), Docker with the NVIDIA runtime, and a
Hugging Face token with access to
[DINOv3](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m), which is gated
and only needed for training. Model weights download on first use.

```bash
docker run --gpus all -it --rm --shm-size 16g -e HF_TOKEN=hf_xxx \
  -v hf-cache:/root/.cache/huggingface \
  -v "$PWD":/workspace/release -v "$PWD/work":/workspace/work \
  -w /workspace/release \
  cr.hf.co/suvadityamuk/trellis2-finetuning:latest
```

Inside the container:

```bash
# 1. meshes -> latents (any format Blender imports, any folder layout)
python preprocess.py --meshes /workspace/work/my_meshes --out /workspace/work/data

# 2. latents -> LoRA, one run per flow
python finetune.py --flow ss    --latents /workspace/work/data
python finetune.py --flow shape --latents /workspace/work/data
```

The `ss` and `shape` presets use a micro-batch of 4 with 4 accumulation steps, the
setting of our runs on an 80 GB A100. On a 24 GB GPU, add
`--set batch_size=2 grad_accum=8`: the effective batch is still 16, and that is the
setting we smoke-tested on an L4.

To skip step 1, train on our preprocessed shoes: pass
`--latents suvadityamuk/gso-footwear-trellis2-latents` and only the files training needs
are downloaded. For a smoke test, add `--set max_steps=5 i_save=5`.

### `preprocess.py`

This script runs the TRELLIS.2 `data_toolkit` stages in order: index, Blender mesh
dumps, O-Voxel dual grids (512 and 1024), shape latents (512 and 1024),
sparse-structure latents (64³), and 16 Cycles conditioning renders per object. It then
writes a seeded held-out split with ground-truth meshes and exports every mesh as a
`.glb`. Each stage skips objects it has already finished, so re-running the same
command resumes.

| flag | default | meaning |
|---|---|---|
| `--meshes` | required | local folder, or a Hub repo id holding the meshes |
| `--out` | `/workspace/work/prep/data` | output folder; this is what `finetune.py --latents` reads |
| `--push-to` | none | dataset repo that receives the output after every stage; if it already has data, the run resumes from it |
| `--max-objects` | all | only the first N meshes, for a dry run |
| `--holdout`, `--seed` | `0.05`, `0` | held-out fraction and split seed |
| `--views` | `16` | conditioning renders per object |
| `--workers` | `8` | parallel Blender processes; going above ~8 makes Blender startups crash |
| `--stages` | all | any subset of `index mesh voxel shape ss render split export` |

For scale: 12 objects take about 16 minutes on an L4, a third of it rendering.

### `finetune.py`

This script loads the pretrained flow and freezes it. It injects PEFT LoRA adapters
(the only trainable weights, kept in fp32) and trains them with the rectified-flow
velocity loss in bf16, under gradient checkpointing and accumulation. Objects in
`holdout.txt` are excluded from training.

| flag | default | meaning |
|---|---|---|
| `--flow ss\|shape` | | use our footwear hyperparameters for that flow (inlined at the top of the script) |
| `--config` | | or a JSON file, `hf://` path or URL; see `container/trellis2/configs/lora/` |
| `--set k=v ...` | none | override single fields, e.g. `max_steps=1000 i_save=100 lr=5e-5` |
| `--latents` | required | `preprocess.py` output: local folder or Hub dataset repo |
| `--output` | `/workspace/work/results` | adapters go to `<output>/<run-name>/adapter-<step>.safetensors` |
| `--push-to` | none | model repo that receives every checkpoint, the config and a model card |
| `--trackio-space` | `$TRACKIO_SPACE_ID` | Trackio Space for the dashboard; without it, logs go to a local dashboard. `--no-trackio` turns logging off |
| `--no-holdout` | off | also train on the held-out objects |

A checkpoint is written every `i_save` steps (about 37 MB of LoRA weights). Re-running
the same command resumes from the newest checkpoint, whether it's on disk or in
`--push-to`. With our presets, 3000 steps took 7.2 h for the sparse-structure flow and
3.6 h for the shape flow on one A100.

### Evaluate

`scripts/eval_geometry.py` generates a mesh from one render of each held-out object,
once with the base model and once with the LoRA, using the same seed. It reports
Chamfer distance, F-score and normal consistency against the ground truth. Run it from
the TRELLIS.2 root:

```bash
cd /workspace/TRELLIS.2
python scripts/eval_geometry.py \
  --test_list /workspace/work/data/eval_testlist.txt --mesh_dir /workspace/work/data/holdout_gt \
  --ss_adapter /workspace/work/results/ss_lora/adapter-3000.safetensors \
  --shape_adapter /workspace/work/results/shape_lora/adapter-3000.safetensors \
  --out_dir /workspace/work/eval --pipeline_type 512 --tau 0.02 --num_points 50000
```

## What's in the images

The TRELLIS.2 tree in every image is upstream
[`microsoft/TRELLIS.2@75fbf01`](https://github.com/microsoft/TRELLIS.2/tree/75fbf0183001ed9876c8dbb35de6b68552ee08bd),
with `container/patches/` applied and `container/trellis2/` copied on top. That
reproduces, file for file, the project's working TRELLIS.2 tree that the images were
built from.

**`patches/trellis2-75fbf01.patch`** changes six upstream files:

| file | change |
|---|---|
| `trellis2/modules/image_feature_extractor.py`, `trellis2/trainers/flow_matching/mixins/image_conditioned.py` | DINOv3 features on transformers 5.x, which nests the encoder blocks under `.model`. **Required**: without it the feature extractor crashes |
| `data_toolkit/blender_script/render_cond.py` | select the GPU device for Cycles correctly (OptiX, falling back to CUDA) |
| `data_toolkit/dump_mesh.py`, `data_toolkit/render_cond.py` | no runtime `sudo apt-get` (the image ships the libraries) |
| `data_toolkit/dump_mesh.py`, `data_toolkit/dual_grid.py` | default `--max_workers` of 8 instead of one per CPU core, which crashed Blender and oversubscribed the GPU |

**`trellis2/`** holds our files, at their paths inside `/workspace/TRELLIS.2`:

| path | role |
|---|---|
| `finetune_lora.py` | the Accelerate + PEFT trainer used for the footwear run (`finetune.py` is the same loop, plus data download and Hub handling) |
| `configs/lora/ss_flow_lora_gso.json`, `slat_flow_img2shape_lora_gso.json` | hyperparameters for the two flows |
| `data_toolkit/datasets/MeshFolder.py` | toolkit adapter for any folder of meshes |
| `data_toolkit/datasets/GSO.py` | toolkit adapter for Google Scanned Objects |
| `byod/prep.py`, `byod/train.py`, `byod/bin/` | the `trellis2-prep` and `trellis2-train` commands of the `-prep` and `-train` images |
| `scripts/eval_geometry.py` | held-out evaluation, base vs LoRA |
| `jobs/run_full.sh` | the HF Jobs entrypoint of the footwear run: materialize GSO, preprocess, split, train both flows, evaluate |
| `jobs/footwear_ids.txt` | the 188 GSO footwear object ids |
| `jobs/make_holdout.py` | group-aware held-out split (inlined in `preprocess.py`) |
| `jobs/log_eval.py`, `jobs/log_mesh_artifacts.py` | send eval results and meshes to Trackio |

The `-prep` and `-train` images run the same pipeline as the two one-script files,
as commands, with Hub input and output:

```bash
hf jobs run --flavor a100-large --timeout 6h --secrets HF_TOKEN \
  cr.hf.co/suvadityamuk/trellis2-finetuning-prep:latest \
  trellis2-prep --meshes you/my-meshes --push-to you/my-meshes-trellis2
hf jobs run --flavor a100-large --timeout 12h --secrets HF_TOKEN \
  cr.hf.co/suvadityamuk/trellis2-finetuning-train:latest \
  trellis2-train --config ss --latents you/my-meshes-trellis2 --push-to you/my-lora
```

### Rebuild the image

The build context is this `release/` folder. The image compiles CUDA extensions, so
build it on a linux/amd64 host with plenty of RAM; `MAX_JOBS=6` needs about 64 GB.

```bash
docker build -f container/Dockerfile -t trellis2-finetuning .
docker build -f container/prep.Dockerfile  --build-arg BASE=trellis2-finetuning -t trellis2-finetuning-prep .
docker build -f container/train.Dockerfile --build-arg BASE=trellis2-finetuning -t trellis2-finetuning-train .
```

By default, CUDA kernels are built for Ampere, Ada and Hopper
(`TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"`). Pass a narrower list with `--build-arg` to
build faster. To use the code without Docker, check out upstream at `75fbf01`, run
`git apply container/patches/trellis2-75fbf01.patch`, copy `container/trellis2/` over
the checkout, and point the scripts at it with `--trellis-root` (or `$TRELLIS2_ROOT`).

## Published artifacts

| | |
|---|---|
| [`cr.hf.co/suvadityamuk/trellis2-finetuning`](https://huggingface.co/containers/suvadityamuk/trellis2-finetuning) | the image (mirror: [`hfsuvadityamuk/trellis2-starter`](https://hub.docker.com/r/hfsuvadityamuk/trellis2-starter)) |
| [`-prep`](https://huggingface.co/containers/suvadityamuk/trellis2-finetuning-prep), [`-train`](https://huggingface.co/containers/suvadityamuk/trellis2-finetuning-train) | the image with `trellis2-prep` / `trellis2-train` as the default command |
| [`suvadityamuk/gso-footwear`](https://huggingface.co/datasets/suvadityamuk/gso-footwear) | the 188 raw footwear scans |
| [`suvadityamuk/gso-footwear-trellis2-latents`](https://huggingface.co/datasets/suvadityamuk/gso-footwear-trellis2-latents) | the same shoes, preprocessed (2 GB) |
| [`suvadityamuk/trellis-2-finetune-gso`](https://huggingface.co/suvadityamuk/trellis-2-finetune-gso) | our adapters (every 500 steps) and evaluation meshes |

## Credits and licenses

TRELLIS.2 is by Xiang et al. at Microsoft ([paper](https://arxiv.org/abs/2512.14692),
MIT; see `LICENSE`). Google Scanned Objects is by Google Research (CC-BY-4.0). DINOv3
is Meta's, under its own license; it is downloaded at run time and never
redistributed.
