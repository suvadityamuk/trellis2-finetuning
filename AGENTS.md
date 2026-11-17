# AGENTS.md

LoRA fine-tuning of TRELLIS.2's geometry flows (sparse structure + shape) on your own 3D
meshes. `README.md` is the full documentation; this file is the short map.

**To run the workflow** (preprocess meshes, train, generate meshes with the adapters), use
the agent skill in [`skills/trellis2-lora/`](skills/trellis2-lora/SKILL.md). It has the
commands, the guardrails (confirm GPU cost, dry-run first, private repos) and the known
failure modes.

## Ground rules

- Nothing here runs without an NVIDIA GPU (24 GB+) and the CUDA extensions in the
  `cr.hf.co/suvadityamuk/trellis2-finetuning` image. On a Mac or CPU-only machine, don't try
  to install TRELLIS.2; launch on HF Jobs (`hf jobs run ...`) instead.
- GPU jobs cost money. Confirm flavor and timeout with the user before launching one.
- The images copy `container/` at build time. Editing a file there doesn't change a
  published image until it is rebuilt (see "Rebuild the images" in the README).

## Where things are

| path | what | in which image |
|---|---|---|
| `preprocess.py`, `finetune.py` | single-file versions of prep and training; run in the base image with this repo mounted | none |
| `container/trellis2/byod/prep.py` | `trellis2-prep`: meshes → latents dataset | `-prep` |
| `container/trellis2/byod/train.py` | `trellis2-train`: config + latents → LoRA repo + Trackio | `-train` |
| `container/trellis2/finetune_lora.py` | the trainer that `trellis2-train` calls | `-train` |
| `container/trellis2/configs/lora/*.json` | LoRA presets (`ss`, `shape`) | all |
| `container/trellis2/scripts/eval_geometry.py` | held-out eval; `apply_adapter()` is the canonical adapter loader | all |
| `container/trellis2/data_toolkit/datasets/MeshFolder.py` | preprocessing adapter for any mesh folder | `-prep` |
| `container/Dockerfile`, `prep.Dockerfile`, `train.Dockerfile` | the three images | |
| `container/patches/trellis2-75fbf01.patch` | changes to upstream TRELLIS.2 at `75fbf01` | all |
| `skills/trellis2-lora/` | agent skill; `scripts/generate.py` turns an image into GLBs with the adapters | none (passed inline) |
| `.claude-plugin/marketplace.json` | lets Claude Code install the skill as a plugin | |

In the images, `container/trellis2/` is overlaid on upstream TRELLIS.2 at `/workspace/TRELLIS.2`.

## Invariants

- The shape flow's MLPs are `SparseLinear`; PEFT can't wrap them. The shape LoRA targets
  attention only. Don't add `mlp.*` to `slat_flow_img2shape_lora_gso.json`.
- Adapters are `get_peft_model_state_dict` dumps. Loading needs the same `rank`, `alpha` and
  `lora_targets` as training; `trellis2-train` pushes `config.json` next to each adapter.
- Adapters target the 512 pipeline (`pipeline_type="512"`).
- `--workers` above 8 in prep crashes Blender.
- The trainer drops incomplete batches, so tiny dry runs need `batch_size=1`.
