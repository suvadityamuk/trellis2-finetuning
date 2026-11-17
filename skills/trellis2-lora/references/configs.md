# trellis2-train configs

`--config ss` and `--config shape` load the presets used for the reference run (179 shoes,
3000 steps each). Start from them; override with `--set key=value` or copy and edit.

The presets are in the GitHub repo:

```bash
BASE=https://raw.githubusercontent.com/suvadityamuk/trellis2-finetuning/master/container/trellis2/configs/lora
curl -sLO $BASE/ss_flow_lora_gso.json
curl -sLO $BASE/slat_flow_img2shape_lora_gso.json
```

Upload an edited file with `hf upload <user>/my-trellis2-lora ss.json configs/ss.json`,
then pass `--config hf://<user>/my-trellis2-lora/configs/ss.json`
(`hf://datasets/<user>/<repo>/...` for a dataset repo).

## Preset values

| field | `ss` | `shape` | meaning |
|---|---|---|---|
| `ckpt` | `.../ss_flow_img_dit_1_3B_64_bf16` | `.../slat_flow_img2shape_dit_1_3B_512_bf16` | which flow is adapted; do not change |
| `sparse` | `false` | `true` | selects the flow; must match `ckpt` |
| `rank` / `alpha` | 8 / 16 | 16 / 32 | LoRA size |
| `lora_targets` | attention + `mlp.mlp.0`, `mlp.mlp.2` | attention only | the shape flow's MLPs are sparse layers PEFT cannot wrap; never add them |
| `lr` | 1e-4 | 1e-4 | AdamW |
| `weight_decay`, `grad_clip` | 0.01, 1.0 | 0.01, 1.0 | |
| `batch_size` × `grad_accum` | 4 × 4 | 4 × 4 | effective batch 16 |
| `max_steps`, `i_save`, `i_log` | 3000, 500, 10 | 3000, 500, 10 | |
| `p_uncond` | 0.1 | 0.1 | condition dropout for classifier-free guidance |
| `t_schedule` | logitNormal(1, 1) | uniform | timestep sampling |

Leave `dataset`, `dataset_args`, `image_cond_model` and `image_size` alone; they must
match what `trellis2-prep` produced.

## Tuning guidance

- **Small dataset:** fewer steps; compare intermediate checkpoints on held-out objects
  (`references/inference.md`) rather than assuming the last is best.
- **Larger dataset:** raise `max_steps` before raising `rank`.
- **Unstable loss / NaN alert:** `lr=5e-5`, keep `grad_clip=1.0`.
- **OOM:** halve `batch_size`, double `grad_accum` (effective batch unchanged).
- Rank changes must be made at training time; generation reads them from the pushed
  `config.json`, so they stay consistent automatically.
