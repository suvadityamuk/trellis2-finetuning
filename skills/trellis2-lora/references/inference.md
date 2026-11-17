# Generating meshes with the adapters

`scripts/generate.py` (in this skill) loads the base TRELLIS.2 pipeline, injects the `ss`
and/or `shape` adapters into the geometry flows, and writes textured GLBs. It must run in
the starter image, which has TRELLIS.2 and its CUDA extensions. The script is not baked into
the image, so pass it inline with `python -c "$(cat ...)"`.

## On HF Jobs

Put the input image(s) somewhere the job can reach: a URL, or a Hub repo
(`hf upload <user>/trellis2-inputs shoe.png --repo-type dataset`).

```bash
SKILL=<path to this skill directory>
hf jobs run -d --flavor l4x1 --timeout 1h --secrets HF_TOKEN \
  cr.hf.co/suvadityamuk/trellis2-finetuning:latest \
  python -c "$(cat $SKILL/scripts/generate.py)" \
    --image hf://datasets/<user>/trellis2-inputs/shoe.png \
    --ss-adapter hf://<user>/my-trellis2-lora/ss_lora/adapter-3000.safetensors \
    --shape-adapter hf://<user>/my-trellis2-lora/shape_lora/adapter-3000.safetensors \
    --compare --push-to <user>/trellis2-outputs --private
```

Results: `https://huggingface.co/datasets/<user>/trellis2-outputs/tree/main/generations`,
with `<name>_base.glb` and `<name>_lora.glb` per image.

To try the published shoe adapters, use
`hf://suvadityamuk/trellis-2-finetune-gso/footwear/{ss,shape}_lora/adapter-3000.safetensors`
(no config.json is next to these, so the script falls back to the matching presets, which
are what trained them).

## Flags

| flag | meaning |
|---|---|
| `--image` | one or more images: local path (in the container), `hf://...`, or URL. Background is removed automatically |
| `--ss-adapter`, `--shape-adapter` | either or both; omit both for base-only output |
| `--ss-config`, `--shape-config` | config that trained the adapter. Default: `config.json` beside a Hub adapter, else the preset |
| `--compare` | also generate with the base model, same seed |
| `--push-to`, `--private` | dataset repo for the GLBs. Needed on Jobs, which delete their disk |
| `--seed` | default 42 |
| `--pipeline-type` | keep `512`; the adapters were trained on 512 latents |
| `--texture-size` | default 2048 |

## Picking a checkpoint

Adapters exist every `i_save` steps. To compare them, run the job several times with
different `adapter-<step>` paths, or use `/workspace/TRELLIS.2/scripts/eval_geometry.py` in
the same image. It scores base vs LoRA on the held-out objects from `trellis2-prep`
(`eval_testlist.txt`, `holdout_gt/` in the latents repo) with Chamfer, F-score and normal
consistency.

## In Python (inside the starter image)

```python
import sys; sys.path[:0] = ["/workspace/TRELLIS.2", "/workspace/TRELLIS.2/scripts"]
from eval_geometry import apply_adapter, SS_KEY
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from PIL import Image

pipe = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B").cuda()
apply_adapter(pipe.models[SS_KEY], "ss.safetensors", rank=8, alpha=16,
              targets=["to_qkv", "to_out", "to_q", "to_kv", "mlp.mlp.0", "mlp.mlp.2"], merge=False)
apply_adapter(pipe.models["shape_slat_flow_model_512"], "shape.safetensors", rank=16, alpha=32,
              targets=["to_qkv", "to_out", "to_q", "to_kv"], merge=False)
mesh = pipe.run(Image.open("shoe.png"), seed=42, pipeline_type="512")[0]
```

Rank, alpha and targets must equal the training config's `rank`, `alpha` and `lora_targets`.
