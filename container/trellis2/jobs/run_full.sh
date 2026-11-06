#!/usr/bin/env bash
# =============================================================================
# TRELLIS.2 geometry-LoRA full run — one self-contained HF Job.
#
# Same shape as the pilot, but trained for real and measured on a held-out split:
#   materialize a GSO subset -> geometry preprocessing -> carve a 5% held-out
#   eval set -> LoRA-train the sparse-structure flow then the shape flow for the
#   full step budget (holdout excluded) -> push adapters -> generate on the
#   held-out images and score Chamfer / F-score vs ground truth (base vs LoRA).
#
# The Jobs filesystem is ephemeral. To survive a crash / timeout we mirror every
# expensive intermediate to CACHE_REPO after each stage and restore it at start,
# so a re-run resumes instead of recomputing. Adapters + eval go to OUTPUT_REPO.
#
# Required env (see launch_full.sh):
#   HF_TOKEN            gated model + Hub access (passed via --secrets)
#   GSO_DATASET_REPO    HF dataset repo id of the GSO WebDataset (tar shards)
#   OUTPUT_REPO         HF model repo id to receive adapters + eval outputs
# Optional env:
#   CACHE_REPO          HF dataset repo id for resumable intermediate artifacts
#   OBJECT_LIST_FILE    allow-list of object ids to train on exclusively
#                       (e.g. jobs/footwear_ids.txt); overrides NUM_OBJECTS
#   NUM_OBJECTS=200  HOLDOUT_FRAC=0.05  TRACKIO_SPACE_ID=<user>/trackio
# =============================================================================
set -euo pipefail

: "${GSO_DATASET_REPO:?set GSO_DATASET_REPO to the HF dataset repo id of the GSO WebDataset}"
: "${OUTPUT_REPO:?set OUTPUT_REPO to a HF model repo id for the adapters}"
export NUM_OBJECTS="${NUM_OBJECTS:-200}"
export HOLDOUT_FRAC="${HOLDOUT_FRAC:-0.05}"
# If set, materialize EXACTLY the object ids in this file (an allow-list, e.g.
# jobs/footwear_ids.txt) and ignore NUM_OBJECTS — this is how the footwear-only
# run is scoped. Unset -> take the first NUM_OBJECTS objects in shard order.
export OBJECT_LIST_FILE="${OBJECT_LIST_FILE:-}"
# Per-object toolkit stages fan out with a thread pool. Left uncapped they default
# to os.cpu_count() workers, which over-subscribes the box: each mesh/render worker
# spawns a `blender -b` process, and ~50 concurrent Blender startups crash 30-60% of
# the time (measured: 48 workers -> 37/50 dumped; 8 workers -> 50/50). Cap it.
export WORKERS="${WORKERS:-8}"
# Reduce allocator fragmentation for the 1.3B flows on a single 80GB A100.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Cap OpenMP/BLAS teams. The o_voxel/numpy stages (esp. encode_shape_latent at 1024)
# fan out with thread pools; unbounded, libgomp sizes each team to the host's full
# core count, and on a big many-core Jobs host that storms pthread_create until it
# fails ("libgomp: Thread creation failed: Resource temporarily unavailable" -> segfault).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
export ROOT="${ROOT:-/workspace/data/gso}"
export GSO_SRC="${GSO_SRC:-$ROOT/_src}"          # materialized meshes (not cached; cheap to rebuild)
export CACHE_REPO="${CACHE_REPO:-}"
export RUN_TAG="${RUN_TAG:-full}"                # subfolder in OUTPUT_REPO (e.g. "footwear")

# Encoder output-dir names are derived by the toolkit as "<ckpt>_<resolution>".
export SHAPE_1024="shape_enc_next_dc_f16c32_fp16_1024"   # feeds SS occupancy
export SHAPE_512="shape_enc_next_dc_f16c32_fp16_512"      # shape-flow training input
export SS_NAME="ss_enc_conv3d_16l8_fp16_64"               # ss-flow training input

cd /workspace/TRELLIS.2
mkdir -p "$ROOT" "$GSO_SRC"

# --------------------------------------------------------------------------- cache
# CACHE_REPO holds a mirror of $ROOT minus $ROOT/_src. Push after every stage so a
# crashed job resumes: each toolkit stage skips objects whose outputs already exist.
cache_pull () {
  [ -z "$CACHE_REPO" ] && return 0
  echo "-- restoring cache from $CACHE_REPO"
  python - <<'PY'
import os
from huggingface_hub import HfApi, snapshot_download
repo = os.environ["CACHE_REPO"]
HfApi().create_repo(repo, repo_type="dataset", exist_ok=True, private=True)
try:
    snapshot_download(repo, repo_type="dataset", local_dir=os.environ["ROOT"])
    print("cache restored")
except Exception as e:
    print("nothing to restore yet:", e)
PY
}
cache_push () {  # $1 = stage label
  [ -z "$CACHE_REPO" ] && return 0
  echo "-- caching after: $1"
  STAGE="$1" python - <<'PY'
import os
from huggingface_hub import HfApi
HfApi().upload_folder(
    folder_path=os.environ["ROOT"], repo_id=os.environ["CACHE_REPO"], repo_type="dataset",
    ignore_patterns=["_src/*", "**/.cache/*"], commit_message=f"cache: {os.environ['STAGE']}")
print("cached:", os.environ["STAGE"])
PY
}

echo "==================== 0/6  restore cache ===================="
cache_pull

echo "==================== 1/6  materialize GSO objects ===================="
# Pull tar shards and write the 3 files Blender needs per object:
#   <GSO_SRC>/<id>/meshes/{model.obj, model.mtl, texture.png}
# Two modes: an explicit allow-list (OBJECT_LIST_FILE, e.g. footwear) scanned
# across ALL shards, or the first NUM_OBJECTS in shard order. Idempotent: objects
# already on disk (from a prior shard) are skipped.
python - <<'PY'
import os, tarfile
from huggingface_hub import HfApi, hf_hub_download

repo, dst = os.environ["GSO_DATASET_REPO"], os.environ["GSO_SRC"]
list_file = os.environ.get("OBJECT_LIST_FILE", "").strip()
want = None
if list_file:
    want = {ln.strip() for ln in open(list_file) if ln.strip() and not ln.startswith("#")}
    print(f"allow-list mode: {len(want)} target ids from {list_file} (NUM_OBJECTS ignored)")
else:
    n = int(os.environ["NUM_OBJECTS"])
    print(f"first-N mode: up to {n} objects in shard order")

FIELDS = {"obj": "model.obj", "mtl": "model.mtl", "texture.png": "texture.png"}
shards = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset") if f.endswith(".tar"))
assert shards, f"no .tar shards found in dataset repo {repo}"

have = set(os.listdir(dst)) if os.path.isdir(dst) else set()
for shard in shards:
    if want is not None and want.issubset(have):   # every allow-listed object found -> stop early
        break
    if want is None and len(have) >= n:
        break
    path = hf_hub_download(repo, shard, repo_type="dataset", local_dir="/tmp/gso_shards")
    with tarfile.open(path) as t:
        by_obj = {}
        for m in (m for m in t.getmembers() if m.isfile()):
            key, _, field = os.path.basename(m.name).partition(".")  # id.texture.png -> (id, texture.png)
            by_obj.setdefault(key, {})[field] = m
        for key, fields in by_obj.items():
            if key in have or not all(f in fields for f in FIELDS):
                continue
            if want is not None and key not in want:   # allow-list: skip everything else
                continue
            if want is None and len(have) >= n:
                break
            meshes = os.path.join(dst, key, "meshes")
            os.makedirs(meshes, exist_ok=True)
            for field, out in FIELDS.items():
                with open(os.path.join(meshes, out), "wb") as fh:
                    fh.write(t.extractfile(fields[field]).read())
            have.add(key)
    os.remove(path)  # each shard is ~0.5 GB; free it once mined
if want is not None:
    missing = sorted(want - have)
    print(f"materialized {len(have & want)}/{len(want)} allow-listed objects -> {dst}")
    if missing:
        print(f"WARNING: {len(missing)} allow-listed ids not found (first few): {missing[:5]}")
else:
    print(f"materialized {len(have)} objects -> {dst}")
PY

echo "==================== 2/6  geometry preprocessing ===================="
bm () { python data_toolkit/build_metadata.py GSO --root "$ROOT" --gso_root "$GSO_SRC"; }
# metadata.csv (indexes _src, sets absolute local_path to each model.obj)
bm && cache_push metadata
# mesh pickles via Blender 4.5.1 (headless, no rendering -> no X server needed)
python data_toolkit/dump_mesh.py GSO --root "$ROOT" --gso_root "$GSO_SRC" --max_workers "$WORKERS"
bm && cache_push mesh
# O-Voxel dual grids at both resolutions the two encoders consume
python data_toolkit/dual_grid.py GSO --root "$ROOT" --gso_root "$GSO_SRC" --resolution 512,1024 --max_workers "$WORKERS"
bm && cache_push dual_grid
# shape latents: 512 (shape flow) and 1024 (source of SS occupancy)
python data_toolkit/encode_shape_latent.py --root "$ROOT" --resolution 512
python data_toolkit/encode_shape_latent.py --root "$ROOT" --resolution 1024
bm && cache_push shape_latent
# sparse-structure latents at 64^3, derived from the 1024 shape-latent coords
python data_toolkit/encode_ss_latent.py --root "$ROOT" --shape_latent_name "$SHAPE_1024" --resolution 64
bm && cache_push ss_latent
# 16-view Cycles conditioning renders — the only stage that needs an X server
xvfb-run -a python data_toolkit/render_cond.py GSO --root "$ROOT" --gso_root "$GSO_SRC" --num_cond_views 16 --max_workers "$WORKERS"
bm && cache_push renders

# Archive the intermediate meshes to Trackio (portable .glb artifact + thumbnail table).
# Non-fatal: a tracking hiccup must never sink the training run.
python jobs/log_mesh_artifacts.py --root "$ROOT" || echo "-- mesh-artifact logging failed (non-fatal)"

echo "==================== 3/6  carve $HOLDOUT_FRAC held-out eval split ===================="
# Deterministic (seeded) so the split is reproducible and the trainer can exclude it.
python jobs/make_holdout.py --root "$ROOT" --frac "$HOLDOUT_FRAC"
cache_push holdout

echo "==================== 4/6  train ss-flow then shape-flow (holdout excluded) ===================="
# Both runs (ss_lora, shape_lora) sync to this Space if TRACKIO_SPACE_ID is set.
[ -n "${TRACKIO_SPACE_ID:-}" ] && echo "Trackio dashboard: https://huggingface.co/spaces/$TRACKIO_SPACE_ID"
# Each dataset root is a JSON object; every sub-path must carry a metadata.csv
# and the union of their columns must supply *_encoded / *_tokens (latent dir),
# aesthetic_score (meta = dataset root), and cond_rendered (renders_cond dir).
SS_ROOTS=$(python - <<'PY'
import json, os
r = os.environ["ROOT"]
print(json.dumps({"gso": {"ss_latent": f"{r}/ss_latents/{os.environ['SS_NAME']}",
                          "render_cond": f"{r}/renders_cond", "meta": r}}))
PY
)
SHAPE_ROOTS=$(python - <<'PY'
import json, os
r = os.environ["ROOT"]
print(json.dumps({"gso": {"shape_latent": f"{r}/shape_latents/{os.environ['SHAPE_512']}",
                          "render_cond": f"{r}/renders_cond", "meta": r}}))
PY
)

# Adapters live UNDER $ROOT so they ride cache_push/cache_pull for free — the key to
# surviving the 12h cap. Each flow is ~9.7h (ss) / ~5.8h (shape) on one A100, so both
# rarely fit one job. We therefore push each flow the instant it finishes (to CACHE_REPO
# *and* OUTPUT_REPO) and skip any flow whose final checkpoint is already on disk. A
# timeout mid-shape keeps the finished ss adapter; just re-run to resume the shape flow.
export ADP="$ROOT/adapters"
mkdir -p "$ADP"

push_adapters () {  # $1 = flow name (ss_lora | shape_lora); publish this flow to OUTPUT_REPO
  SUB="$1" python - <<'PY'
import os
from huggingface_hub import HfApi
api, repo, sub = HfApi(), os.environ["OUTPUT_REPO"], os.environ["SUB"]
api.create_repo(repo, repo_type="model", exist_ok=True)
api.upload_folder(folder_path=os.path.join(os.environ["ADP"], sub),
                  repo_id=repo, path_in_repo=f"{os.environ['RUN_TAG']}/{sub}")
print("pushed", sub, "->", repo, "/", os.environ["RUN_TAG"])
PY
}

train_flow () {  # $1 = flow name, $2 = config json, $3 = data roots json
  name="$1"; cfg="$2"; roots="$3"
  max=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['max_steps'])" "$cfg")
  if [ -f "$ADP/$name/adapter-$max.safetensors" ]; then
    echo "-- $name already complete (adapter-$max restored from cache); skipping training"
  else
    accelerate launch --num_processes 1 --mixed_precision bf16 finetune_lora.py \
        --config "$cfg" --data_dir "$roots" \
        --output_dir "$ADP/$name" --holdout "$ROOT/holdout.txt"
    cache_push "train_$name"     # adapters are under $ROOT -> mirrored to CACHE_REPO now
  fi
  push_adapters "$name"          # ...and published to OUTPUT_REPO, so a later timeout can't lose it
}

train_flow ss_lora    configs/lora/ss_flow_lora_gso.json            "$SS_ROOTS"
train_flow shape_lora configs/lora/slat_flow_img2shape_lora_gso.json "$SHAPE_ROOTS"

echo "==================== 5/6  adapters already pushed to $OUTPUT_REPO per-flow ===================="
echo "-- ss_lora + shape_lora live under $OUTPUT_REPO/$RUN_TAG (pushed as each flow finished)"

echo "==================== 6/6  held-out eval (base vs LoRA) ===================="
# Adapters are already safe on the Hub, so eval is non-fatal: a generation hiccup
# must not discard a finished training run. Picks the last checkpoint of each flow.
SS_ADP=$(ls -t "$ADP"/ss_lora/adapter-*.safetensors | head -1)
SHAPE_ADP=$(ls -t "$ADP"/shape_lora/adapter-*.safetensors | head -1)
if python scripts/eval_geometry.py \
      --test_list "$ROOT/eval_testlist.txt" --mesh_dir "$ROOT/holdout_gt" --out_dir results/eval \
      --ss_adapter "$SS_ADP" --shape_adapter "$SHAPE_ADP" \
      --ss_config configs/lora/ss_flow_lora_gso.json \
      --shape_config configs/lora/slat_flow_img2shape_lora_gso.json \
      --pipeline_type 512 --tau 0.02 --num_points 50000 --num_vis 5; then
  python jobs/log_eval.py --eval_dir results/eval || echo "-- eval-logging failed (non-fatal)"
  python - <<'PY'
import os
from huggingface_hub import HfApi
HfApi().upload_folder(folder_path="results/eval", repo_id=os.environ["OUTPUT_REPO"],
                      path_in_repo=os.environ["RUN_TAG"] + "/eval")
print("pushed eval ->", os.environ["OUTPUT_REPO"])
PY
else
  echo "-- held-out eval failed (non-fatal); adapters are already pushed"
fi

echo "FULL RUN COMPLETE"
