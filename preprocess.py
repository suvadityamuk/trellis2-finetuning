"""Turn a folder of 3D meshes into TRELLIS.2 geometry-training latents, in one script.

    python preprocess.py --meshes ./my_meshes --out ./data
    python preprocess.py --meshes <user>/<mesh-dataset> --push-to <user>/<latents-dataset>

Every mesh file Blender can import (.glb .gltf .obj .fbx .ply .stl .usd .blend), in any
folder layout, becomes one object keyed by the SHA-256 of the file. The stages are the
TRELLIS.2 ``data_toolkit`` scripts, run in order:

    1 index    metadata.csv over every mesh file
    2 mesh     Blender 4.5 mesh dumps                     mesh_dumps/
    3 voxel    O-Voxel dual grids at 512 and 1024         dual_grid_{512,1024}/
    4 shape    shape latents at 512 and 1024              shape_latents/
    5 ss       sparse-structure latents (64^3, from 1024) ss_latents/
    6 render   Cycles conditioning views (Blender 3.0)    renders_cond/
    7 split    held-out split + ground-truth meshes       holdout.txt, holdout_gt/, eval_testlist.txt
    8 export   every mesh as a portable .glb              mesh_glb/

Each stage skips objects it already finished, so re-running resumes. With --push-to,
the output folder is uploaded after every stage, and an existing repo is downloaded
first, so a job that hits its timeout can simply be started again.

The output is what ``finetune.py --latents`` reads. Runs inside the
``cr.hf.co/suvadityamuk/trellis2-finetuning`` image (TRELLIS.2 at /workspace/TRELLIS.2,
both Blenders under /tmp, xvfb). Needs one NVIDIA GPU; DINOv3 access is not needed.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import random
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SHAPE_1024 = "shape_enc_next_dc_f16c32_fp16_1024"         # 1024 shape latents -> SS occupancy
STAGES = ["index", "mesh", "voxel", "shape", "ss", "render", "split", "export"]
HUB_IGNORE = [".cache/**", "**/.cache/**", "**/new_records/**"]


# ------------------------------------------------------------------- toolkit plumbing
def ensure_mesh_adapter(trellis: str) -> None:
    """The toolkit imports dataset adapters from data_toolkit/datasets/. Ours indexes any mesh folder.

    Installs (or refreshes) it from this release when the copy in the tree is missing or
    differs, e.g. on an older image or a plain TRELLIS.2 checkout.
    """
    dst = os.path.join(trellis, "data_toolkit", "datasets")
    src = os.path.join(HERE, "container", "trellis2", "data_toolkit", "datasets")
    if not os.path.isdir(src):                            # e.g. inside the image, without the release mounted
        if not os.path.exists(os.path.join(dst, "MeshFolder.py")):
            sys.exit(f"MeshFolder adapter missing from {dst} and not found at {src}")
        return
    os.makedirs(dst, exist_ok=True)
    for name in ("__init__.py", "MeshFolder.py"):
        target = os.path.join(dst, name)
        if not os.path.exists(target) or open(target, "rb").read() != open(os.path.join(src, name), "rb").read():
            shutil.copy(os.path.join(src, name), target)
            print(f"-- installed {name} into {dst}")


class Toolkit:
    """Runs one data_toolkit script with the arguments every stage shares."""

    def __init__(self, trellis: str, root: str, mesh_root: str, max_objects: int | None):
        self.trellis, self.root, self.mesh_root, self.max_objects = trellis, root, mesh_root, max_objects

    def __call__(self, script: str, *extra: str, per_mesh: bool = True, xvfb: bool = False) -> None:
        cmd = ["python", f"data_toolkit/{script}"]
        if per_mesh:                                      # scripts that walk the source meshes
            cmd += ["MeshFolder", "--mesh_root", self.mesh_root]
            if self.max_objects:
                cmd += ["--max_objects", str(self.max_objects)]
        cmd += ["--root", self.root, *extra]
        if xvfb:                                          # Blender 3.0 Cycles needs an X server
            cmd = ["xvfb-run", "-a", *cmd]
        print("+", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=self.trellis, check=True)

    def metadata(self) -> None:
        """Fold every stage's new records into metadata.csv (what the trainers read)."""
        self("build_metadata.py")


# ------------------------------------------------------------------ held-out split
def mesh_from_dump(pickle_path: str):
    """Merge the sub-meshes of one Blender dump into a single trimesh."""
    import trimesh
    objects = pickle.load(open(pickle_path, "rb"))["objects"]
    parts = [trimesh.Trimesh(o["vertices"], o["faces"], process=False)  # keep vertices as dumped
             for o in objects if len(o["faces"])]
    return trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]


def base_model(object_id: str) -> str:
    """Collapse an object id to its base model so colorway / size variants group together.

    Written for Google Scanned Objects names: drops a trailing `_in_<color>` /
    `_Color_<...>` / `_Size_<...>` clause and a final long hash or CamelCase colorway
    token. For other naming it leaves ids distinct, i.e. a plain seeded split.
    """
    s = re.split(r"_(?:in|Color|Size)_", object_id)[0]
    s = re.sub(r"_[A-Za-z0-9]{10,}$", "", s)
    return s.lower()


def make_holdout(root: str, frac: float, seed: int) -> None:
    """Seeded, group-aware split. Writes holdout.txt, holdout_gt/<sha>.glb, eval_testlist.txt.

    Only objects that made it through the whole pipeline (a mesh dump for ground truth
    and a render for the input image) are candidates, and near-duplicate variants of one
    model always land on the same side, so no held-out shape leaks in via a sibling.
    """
    import pandas as pd
    dumps, renders = os.path.join(root, "mesh_dumps"), os.path.join(root, "renders_cond")
    dumped = {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(dumps, "*.pickle"))}
    rendered = {d for d in os.listdir(renders) if os.path.isfile(os.path.join(renders, d, "transforms.json"))} \
        if os.path.isdir(renders) else set()
    eligible = dumped & rendered
    assert eligible, "no fully preprocessed objects (need a mesh dump AND a render) to hold out"

    meta = pd.read_csv(os.path.join(root, "metadata.csv")).set_index("sha256")
    groups: dict[str, list[str]] = {}
    for sha in eligible:
        obj_id = str(meta.loc[sha, "file_identifier"]) if sha in meta.index else sha
        groups.setdefault(base_model(obj_id), []).append(sha)

    keys = sorted(groups)
    random.Random(seed).shuffle(keys)                     # shuffle whole groups, take until ~frac
    target = max(1, round(len(eligible) * frac))
    holdout: list[str] = []
    for key in keys:
        if len(holdout) >= target:
            break
        holdout.extend(groups[key])
    holdout.sort()

    with open(os.path.join(root, "holdout.txt"), "w") as f:
        f.write("\n".join(holdout) + "\n")
    gt_dir = os.path.join(root, "holdout_gt")
    os.makedirs(gt_dir, exist_ok=True)
    lines = []
    for sha in holdout:
        mesh_from_dump(os.path.join(dumps, f"{sha}.pickle")).export(os.path.join(gt_dir, f"{sha}.glb"))
        frames = json.load(open(os.path.join(renders, sha, "transforms.json")))["frames"]
        lines.append(f"{sha} {os.path.join(renders, sha, frames[0]['file_path'])}")
    with open(os.path.join(root, "eval_testlist.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"held out {len(holdout)}/{len(eligible)} objects; training sees {len(eligible) - len(holdout)}")


def export_glb(root: str) -> None:
    out = os.path.join(root, "mesh_glb")
    os.makedirs(out, exist_ok=True)
    for path in sorted(glob.glob(os.path.join(root, "mesh_dumps", "*.pickle"))):
        dst = os.path.join(out, os.path.basename(path).replace(".pickle", ".glb"))
        if not os.path.exists(dst):
            mesh_from_dump(path).export(dst)


# ------------------------------------------------------------------------------- Hub
def fetch_meshes(spec: str, repo_type: str, include: list[str] | None, work: str) -> str:
    if os.path.isdir(spec):
        return os.path.abspath(spec)
    from huggingface_hub import snapshot_download
    dst = os.path.join(work, "meshes")
    snapshot_download(spec, repo_type=repo_type, local_dir=dst, allow_patterns=include)
    return dst


def restore(repo: str | None, root: str, private: bool) -> None:
    """Resume: download whatever an earlier (timed-out) run already pushed."""
    if not repo:
        return
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=private, exist_ok=True)
    if api.list_repo_files(repo, repo_type="dataset") != [".gitattributes"]:
        print(f"-- resuming from {repo}")
        snapshot_download(repo, repo_type="dataset", local_dir=root)


def push(repo: str | None, root: str, stage: str) -> None:
    if not repo:
        return
    from huggingface_hub import HfApi
    HfApi().upload_folder(folder_path=root, repo_id=repo, repo_type="dataset",
                          ignore_patterns=HUB_IGNORE, commit_message=f"preprocess: {stage}")
    print(f"-- pushed after '{stage}' -> https://huggingface.co/datasets/{repo}")


def write_card(root: str, source: str) -> None:
    import pandas as pd
    count = lambda pattern: len(glob.glob(os.path.join(root, pattern)))
    holdout = os.path.join(root, "holdout.txt")
    stats = {
        "meshes indexed": len(pd.read_csv(os.path.join(root, "metadata.csv"))),
        "sparse-structure latents": count("ss_latents/*/*.npz"),
        "shape latents (512)": count("shape_latents/*_512/*.npz"),
        "conditioning renders": count("renders_cond/*/transforms.json"),
        "held out": len(open(holdout).read().split()) if os.path.exists(holdout) else 0,
    }
    rows = "\n".join(f"| {k} | {v} |" for k, v in stats.items())
    with open(os.path.join(root, "README.md"), "w") as f:
        f.write(f"""---
tags:
- trellis2
- 3d
- latents
---

# TRELLIS.2 geometry-training data

Produced by `preprocess.py` from {source}. Train on it with `finetune.py --latents <this repo>`.

| | objects |
|---|---|
{rows}

| path | used by |
|---|---|
| `ss_latents/ss_enc_conv3d_16l8_fp16_64/` | sparse-structure flow |
| `shape_latents/shape_enc_next_dc_f16c32_fp16_512/` | shape flow |
| `renders_cond/<sha256>/` | both (DINOv3 conditioning images) |
| `metadata.csv`, `holdout.txt` | both (index, held-out split) |
| `holdout_gt/`, `eval_testlist.txt`, `mesh_glb/` | evaluation |
| `mesh_dumps/`, `dual_grid_*/`, `shape_latents/*_1024/` | intermediates (resume, re-encoding) |
""")


# ------------------------------------------------------------------------------- CLI
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meshes", required=True, help="local mesh directory, or a Hub repo id holding the meshes")
    ap.add_argument("--meshes-repo-type", default="dataset", choices=["dataset", "model"])
    ap.add_argument("--meshes-include", nargs="*", default=None,
                    help="only download Hub files matching these globs, e.g. 'chairs/**'")
    ap.add_argument("--out", default="/workspace/work/prep/data", help="output folder (the training dataset)")
    ap.add_argument("--push-to", default=None, help="dataset repo id that receives the output after every stage")
    ap.add_argument("--private", action="store_true", help="create --push-to as private")
    ap.add_argument("--max-objects", type=int, default=None, help="only the first N meshes (sorted by path)")
    ap.add_argument("--holdout", type=float, default=0.05, help="fraction held out for evaluation")
    ap.add_argument("--seed", type=int, default=0, help="seed of the held-out split")
    ap.add_argument("--views", type=int, default=16, help="conditioning renders per object")
    ap.add_argument("--workers", type=int, default=8,
                    help="parallel Blender processes (above ~8, Blender startups start crashing)")
    ap.add_argument("--stages", nargs="*", default=STAGES, choices=STAGES, help="run only these stages")
    ap.add_argument("--trellis-root", default=os.environ.get("TRELLIS2_ROOT", "/workspace/TRELLIS.2"),
                    help="TRELLIS.2 checkout with our patch applied (default $TRELLIS2_ROOT or /workspace/TRELLIS.2)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    # Unbounded OpenMP teams storm pthread_create on many-core hosts and segfault the encoders.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "8")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    root = os.path.abspath(args.out)
    os.makedirs(root, exist_ok=True)
    ensure_mesh_adapter(args.trellis_root)
    restore(args.push_to, root, args.private)
    mesh_root = fetch_meshes(args.meshes, args.meshes_repo_type, args.meshes_include, os.path.dirname(root))
    run = Toolkit(args.trellis_root, root, mesh_root, args.max_objects)
    workers = ["--max_workers", str(args.workers)]

    steps = {
        "index": run.metadata,
        "mesh": lambda: run("dump_mesh.py", *workers),
        "voxel": lambda: run("dual_grid.py", "--resolution", "512,1024", *workers),
        "shape": lambda: [run("encode_shape_latent.py", "--resolution", r, per_mesh=False) for r in ("512", "1024")],
        "ss": lambda: run("encode_ss_latent.py", "--shape_latent_name", SHAPE_1024, "--resolution", "64",
                          per_mesh=False),
        "render": lambda: run("render_cond.py", "--num_cond_views", str(args.views), *workers, xvfb=True),
        "split": lambda: make_holdout(root, args.holdout, args.seed),
        "export": lambda: export_glb(root),
    }
    selected = [s for s in STAGES if s in args.stages]
    for i, stage in enumerate(STAGES, 1):
        if stage not in selected:
            continue
        print(f"==================== {i}/{len(STAGES)}  {stage} ====================", flush=True)
        steps[stage]()
        if stage not in ("index", "split", "export"):     # toolkit stages write new records
            run.metadata()
        if stage == selected[-1]:
            write_card(root, args.meshes if not os.path.isdir(args.meshes) else "a local mesh folder")
        push(args.push_to, root, stage)

    stats = os.path.join(root, "statistics.txt")
    if os.path.exists(stats):
        print(open(stats).read())
    print("PREPROCESS DONE", f"https://huggingface.co/datasets/{args.push_to}" if args.push_to else root)


if __name__ == "__main__":
    main()
