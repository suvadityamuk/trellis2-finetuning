"""trellis2-prep: turn any folder of 3D meshes into TRELLIS.2 geometry-training data.

    trellis2-prep --meshes <user>/<mesh-dataset> --push-to <user>/<latents-dataset>

Stages (each skips objects it already finished, so re-running resumes):

    1 index      metadata.csv over every mesh file
    2 mesh       Blender 4.5 mesh dumps           mesh_dumps/
    3 voxel      O-Voxel dual grids at 512, 1024  dual_grid_{512,1024}/
    4 shape      shape latents at 512, 1024       shape_latents/
    5 ss         sparse-structure latents (64^3)  ss_latents/
    6 render     Cycles conditioning views        renders_cond/
    7 split      held-out split + GT meshes       holdout.txt, holdout_gt/, eval_testlist.txt
    8 export     every mesh as a portable .glb    mesh_glb/

After every stage the whole work tree is uploaded to the --push-to dataset repo, and at
start an existing repo is downloaded first, so a job that times out can simply be re-run.
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import subprocess
import sys

import pandas as pd
import trimesh
from huggingface_hub import HfApi, snapshot_download

TRELLIS = "/workspace/TRELLIS.2"
SHAPE_1024 = "shape_enc_next_dc_f16c32_fp16_1024"
STAGES = ["index", "mesh", "voxel", "shape", "ss", "render", "split", "export"]
IGNORE = [".cache/**", "**/.cache/**", "**/new_records/**"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="trellis2-prep", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meshes", required=True,
                    help="Hub repo id holding the meshes (e.g. user/my-meshes), or a local directory")
    ap.add_argument("--meshes-repo-type", default="dataset", choices=["dataset", "model"])
    ap.add_argument("--meshes-include", nargs="*", default=None,
                    help="only download files matching these globs (e.g. 'chairs/**')")
    ap.add_argument("--push-to", default=None,
                    help="dataset repo id that receives every stage's output (omit to keep it local)")
    ap.add_argument("--private", action="store_true", help="create --push-to as a private repo")
    ap.add_argument("--work", default="/workspace/work/prep", help="local working directory")
    ap.add_argument("--max-objects", type=int, default=None, help="process only the first N meshes")
    ap.add_argument("--holdout", type=float, default=0.05, help="fraction held out for evaluation")
    ap.add_argument("--seed", type=int, default=0, help="seed of the held-out split")
    ap.add_argument("--views", type=int, default=16, help="conditioning renders per object")
    ap.add_argument("--workers", type=int, default=8,
                    help="parallel Blender processes (more than ~8 makes Blender startups crash)")
    ap.add_argument("--stages", nargs="*", default=STAGES, choices=STAGES, help="subset of stages to run")
    return ap.parse_args()


class Prep:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.root = os.path.join(args.work, "data")
        self.api = HfApi()
        os.makedirs(self.root, exist_ok=True)

    # ------------------------------------------------------------------ hub
    def restore(self) -> None:
        repo = self.args.push_to
        if not repo:
            return
        self.api.create_repo(repo, repo_type="dataset", private=self.args.private, exist_ok=True)
        if self.api.list_repo_files(repo, repo_type="dataset") != [".gitattributes"]:
            print(f"-- resuming from {repo}")
            snapshot_download(repo, repo_type="dataset", local_dir=self.root)

    def push(self, stage: str) -> None:
        if self.args.push_to:
            self.api.upload_folder(folder_path=self.root, repo_id=self.args.push_to, repo_type="dataset",
                                   ignore_patterns=IGNORE, commit_message=f"trellis2-prep: {stage}")
            print(f"-- pushed after '{stage}' -> https://huggingface.co/datasets/{self.args.push_to}")

    # --------------------------------------------------------------- meshes
    def fetch_meshes(self) -> str:
        if os.path.isdir(self.args.meshes):
            return os.path.abspath(self.args.meshes)
        dst = os.path.join(self.args.work, "meshes")
        snapshot_download(self.args.meshes, repo_type=self.args.meshes_repo_type, local_dir=dst,
                          allow_patterns=self.args.meshes_include)
        return dst

    # -------------------------------------------------------------- toolkit
    def toolkit(self, script: str, *extra: str, subset: bool = True, xvfb: bool = False) -> None:
        cmd = ["python", f"data_toolkit/{script}"]
        if subset:
            cmd += ["MeshFolder", "--mesh_root", self.mesh_root]
            if self.args.max_objects:
                cmd += ["--max_objects", str(self.args.max_objects)]
        cmd += ["--root", self.root, *extra]
        if xvfb:
            cmd = ["xvfb-run", "-a", *cmd]
        print("+", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=TRELLIS, check=True)

    def metadata(self) -> None:
        self.toolkit("build_metadata.py")

    def export_glb(self) -> None:
        out = os.path.join(self.root, "mesh_glb")
        os.makedirs(out, exist_ok=True)
        for path in sorted(glob.glob(os.path.join(self.root, "mesh_dumps", "*.pickle"))):
            dst = os.path.join(out, os.path.basename(path).replace(".pickle", ".glb"))
            if os.path.exists(dst):
                continue
            parts = [trimesh.Trimesh(o["vertices"], o["faces"], process=False)
                     for o in pickle.load(open(path, "rb"))["objects"] if len(o["faces"])]
            trimesh.util.concatenate(parts).export(dst)

    def write_card(self) -> None:
        meta = pd.read_csv(os.path.join(self.root, "metadata.csv"))
        holdout_file = os.path.join(self.root, "holdout.txt")
        held = len(open(holdout_file).read().split()) if os.path.exists(holdout_file) else 0
        count = lambda d: len(glob.glob(os.path.join(self.root, d)))
        stats = {
            "meshes indexed": len(meta),
            "sparse-structure latents": count("ss_latents/*/*.npz"),
            "shape latents (512)": count("shape_latents/*_512/*.npz"),
            "conditioning renders": count("renders_cond/*/transforms.json"),
            "held out": held,
        }
        rows = "\n".join(f"| {k} | {v} |" for k, v in stats.items())
        source = self.args.meshes if not os.path.isdir(self.args.meshes) else "a local mesh folder"
        with open(os.path.join(self.root, "README.md"), "w") as f:
            f.write(f"""---
tags:
- trellis2
- 3d
- latents
---

# TRELLIS.2 geometry-training data

Produced by `trellis2-prep` from {source}. Train on it with `trellis2-train --latents <this repo>`.

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

    # ------------------------------------------------------------------ run
    def run(self) -> None:
        a = self.args
        self.restore()
        self.mesh_root = self.fetch_meshes()
        steps = {
            "index": lambda: self.metadata(),
            "mesh": lambda: self.toolkit("dump_mesh.py", "--max_workers", str(a.workers)),
            "voxel": lambda: self.toolkit("dual_grid.py", "--resolution", "512,1024", "--max_workers", str(a.workers)),
            "shape": lambda: [self.toolkit("encode_shape_latent.py", "--resolution", r, subset=False)
                              for r in ("512", "1024")],
            "ss": lambda: self.toolkit("encode_ss_latent.py", "--shape_latent_name", SHAPE_1024,
                                       "--resolution", "64", subset=False),
            "render": lambda: self.toolkit("render_cond.py", "--num_cond_views", str(a.views),
                                           "--max_workers", str(a.workers), xvfb=True),
            "split": lambda: subprocess.run(
                ["python", "jobs/make_holdout.py", "--root", self.root, "--frac", str(a.holdout),
                 "--seed", str(a.seed)], cwd=TRELLIS, check=True),
            "export": self.export_glb,
        }
        for i, stage in enumerate(STAGES, 1):
            if stage not in a.stages:
                continue
            print(f"==================== {i}/{len(STAGES)}  {stage} ====================", flush=True)
            steps[stage]()
            if stage not in ("index", "split", "export"):
                self.metadata()
            if stage == STAGES[-1] or stage == a.stages[-1]:
                self.write_card()
            self.push(stage)
        print(open(os.path.join(self.root, "statistics.txt")).read())
        print("PREP DONE")


def main() -> None:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "8")   # unbounded OpenMP teams crash many-core Jobs hosts
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    Prep(parse_args()).run()


if __name__ == "__main__":
    sys.exit(main())
