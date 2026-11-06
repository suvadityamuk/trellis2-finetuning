"""Carve a deterministic, group-aware held-out evaluation split.

The trainer must never see the objects we measure generalization on, so we fix a
small fraction (default 5%) up front with a seeded shuffle. Two extra guarantees:

  * eligibility: an object is a candidate only if it made it through the *whole*
    geometry pipeline — it has both a Blender mesh dump (our ground truth) and a
    conditioning render (the eval input image);
  * group-aware: GSO ships many *colorway siblings* of one physical model (e.g.
    `ASICS_GELLinksmaster_WhiteCoffeeSand` vs `..._WhiteRasberryGunmetal`, or
    `Womens_Bluefish_2Eye_Boat_Shoe_in_Linen_Oat` vs `..._in_Tan`) that share
    essentially the same geometry. We group by base model and keep every sibling
    on the SAME side of the split, so a held-out shape never leaks in via a
    same-mesh colorway that was trained on.

Writes under ``--root``:
  * ``holdout.txt``          - one sha256 per line, for ``finetune_lora --holdout``
  * ``holdout_gt/<sha>.glb`` - portable ground-truth meshes (from the Blender dumps)
  * ``eval_testlist.txt``    - ``<sha> <cond_image>`` per line, for ``eval_geometry``
                                (resolved against ``holdout_gt`` via its ``--mesh_dir``)

    python jobs/make_holdout.py --root /workspace/data/gso --frac 0.05
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import random
import re

import pandas as pd
import trimesh


def dump_to_glb(pickle_path: str, glb_path: str) -> None:
    """Merge the sub-meshes of one Blender dump into a single portable .glb."""
    objects = pickle.load(open(pickle_path, "rb"))["objects"]
    parts = [trimesh.Trimesh(o["vertices"], o["faces"], process=False)  # process=False: keep verts as dumped
             for o in objects if len(o["faces"])]
    mesh = trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]
    mesh.export(glb_path)  # .glb inferred from the extension


def first_cond_view(renders_root: str, sha256: str) -> str:
    """Path to an object's first conditioning render (the eval input image)."""
    frames = json.load(open(os.path.join(renders_root, sha256, "transforms.json")))["frames"]
    return os.path.join(renders_root, sha256, frames[0]["file_path"])


def base_model(object_id: str) -> str:
    """Collapse a GSO id to its base model so colorway/size siblings group together.

    Drops a trailing `_in_<color>` / `_Color_<...>` / `_Size_<...>` clause, an
    11-ish-char random hash suffix, and a final CamelCase colorway token
    (e.g. `WhiteCoffeeSand`). Heuristic, but it catches the obvious same-mesh
    variants; distinct models keep distinct keys.
    """
    s = re.split(r"_(?:in|Color|Size)_", object_id)[0]
    s = re.sub(r"_[A-Za-z0-9]{10,}$", "", s)   # trailing hash or long colorway word
    return s.lower()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="preprocessing root (holds metadata.csv, mesh_dumps/, renders_cond/)")
    ap.add_argument("--frac", type=float, default=0.05, help="fraction of objects to hold out")
    ap.add_argument("--seed", type=int, default=0, help="split seed (fixed so the split is reproducible)")
    args = ap.parse_args()

    dumps_dir = os.path.join(args.root, "mesh_dumps")
    renders_root = os.path.join(args.root, "renders_cond")

    dumped = {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(dumps_dir, "*.pickle"))}
    rendered = {d for d in os.listdir(renders_root)
                if os.path.isfile(os.path.join(renders_root, d, "transforms.json"))} if os.path.isdir(renders_root) else set()
    eligible = dumped & rendered  # both a GT mesh and an input image
    assert eligible, "no fully-preprocessed objects (need a mesh dump AND a render) to hold out"

    # sha256 -> object id, so we can group colorway siblings by base model.
    meta = pd.read_csv(os.path.join(args.root, "metadata.csv")).set_index("sha256")
    groups: dict[str, list[str]] = {}
    for sha in eligible:
        obj_id = str(meta.loc[sha, "file_identifier"]) if sha in meta.index else sha
        groups.setdefault(base_model(obj_id), []).append(sha)

    # Shuffle whole groups, then take groups until we've held out ~frac of objects.
    keys = sorted(groups)
    random.Random(args.seed).shuffle(keys)
    target = max(1, round(len(eligible) * args.frac))
    holdout: list[str] = []
    for key in keys:
        if len(holdout) >= target:
            break
        holdout.extend(groups[key])
    holdout = sorted(holdout)

    with open(os.path.join(args.root, "holdout.txt"), "w") as f:
        f.write("\n".join(holdout) + "\n")

    gt_dir = os.path.join(args.root, "holdout_gt")
    os.makedirs(gt_dir, exist_ok=True)
    lines = []
    for sha256 in holdout:
        dump_to_glb(os.path.join(dumps_dir, f"{sha256}.pickle"), os.path.join(gt_dir, f"{sha256}.glb"))
        lines.append(f"{sha256} {first_cond_view(renders_root, sha256)}")
    with open(os.path.join(args.root, "eval_testlist.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"held out {len(holdout)}/{len(eligible)} objects across "
          f"{sum(1 for k in keys if any(s in holdout for s in groups[k]))} model groups "
          f"({args.frac:.0%} target); train sees {len(eligible) - len(holdout)}")


if __name__ == "__main__":
    main()
