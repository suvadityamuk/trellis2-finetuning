"""Log the pilot's intermediate mesh geometry to Trackio.

Trackio has no 3D-object media type (it renders Image / Video / Audio / Table, but
not a rotatable mesh), so we can't preview a mesh inline the way W&B's Object3D does.
Instead we do the sensible fallback — store the meshes as blobs, but keep them tied
to the experiment:

  1. convert each Blender mesh dump ({vertices, faces}) into a portable ``.glb``
     (openable in any 3D viewer and by the Hub's built-in model3d preview), and
  2. attach the whole set as a versioned Trackio *artifact* — content-addressed
     blobs that sync to the run's Space bucket with lineage — alongside a small
     *table* of thumbnails so the run still shows something visual.

Runs after preprocessing, against the same project/Space as the training runs:

    python jobs/log_mesh_artifacts.py --root /workspace/data/gso
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle

import pandas as pd
import trackio
import trimesh


def dump_to_glb(pickle_path: str, glb_path: str) -> tuple[int, int]:
    """Merge the sub-meshes in one Blender dump into a single .glb; return (n_vertices, n_faces)."""
    objects = pickle.load(open(pickle_path, "rb"))["objects"]
    parts = [trimesh.Trimesh(o["vertices"], o["faces"], process=False)  # process=False: keep verts as dumped
             for o in objects if len(o["faces"])]
    mesh = trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]
    mesh.export(glb_path)                                              # .glb inferred from the extension
    return len(mesh.vertices), len(mesh.faces)


def preview_image(root: str, sha256: str) -> trackio.Image | None:
    """First conditioning render of an object, wrapped as a Trackio image for the thumbnail table."""
    views = sorted(glob.glob(os.path.join(root, "renders_cond", sha256, "*.png")))
    return trackio.Image(value=views[0], caption=sha256[:12]) if views else None


def main() -> None:
    ap = argparse.ArgumentParser(description="Log intermediate meshes to Trackio (as .glb artifacts + a preview table).")
    ap.add_argument("--root", required=True, help="pilot data root holding mesh_dumps/ and renders_cond/")
    ap.add_argument("--project", default="trellis2-geoflow-lora")
    ap.add_argument("--name", default="geometry-preprocessing")
    args = ap.parse_args()

    dumps = sorted(glob.glob(os.path.join(args.root, "mesh_dumps", "*.pickle")))
    if not dumps:
        print("no mesh dumps under", args.root, "- nothing to log")
        return
    glb_dir = os.path.join(args.root, "mesh_glb")
    os.makedirs(glb_dir, exist_ok=True)

    trackio.init(project=args.project, name=args.name,
                 space_id=os.environ.get("TRACKIO_SPACE_ID"),
                 private=os.environ.get("TRACKIO_PRIVATE", "0") == "1")

    rows = []
    for path in dumps:
        sha256 = os.path.splitext(os.path.basename(path))[0]
        n_vertices, n_faces = dump_to_glb(path, os.path.join(glb_dir, f"{sha256}.glb"))
        rows.append({"object": sha256[:12], "n_vertices": n_vertices,
                     "n_faces": n_faces, "preview": preview_image(args.root, sha256)})
    trackio.log({"meshes": trackio.Table(dataframe=pd.DataFrame(rows))})   # visual index in the dashboard

    # The meshes themselves ride along as a downloadable artifact (Trackio can't render 3D inline):
    # both the portable .glb set and the exact raw dumps, versioned with lineage to this run.
    artifact = trackio.Artifact(
        name="gso-meshes", type="dataset",
        description="Intermediate GSO meshes: normalized Blender dumps exported as portable .glb.",
        metadata={"count": len(dumps), "producer": "data_toolkit/dump_mesh.py"})
    artifact.add_dir(glb_dir)
    artifact.add_dir(os.path.join(args.root, "mesh_dumps"))
    trackio.log_artifact(artifact)

    trackio.finish()
    print(f"logged {len(dumps)} meshes -> artifact 'gso-meshes' (+ preview table)")


if __name__ == "__main__":
    main()
