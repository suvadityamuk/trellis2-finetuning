import os, sys; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
"""
Evaluate TRELLIS.2 image-to-3D *geometry* on a held-out GSO split, comparing the
BASE model against the LoRA-adapted geometry flows (sparse-structure + shape),
while DINOv3, both SC-VAEs and the material/PBR flow stay frozen/unchanged.

For every held-out object we take one conditioning render (the input image),
generate a mesh with (a) the base pipeline and (b) the LoRA pipeline under the
*same* seed, then compare each to the ground-truth GSO mesh after a consistent
normalization (center + unit scale -> [-0.5, 0.5] AABB, matching TRELLIS.2).

Metrics (trimesh + numpy + scipy KD-tree only):
  * Chamfer distance   (bidirectional, sampled surface points)
  * F-score@tau        (tau = fraction of the GT bbox diagonal, CLI arg)
  * Normal consistency (mean |cos| of matched surface normals, bidirectional)
Silhouette IoU is intentionally skipped (see note at the bottom of this file).

Example:
    python scripts/eval_geometry.py \
        --test_list held_out.txt --out_dir eval_out \
        --ss_adapter ckpts/ss/adapter-5000.safetensors \
        --shape_adapter ckpts/shape/adapter-5000.safetensors \
        --rank 16 --alpha 32 --tau 0.01 --num_points 100000
"""
import csv
import json
import argparse
import numpy as np
import torch
import trimesh
from PIL import Image
from scipy.spatial import cKDTree

from trellis2.pipelines import Trellis2ImageTo3DPipeline

# ---------------------------------------------------------------------------
# Pipeline model keys (verified against trellis2/pipelines/trellis2_image_to_3d.py)
#   - sparse-structure flow denoiser : models['sparse_structure_flow_model']
#   - shape (img2shape) flow denoiser: models['shape_slat_flow_model_{512,1024}']
#   - frozen: sparse_structure_decoder, shape_slat_decoder, tex_slat_* , DINOv3
# ---------------------------------------------------------------------------
SS_KEY = "sparse_structure_flow_model"
# LoRA targets must match the trainer per stage. The dense sparse-structure DiT adapts
# attention + MLP; the sparse shape DiT adapts attention only (its MLP is SparseLinear over
# SparseTensors, which PEFT's Linear LoRA cannot wrap). We read them from the config when given.
SS_TARGETS = ["to_qkv", "to_out", "to_q", "to_kv", "mlp.mlp.0", "mlp.mlp.2"]
SHAPE_TARGETS = ["to_qkv", "to_out", "to_q", "to_kv"]


def load_lora_spec(config_path: str, default_rank: int, default_alpha: int, default_targets: list):
    """Read (rank, alpha, target_modules) from a trainer LoRA config so eval matches training exactly."""
    if config_path and os.path.isfile(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        return cfg.get("rank", default_rank), cfg.get("alpha", default_alpha), cfg.get("lora_targets", default_targets)
    return default_rank, default_alpha, default_targets


def shape_keys(pipeline_type: str) -> list:
    """Shape-flow denoiser key(s) actually used for a given pipeline type."""
    if pipeline_type == "512":
        return ["shape_slat_flow_model_512"]
    if pipeline_type == "1024":
        return ["shape_slat_flow_model_1024"]
    # cascade variants run both the 512 (LR) and 1024 (HR) shape flows.
    return ["shape_slat_flow_model_512", "shape_slat_flow_model_1024"]


# ---------------------------------------------------------------------------
# LoRA application (must mirror the trainer's PEFT save format)
# ---------------------------------------------------------------------------
def apply_adapter(denoiser: torch.nn.Module, adapter_path: str, rank: int,
                  alpha: int, targets: list, merge: bool) -> None:
    """Inject a PEFT LoRA adapter into a denoiser and load its weights.

    Mirrors the trainer, which saved `peft.get_peft_model_state_dict(denoiser)`
    with LoraConfig(target_modules=<per-stage targets>, lora_dropout=0, bias="none").
    """
    from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict
    from safetensors.torch import load_file

    cfg = LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=0.0, bias="none",
        target_modules=targets,
    )
    inject_adapter_in_model(cfg, denoiser)
    state = load_file(adapter_path)
    result = set_peft_model_state_dict(denoiser, state)
    missing = getattr(result, "unexpected_keys", None)
    if missing:
        print(f"[lora] {adapter_path}: {len(missing)} unexpected keys")
    if merge:
        _merge_lora(denoiser)
    print(f"[lora] applied {adapter_path} (r={rank}, alpha={alpha}, merge={merge})")


def _merge_lora(denoiser: torch.nn.Module) -> None:
    """Fold LoRA deltas into the base weights for a plain forward pass."""
    if hasattr(denoiser, "merge_adapter"):
        try:
            denoiser.merge_adapter()
            return
        except Exception as e:  # noqa: BLE001
            print(f"[lora] merge_adapter() failed ({e}); merging per-layer")
    from peft.tuners.tuners_utils import BaseTunerLayer
    for module in denoiser.modules():
        if isinstance(module, BaseTunerLayer):
            module.merge()


# ---------------------------------------------------------------------------
# Mesh IO + normalization
# ---------------------------------------------------------------------------
def normalize(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Center AABB at origin and scale longest edge to 1 -> fits [-0.5, 0.5]^3."""
    center = mesh.bounds.mean(axis=0)
    mesh.apply_translation(-center)
    extent = float(mesh.extents.max())
    if extent > 0:
        mesh.apply_scale(1.0 / extent)
    return mesh


def load_gt_mesh(path: str) -> trimesh.Trimesh:
    mesh = trimesh.load(path, process=False, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh) or mesh.vertices.shape[0] == 0:
        raise ValueError(f"empty/invalid GT mesh: {path}")
    return normalize(mesh)


def to_trimesh(gen) -> trimesh.Trimesh:
    """Convert a generated MeshWithVoxel to a normalized trimesh (geometry only).

    `gen.vertices` (float [N,3]) and `gen.faces` (int [M,3]) are the geometry
    fields; the generated mesh already lives in the [-0.5, 0.5] AABB, but we
    re-normalize so both sides use an identical convention.
    """
    verts = gen.vertices.detach().cpu().numpy().astype(np.float64)
    faces = gen.faces.detach().cpu().numpy().astype(np.int64)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    return normalize(mesh)


# ---------------------------------------------------------------------------
# Geometry metrics
# ---------------------------------------------------------------------------
def sample_surface(mesh: trimesh.Trimesh, n: int):
    """Return (points [n,3], normals [n,3]) sampled uniformly on the surface."""
    points, face_idx = trimesh.sample.sample_surface(mesh, n)
    normals = mesh.face_normals[face_idx]
    return np.asarray(points), np.asarray(normals)


def compute_metrics(gen: trimesh.Trimesh, gt: trimesh.Trimesh,
                    tau_frac: float, num_points: int) -> dict:
    gp, gn = sample_surface(gen, num_points)
    tp, tn = sample_surface(gt, num_points)

    gt_tree = cKDTree(tp)
    gen_tree = cKDTree(gp)
    d_g2t, i_g2t = gt_tree.query(gp)   # gen -> gt
    d_t2g, i_t2g = gen_tree.query(tp)  # gt  -> gen

    diag = float(np.linalg.norm(gt.extents))
    tau = tau_frac * diag

    chamfer = float(d_g2t.mean() + d_t2g.mean())
    precision = float((d_g2t < tau).mean())
    recall = float((d_t2g < tau).mean())
    fscore = 0.0 if (precision + recall) == 0 else \
        2 * precision * recall / (precision + recall)

    nc_g = np.abs(np.sum(gn * tn[i_g2t], axis=1))
    nc_t = np.abs(np.sum(tn * gn[i_t2g], axis=1))
    normal_consistency = float(0.5 * (nc_g.mean() + nc_t.mean()))

    return {
        "chamfer": chamfer,
        "fscore": fscore,
        "precision": precision,
        "recall": recall,
        "normal_consistency": normal_consistency,
        "tau_abs": tau,
    }


# ---------------------------------------------------------------------------
# Test list parsing
# ---------------------------------------------------------------------------
def parse_test_list(path: str, mesh_dir: str):
    """Each line: `<mesh_path_or_id>[ ,|whitespace]<conditioning_image>`."""
    items = []
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in (line.split(",") if "," in line else line.split())]
            if len(parts) < 2:
                raise ValueError(f"need '<mesh> <image>' per line, got: {raw!r}")
            mesh_ref, image_path = parts[0], parts[1]
            mesh_path = resolve_mesh(mesh_ref, mesh_dir)
            obj_id = os.path.splitext(os.path.basename(mesh_ref))[0]
            items.append((obj_id, mesh_path, image_path))
    return items


def resolve_mesh(ref: str, mesh_dir: str) -> str:
    if os.path.isfile(ref):
        return ref
    if mesh_dir:
        for ext in (".glb", ".obj"):
            cand = os.path.join(mesh_dir, ref + ext)
            if os.path.isfile(cand):
                return cand
    raise FileNotFoundError(f"cannot resolve mesh '{ref}' (mesh_dir={mesh_dir!r})")


# ---------------------------------------------------------------------------
# Generation + best-effort visualization
# ---------------------------------------------------------------------------
def generate(pipeline, image_path: str, args):
    image = Image.open(image_path)
    out = pipeline.run(
        image, num_samples=1, seed=args.seed,
        preprocess_image=not args.no_preprocess,
        pipeline_type=args.pipeline_type,
    )
    return out[0]


def save_glb(gen, path: str) -> None:
    """Save the generated geometry (vertices+faces) as GLB for offline viewing."""
    verts = gen.vertices.detach().cpu().numpy()
    faces = gen.faces.detach().cpu().numpy()
    trimesh.Trimesh(vertices=verts, faces=faces, process=False).export(path)


def try_render_pair(base_gen, lora_gen, path: str) -> bool:
    """Best-effort side-by-side snapshot PNG (needs the CUDA renderer)."""
    try:
        from trellis2.utils import render_utils
        rows = []
        for gen in (base_gen, lora_gen):
            frames = render_utils.render_snapshot([gen], resolution=384, nviews=1)
            rows.append(frames["color"][0])
        Image.fromarray(np.concatenate(rows, axis=1)).save(path)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[vis] renderer unavailable ({e}); saved GLB instead")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--test_list", required=True,
                   help="file: '<mesh_path_or_id> <cond_image>' per line")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--ss_adapter", default=None, help="sparse-structure LoRA .safetensors")
    p.add_argument("--shape_adapter", default=None, help="shape-flow LoRA .safetensors")
    p.add_argument("--ss_config", default="configs/lora/ss_flow_lora_gso.json",
                   help="trainer LoRA config for the ss adapter (source of rank/alpha/targets)")
    p.add_argument("--shape_config", default="configs/lora/slat_flow_img2shape_lora_gso.json",
                   help="trainer LoRA config for the shape adapter (source of rank/alpha/targets)")
    p.add_argument("--rank", type=int, default=16, help="fallback LoRA rank when no config is given")
    p.add_argument("--alpha", type=int, default=None, help="fallback LoRA alpha (=rank if unset)")
    p.add_argument("--merge", action="store_true", help="merge LoRA into base weights")
    p.add_argument("--tau", type=float, default=0.01, help="F-score tau as fraction of GT bbox diagonal")
    p.add_argument("--num_points", type=int, default=100000)
    p.add_argument("--pipeline_type", default="512",
                   choices=["512", "1024", "1024_cascade", "1536_cascade"])
    p.add_argument("--model", default="microsoft/TRELLIS.2-4B")
    p.add_argument("--mesh_dir", default=None, help="resolve bare object ids to {id}.glb/.obj here")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_preprocess", action="store_true", help="skip rembg/crop on input images")
    p.add_argument("--max_objects", type=int, default=0, help="cap objects (0 = all)")
    p.add_argument("--num_vis", type=int, default=3, help="objects to save GLBs / renders for")
    return p.parse_args()


def evaluate_split(pipeline, items, tag, args):
    """Run the pipeline over all objects, returning per-object metric rows."""
    rows = []
    vis_dir = os.path.join(args.out_dir, "vis")
    for idx, (obj_id, mesh_path, image_path) in enumerate(items):
        row = {"object_id": obj_id, "model": tag, "cond_image": image_path,
               "gt_source": mesh_path}
        try:
            gt = load_gt_mesh(mesh_path)
            gen = generate(pipeline, image_path, args)
            gen_tm = to_trimesh(gen)
            row.update(compute_metrics(gen_tm, gt, args.tau, args.num_points))
            if idx < args.num_vis:
                os.makedirs(vis_dir, exist_ok=True)
                save_glb(gen, os.path.join(vis_dir, f"{obj_id}_{tag}.glb"))
                row["_gen"] = gen  # kept transiently for optional paired render
            print(f"[{tag}] {obj_id}: chamfer={row['chamfer']:.5f} "
                  f"f{args.tau}={row['fscore']:.4f} nc={row['normal_consistency']:.4f}")
        except Exception as e:  # noqa: BLE001
            print(f"[{tag}] {obj_id}: FAILED ({e})")
            for k in ("chamfer", "fscore", "precision", "recall",
                      "normal_consistency", "tau_abs"):
                row[k] = float("nan")
        rows.append(row)
    return rows


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def mean_row(rows, tag):
    metric_keys = ("chamfer", "fscore", "precision", "recall", "normal_consistency", "tau_abs")
    out = {"object_id": "__mean__", "model": tag}
    for k in metric_keys:
        vals = [r[k] for r in rows if k in r and not np.isnan(r[k])]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out


def main():
    args = parse_args()
    assert args.ss_adapter or args.shape_adapter, \
        "provide at least one of --ss_adapter / --shape_adapter for the LoRA pass"
    os.makedirs(args.out_dir, exist_ok=True)

    items = parse_test_list(args.test_list, args.mesh_dir)
    if args.max_objects > 0:
        items = items[:args.max_objects]
    print(f"[eval] {len(items)} held-out objects | pipeline_type={args.pipeline_type}")

    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.model)
    if torch.cuda.is_available():
        pipeline.cuda()
    else:
        print("[eval] WARNING: CUDA not available; generation will likely fail")

    # Pass 1: BASE (no adapters).
    base_rows = evaluate_split(pipeline, items, "base", args)

    # Inject LoRA into the geometry flows only; everything else stays frozen.
    if args.ss_adapter:
        r, a, tgts = load_lora_spec(args.ss_config, args.rank, args.alpha or args.rank, SS_TARGETS)
        apply_adapter(pipeline.models[SS_KEY], args.ss_adapter, r, a, tgts, args.merge)
    if args.shape_adapter:
        r, a, tgts = load_lora_spec(args.shape_config, args.rank, args.alpha or args.rank, SHAPE_TARGETS)
        for key in shape_keys(args.pipeline_type):
            apply_adapter(pipeline.models[key], args.shape_adapter, r, a, tgts, args.merge)

    # Pass 2: LoRA.
    lora_rows = evaluate_split(pipeline, items, "lora", args)

    # Optional paired base-vs-LoRA snapshot PNGs (best effort).
    vis_dir = os.path.join(args.out_dir, "vis")
    base_by_id = {r["object_id"]: r for r in base_rows}
    for lr in lora_rows[:args.num_vis]:
        br = base_by_id.get(lr["object_id"])
        if br and "_gen" in br and "_gen" in lr:
            try_render_pair(br["_gen"], lr["_gen"],
                            os.path.join(vis_dir, f"{lr['object_id']}_pair.png"))

    for r in base_rows + lora_rows:
        r.pop("_gen", None)

    all_rows = base_rows + [mean_row(base_rows, "base")] + \
        lora_rows + [mean_row(lora_rows, "lora")]
    fields = ["object_id", "model", "chamfer", "fscore", "precision", "recall",
              "normal_consistency", "tau_abs", "cond_image", "gt_source"]
    csv_path = os.path.join(args.out_dir, "metrics.csv")
    write_csv(csv_path, all_rows, fields)

    bm, lm = mean_row(base_rows, "base"), mean_row(lora_rows, "lora")
    print("\n=== MEAN (base -> lora) ===")
    print(f"  chamfer            : {bm['chamfer']:.5f} -> {lm['chamfer']:.5f}")
    print(f"  f-score@{args.tau}   : {bm['fscore']:.4f} -> {lm['fscore']:.4f}")
    print(f"  normal_consistency : {bm['normal_consistency']:.4f} -> {lm['normal_consistency']:.4f}")
    print(f"[eval] wrote {csv_path}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()

# ---------------------------------------------------------------------------
# NOTE: Silhouette IoU is skipped. It requires rendering both the generated
# MeshWithVoxel and the GT trimesh from identical cameras; the TRELLIS.2
# renderer needs CUDA (nvdiffrast/flex_gemm), an EnvMap, and operates on
# trellis2 representations rather than trimesh, so aligning both sides is not
# "easy" with the existing utils. We instead save generated meshes as GLB and
# attempt best-effort side-by-side snapshot PNGs when the renderer is present.
# ---------------------------------------------------------------------------
