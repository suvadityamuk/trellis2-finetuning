"""Generate a 3D mesh from one image with TRELLIS.2, optionally with geometry LoRA adapters.

Runs inside the starter image (cr.hf.co/suvadityamuk/trellis2-finetuning or its -prep/-train
variants), which has TRELLIS.2 at /workspace/TRELLIS.2. The LoRA loading is imported from
scripts/eval_geometry.py there, so it always matches how the adapters were trained.

    python generate.py --image shoe.png \
        --ss-adapter hf://suvadityamuk/trellis-2-finetune-gso/footwear/ss_lora/adapter-3000.safetensors \
        --shape-adapter hf://suvadityamuk/trellis-2-finetune-gso/footwear/shape_lora/adapter-3000.safetensors \
        --compare --out out/ --push-to your-username/trellis2-outputs

Adapters and configs accept a local path, `hf://<repo>/<path>` (model repo) or
`hf://datasets/<repo>/<path>`. Rank, alpha and targets come from --ss-config/--shape-config,
else from the config.json that trellis2-train pushes next to each adapter, else from the
preset in configs/lora/.
"""
import os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import argparse
import sys
import urllib.request

TRELLIS = "/workspace/TRELLIS.2"
sys.path[:0] = [TRELLIS, os.path.join(TRELLIS, "scripts")]

import torch
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image

from eval_geometry import SS_KEY, SHAPE_TARGETS, SS_TARGETS, apply_adapter, load_lora_spec, shape_keys
from trellis2.pipelines import Trellis2ImageTo3DPipeline

PRESETS = {"ss": f"{TRELLIS}/configs/lora/ss_flow_lora_gso.json",
           "shape": f"{TRELLIS}/configs/lora/slat_flow_img2shape_lora_gso.json"}


def split_hf(ref: str):
    """hf://[datasets/]owner/name/path -> (repo_id, repo_type, path)."""
    parts = ref[len("hf://"):].split("/")
    repo_type = "model"
    if parts[0] in ("datasets", "models"):
        repo_type = parts.pop(0)[:-1]
    return "/".join(parts[:2]), repo_type, "/".join(parts[2:])


def fetch(ref: str) -> str:
    """Resolve a local path, hf:// reference or URL to a local file."""
    if ref.startswith("hf://"):
        repo, repo_type, path = split_hf(ref)
        return hf_hub_download(repo, path, repo_type=repo_type)
    if ref.startswith(("http://", "https://")):
        dst = os.path.join("/tmp", os.path.basename(ref.split("?")[0]) or "input")
        urllib.request.urlretrieve(ref, dst)
        return dst
    return ref


def adapter_config(adapter: str, config: str | None, kind: str) -> str:
    """Explicit config, else config.json beside a Hub adapter, else the preset."""
    if config:
        return fetch(config)
    if adapter.startswith("hf://"):
        repo, repo_type, path = split_hf(adapter)
        cfg = f"{os.path.dirname(path)}/config.json".lstrip("/")
        if HfApi().file_exists(repo, cfg, repo_type=repo_type):
            return hf_hub_download(repo, cfg, repo_type=repo_type)
    print(f"[generate] no config for {adapter}; using the {kind} preset")
    return PRESETS[kind]


def export_glb(mesh, path: str, texture_size: int) -> None:
    import o_voxel
    mesh.simplify(16777216)  # nvdiffrast limit
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs, coords=mesh.coords,
        attr_layout=mesh.layout, voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], decimation_target=1000000,
        texture_size=texture_size, remesh=True, remesh_band=1, remesh_project=0, verbose=False)
    glb.export(path, extension_webp=True)
    print(f"[generate] wrote {path}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", required=True, nargs="+", help="input image(s): path, hf:// or URL")
    p.add_argument("--ss-adapter", help="sparse-structure LoRA .safetensors")
    p.add_argument("--shape-adapter", help="shape-flow LoRA .safetensors")
    p.add_argument("--ss-config", help="config that trained --ss-adapter (rank/alpha/targets)")
    p.add_argument("--shape-config", help="config that trained --shape-adapter")
    p.add_argument("--compare", action="store_true", help="also generate with the base model, same seed")
    p.add_argument("--out", default="/workspace/work/generate", help="output directory")
    p.add_argument("--push-to", help="dataset repo that receives the GLBs (jobs are ephemeral)")
    p.add_argument("--private", action="store_true", help="create --push-to as private")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pipeline-type", default="512", help="512 matches the trained adapters")
    p.add_argument("--texture-size", type=int, default=2048)
    p.add_argument("--model", default="microsoft/TRELLIS.2-4B")
    return p.parse_args()


def run_all(pipeline, images, tag, args):
    for ref in images:
        name = os.path.splitext(os.path.basename(ref.split("?")[0]))[0]
        out = pipeline.run(Image.open(fetch(ref)), num_samples=1, seed=args.seed,
                           pipeline_type=args.pipeline_type)[0]
        export_glb(out, os.path.join(args.out, f"{name}_{tag}.glb"), args.texture_size)


def main():
    args = parse_args()
    assert torch.cuda.is_available(), "TRELLIS.2 needs a CUDA GPU"
    os.makedirs(args.out, exist_ok=True)

    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.model)
    pipeline.cuda()

    lora = bool(args.ss_adapter or args.shape_adapter)
    if args.compare or not lora:
        run_all(pipeline, args.image, "base", args)
    if lora:
        if args.ss_adapter:
            r, a, t = load_lora_spec(adapter_config(args.ss_adapter, args.ss_config, "ss"), 8, 16, SS_TARGETS)
            apply_adapter(pipeline.models[SS_KEY], fetch(args.ss_adapter), r, a, t, merge=False)
        if args.shape_adapter:
            r, a, t = load_lora_spec(adapter_config(args.shape_adapter, args.shape_config, "shape"), 16, 32, SHAPE_TARGETS)
            for key in shape_keys(args.pipeline_type):
                apply_adapter(pipeline.models[key], fetch(args.shape_adapter), r, a, t, merge=False)
        run_all(pipeline, args.image, "lora", args)

    if args.push_to:
        api = HfApi()
        api.create_repo(args.push_to, repo_type="dataset", private=args.private, exist_ok=True)
        api.upload_folder(repo_id=args.push_to, repo_type="dataset", folder_path=args.out,
                          path_in_repo="generations", allow_patterns=["*.glb"])
        print(f"[generate] uploaded to https://huggingface.co/datasets/{args.push_to}/tree/main/generations")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
