"""Log the held-out geometry eval (base vs LoRA) to Trackio.

``scripts/eval_geometry.py`` writes ``metrics.csv`` (per-object rows plus a
``__mean__`` row for each of ``base`` and ``lora``) and a few generated meshes under
``vis/``. This surfaces that in the *same* Space as the training runs:

  * base/lora means as scalar metrics (+ the base->lora deltas, the headline result),
  * the per-object numbers as a Table, and
  * the generated meshes + csv as a downloadable artifact.

    python jobs/log_eval.py --eval_dir results/eval
"""
from __future__ import annotations

import argparse
import os

import pandas as pd
import trackio

METRICS = ("chamfer", "fscore", "normal_consistency", "precision", "recall")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval_dir", required=True, help="eval_geometry.py output dir (holds metrics.csv, vis/)")
    ap.add_argument("--project", default="trellis2-geoflow-lora")
    ap.add_argument("--name", default="holdout-eval")
    args = ap.parse_args()

    df = pd.read_csv(os.path.join(args.eval_dir, "metrics.csv"))
    trackio.init(project=args.project, name=args.name,
                 space_id=os.environ.get("TRACKIO_SPACE_ID"),
                 private=os.environ.get("TRACKIO_PRIVATE", "0") == "1")

    means = {r["model"]: r for _, r in df[df.object_id == "__mean__"].iterrows()}
    metrics = {f"{tag}/{k}": float(r[k]) for tag, r in means.items() for k in METRICS}
    if "base" in means and "lora" in means:  # the headline: did the LoRA move geometry the right way?
        metrics["delta/chamfer"] = float(means["lora"]["chamfer"] - means["base"]["chamfer"])  # want < 0
        metrics["delta/fscore"] = float(means["lora"]["fscore"] - means["base"]["fscore"])      # want > 0

    per_object = df[df.object_id != "__mean__"].copy()
    trackio.log({**metrics, "holdout_metrics": trackio.Table(dataframe=per_object)})

    artifact = trackio.Artifact(name="holdout-eval", type="evaluation",
                                description="Base-vs-LoRA geometry metrics + generated meshes on the held-out split.")
    artifact.add_file(os.path.join(args.eval_dir, "metrics.csv"))
    vis_dir = os.path.join(args.eval_dir, "vis")
    if os.path.isdir(vis_dir):
        artifact.add_dir(vis_dir)
    trackio.log_artifact(artifact)
    trackio.finish()

    if "base" in means and "lora" in means:
        print(f"chamfer {means['base']['chamfer']:.5f} -> {means['lora']['chamfer']:.5f} | "
              f"fscore {means['base']['fscore']:.4f} -> {means['lora']['fscore']:.4f}")


if __name__ == "__main__":
    main()
