"""Finetune-vs-from-scratch comparison report.

After running src.train.run_arm for every arm in configs/ablation.yaml, this
script reads the per-arm metrics JSONs and produces:

  1. outcome-by-arm table (AUROC/AUPRC/ECE per outcome, per arm)
  2. headroom figure data (gain over no-pretrain baseline)
  3. transfer-gap data (in-domain vs zero-shot held-out)

Evidence anchors:
  frozen_backbone_head_only  Al Attrach 2025 (frozen>trainable); Mataraso 2025
  joint_finetune             Al Attrach 2025 (unfreeze hurts)
  from_scratch               TOO-BERT PMC12177421
  no_pretrain_baseline       negative control

Usage:
    python -m src.eval.ablation_compare --results results/ablation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_arm_metrics(results_dir: Path, arm: str) -> dict | None:
    path = results_dir / arm / "metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def build_outcome_table(arms: dict, results: dict[str, dict]) -> list:
    """Each row is (outcome, {arm: {auroc, auprc, ece}})."""
    outcomes = {}
    for arm_name, metrics in results.items():
        if metrics is None:
            continue
        for task, task_metrics in metrics.get("tasks", {}).items():
            outcomes.setdefault(task, {})[arm_name] = {
                "auroc": task_metrics.get("auroc"),
                "auprc": task_metrics.get("auprc"),
                "ece": task_metrics.get("ece"),
            }
    return [
        {"outcome": outcome, **arms_map}
        for outcome, arms_map in sorted(outcomes.items())
    ]


def build_headroom_table(results: dict[str, dict],
                         baseline_arm: str = "no_pretrain_baseline") -> list:
    """AUROC gain over the no-pretrain baseline, per outcome per arm."""
    baseline = results.get(baseline_arm, {}).get("tasks", {})
    rows = []
    for arm_name, metrics in results.items():
        if arm_name == baseline_arm or metrics is None:
            continue
        for task, task_metrics in metrics.get("tasks", {}).items():
            base_auroc = baseline.get(task, {}).get("auroc", 0)
            if base_auroc is None:
                continue
            gain = (task_metrics.get("auroc") or 0) - (base_auroc or 0)
            rows.append({
                "arm": arm_name,
                "outcome": task,
                "arm_auroc": task_metrics.get("auroc"),
                "baseline_auroc": base_auroc,
                "gain": round(gain, 4),
            })
    return sorted(rows, key=lambda r: r["outcome"])


def build_transfer_table(results: dict[str, dict],
                         in_domain_outcomes: list[str],
                         zero_shot_outcomes: list[str]) -> list:
    """Compare in-domain vs zero-shot AUROC."""
    rows = []
    for arm_name, metrics in results.items():
        if metrics is None:
            continue
        tasks = metrics.get("tasks", {})
        domain_aurocs = [tasks[t]["auroc"] for t in in_domain_outcomes
                         if t in tasks and tasks[t].get("auroc") is not None]
        zero_aurocs = [tasks[t]["auroc"] for t in zero_shot_outcomes
                       if t in tasks and tasks[t].get("auroc") is not None]
        if domain_aurocs and zero_aurocs:
            rows.append({
                "arm": arm_name,
                "in_domain_mean_auroc": round(sum(domain_aurocs) / len(domain_aurocs), 4),
                "zero_shot_mean_auroc": round(sum(zero_aurocs) / len(zero_aurocs), 4),
                "transfer_gap": round(
                    (sum(domain_aurocs) / len(domain_aurocs))
                    - (sum(zero_aurocs) / len(zero_aurocs)),
                    4,
                ),
            })
    return sorted(rows, key=lambda r: abs(r["transfer_gap"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/ablation")
    ap.add_argument("--ablation-config", default="configs/ablation.yaml")
    args = ap.parse_args()

    import yaml
    abl = yaml.safe_load(Path(args.ablation_config).read_text())
    results_dir = Path(args.results)
    arms = abl["arms"]

    results = {}
    for arm_name in arms:
        arm_metrics = load_arm_metrics(results_dir, arm_name)
        if arm_metrics is None:
            print(f"[skip] {arm_name} — no metrics.json found")
            continue
        results[arm_name] = arm_metrics

    if not results:
        print("No arm results found. Run src.train.run_arm for each arm first.")
        return

    outcome_table = build_outcome_table(arms, results)
    headroom = build_headroom_table(results)
    in_domain = abl["shared"]["outcomes"]
    zero_shot = abl["shared"]["zero_shot_outcomes"]
    transfer = build_transfer_table(results, in_domain, zero_shot)

    print(json.dumps({
        "outcome_by_arm": outcome_table,
        "headroom_over_baseline": headroom,
        "transfer_gap": transfer,
        "evidence": {
            arm_name: arm["tags"]
            for arm_name, arm in arms.items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()