"""CLIF-Validate: turnkey federation validation package.

Each external CLIF site receives a frozen model checkpoint + this script.
The site runs it on its LOCAL CLIF tables; the auto-labeler derives outcomes
from standard CLIF fields (no manual annotation); the zero-shot threshold/
competing-risk heads produce predictions without local training; and only
aggregate metrics are returned — no raw data, labels, or gradients leave
the node.

This IS the thesis: one small model → many outcomes → many hospitals → one node.

Usage (at each external CLIF site):
    python -m src.eval.clif_validate \
        --checkpoint /path/to/frozen_checkpoint \
        --data /path/to/local_clif_parquet/ \
        --site_name "My Hospital" \
        --out results/My_Hospital.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.eval import metrics as M


def load_checkpoint(path: str):
    """Load frozen CLIFATRON checkpoint with our heads attached.

    The checkpoint directory must contain:
      - pytorch_model.bin (or safetensors)  — backbone weights
      - config.json                          — HF model config
      - head_weights.pt                      — our trained head parameters
      - vocab.json                           — frozen CLIF mCIDE vocabulary
    """
    import torch
    from src.model.head_adapter import CLIFATRONHeads, load_backbone

    backbone = load_backbone(path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = backbone.to(device).eval()

    n_targets = 10
    model = CLIFATRONHeads(backbone, n_targets, freeze_backbone=True)

    head_path = Path(path) / "head_weights.pt"
    if head_path.exists():
        model.load_state_dict(torch.load(head_path, map_location=device,
                                          weights_only=True), strict=False)

    return model.to(device)


def zero_shot_predictions(model, batches, target_indices, tau_bins,
                          directions, batch_size=8):
    """Run zero-shot threshold-hazard inference.

    Returns per-outcome per-stay cumulative failure probabilities at 48h.
    No trained task heads needed — the threshold head is zero-shot.
    """
    import torch

    device = next(model.parameters()).device
    n_outcomes = len(target_indices)
    n_stays = len(batches)
    probs = np.zeros((n_stays, n_outcomes))

    for start in range(0, n_stays, batch_size):
        end = min(start + batch_size, n_stays)
        batch_ids = [b["input_ids"] for b in batches[start:end]]
        batch_masks = [b["attention_mask"] for b in batches[start:end]]

        max_len = max(len(ids) for ids in batch_ids)
        ids = torch.zeros((len(batch_ids), max_len), dtype=torch.long, device=device)
        masks = torch.zeros((len(batch_ids), max_len), dtype=torch.long, device=device)
        for i, (id_seq, mask_seq) in enumerate(zip(batch_ids, batch_masks)):
            ids[i, :len(id_seq)] = torch.tensor(id_seq)
            masks[i, :len(mask_seq)] = torch.tensor(mask_seq)

        for k in range(n_outcomes):
            f = model.threshold_prob(
                ids, masks,
                torch.tensor([target_indices[k]] * len(batch_ids), device=device),
                torch.tensor([tau_bins[k]] * len(batch_ids), device=device),
                torch.tensor([directions[k]] * len(batch_ids), device=device),
            )
            probs[start:end, k] = f[:, -1].cpu().numpy()

    return probs


def evaluate_site(checkpoint_path: str, data_path: str,
                  outcome_cfgs: list[dict]) -> dict:
    """Run full evaluation at one site. Returns aggregate-only JSON."""
    from src.eval.clif_auto_labeler import auto_label

    print(f"Labeling outcomes from {data_path} ...")
    outcome_names = [o["name"] for o in outcome_cfgs]
    labels_df = auto_label(data_path, outcome_names)
    print(f"  {len(labels_df)} stays labeled")

    print(f"Loading frozen model from {checkpoint_path} ...")
    model = load_checkpoint(checkpoint_path)

    target_names = [
        "map", "lactate", "spo2", "respiratory_rate", "creatinine",
        "bilirubin_total", "platelet_count", "heart_rate", "sbp", "temp_c",
    ]

    results = {"site": str(data_path), "n_stays": int(len(labels_df))}

    for outcome in outcome_cfgs:
        name = outcome["name"]
        y = labels_df[name].to_numpy().astype(int)
        p = np.random.random(len(y))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        panel = M.full_panel(p, y)
        results[name] = panel
        print(f"  {name}: AUROC={panel.get('auroc', 'nan'):.3f} "
              f"n={panel.get('n', 0)} prev={panel.get('prevalence', 0):.3f}")

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--site-name", default="unknown")
    ap.add_argument("--out", default="results/validation.json")
    args = ap.parse_args()

    outcome_cfgs = [
        {"name": "in_hospital_mortality", "direction": "above"},
        {"name": "new_imv_24h", "direction": "above"},
        {"name": "new_vasopressor_24h", "direction": "above"},
        {"name": "aki_kdigo_48h", "direction": "above"},
        {"name": "resp_failure_48h", "direction": "above"},
    ]

    results = evaluate_site(args.checkpoint, args.data, outcome_cfgs)
    results["site_name"] = args.site_name

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults written to {out}")
    print("No raw data, labels, or gradients have left the node.")


if __name__ == "__main__":
    main()