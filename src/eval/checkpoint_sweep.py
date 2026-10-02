"""Compare training checkpoints on the held-out validation partition."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.collate import collate_model_samples
from src.data.dataset import ModelDataset
from src.data.segments import artifact_binding, load_vocab_blob, n_value_bins
from src.data.targets import TargetBuilder
from src.data.value_stats import load_value_stats
from src.train.checkpoint import verify_checkpoint_binding
from src.train.engine import _prepare_batch
from src.train.pretrain import Model, _load_decile_records


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_best(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the finite result with the lowest total validation loss."""
    finite = [row for row in results if torch.isfinite(torch.tensor(row["losses"]["total"]))]
    if not finite:
        raise ValueError("checkpoint sweep produced no finite validation losses")
    return min(finite, key=lambda row: float(row["losses"]["total"]))


def evaluate_checkpoint(
    checkpoint: str | Path,
    *,
    model: Model,
    loader: DataLoader,
    device: torch.device,
    vocab_blob: dict,
) -> dict[str, Any]:
    """Validation losses of one checkpoint, refused unless it is bound to `vocab_blob`
    (the vocab.json the validation shard was encoded with)."""
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    verify_checkpoint_binding(blob, vocab_blob)
    state = blob.get("model")
    if not isinstance(state, dict):
        raise ValueError(f"{checkpoint} does not contain a model state dictionary")
    bad = [
        name
        for name, value in state.items()
        if isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all())
    ]
    if bad:
        raise ValueError(f"{checkpoint} contains non-finite tensors: {', '.join(bad[:5])}")

    model.load_state_dict(state)
    model.to(device).eval()
    sums = {name: 0.0 for name in ("ntp", "cr", "th", "val", "total")}
    batches = 0
    with torch.no_grad(), torch.autocast(
        "cuda" if device.type == "cuda" else "cpu", dtype=torch.bfloat16
    ):
        for batch in loader:
            losses = model(_prepare_batch(batch, device))
            nonfinite = [
                name
                for name, value in losses.items()
                if isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all())
            ]
            if nonfinite:
                raise FloatingPointError(
                    f"{checkpoint} produced non-finite validation losses: {', '.join(nonfinite)}"
                )
            for name in sums:
                sums[name] += float(losses[name].detach())
            batches += 1
    if batches == 0:
        raise ValueError("validation loader produced zero batches")

    return {
        "checkpoint": Path(checkpoint).name,
        "checkpoint_sha256": _sha256_file(checkpoint),
        "step": int(blob.get("step", -1)),
        "epoch": int(blob.get("epoch", -1)),
        "validation_batches": batches,
        "losses": {name: value / batches for name, value in sums.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--value-stats", required=True)
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--out", default="results/checkpoint_sweep.json")
    args = parser.parse_args()

    data_dir = Path(args.data)
    model_config = yaml.safe_load(Path(args.model_config).read_text())
    data_config = yaml.safe_load(Path(args.data_config).read_text())
    vocab_blob = load_vocab_blob(data_dir / "vocab.json")
    binding = artifact_binding(vocab_blob)  # refuses a pre-v2 vocabulary
    value_stats = load_value_stats(
        args.value_stats,
        expected_vocab_hash=binding["vocabulary"],
        expected_segments_hash=binding["numeric_edges"],
        expected_fit_partition="train",
    )

    events_path = data_dir / "events_with_outcomes.parquet"
    records = _load_decile_records(events_path, partition="validation")
    target_builder = TargetBuilder(
        vocab_size=model_config["trunk"].get("target_vocab", 10000),
        n_time_bins=model_config["heads"]["competing_risk"]["n_time_bins"],
        horizon_hours=model_config["heads"]["competing_risk"].get("horizon_hours", 48),
        value_stats=value_stats,
        run_seed=42,
    )
    dataset = ModelDataset(
        records,
        representation="decile",
        target_builder=target_builder,
        expected_hashes=binding,
        epoch=0,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_model_samples,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    n_targets = len(data_config["target_concepts"])
    vocab_size = model_config["trunk"].get("target_vocab", 10000)
    model = Model(vocab_size, n_targets, model_config, n_value_bins=n_value_bins(vocab_blob))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    results = []
    for checkpoint in args.checkpoints:
        result = evaluate_checkpoint(
            checkpoint,
            model=model,
            loader=loader,
            device=device,
            vocab_blob=vocab_blob,
        )
        results.append(result)
        losses = result["losses"]
        print(
            f"step={result['step']} total={losses['total']:.4f} "
            f"ntp={losses['ntp']:.4f} cr={losses['cr']:.4f} "
            f"th={losses['th']:.4f} val={losses['val']:.4f}",
            flush=True,
        )

    best = select_best(results)
    payload = {
        "partition": "validation",
        "selection_metric": "total_loss",
        "results": results,
        "selected_checkpoint": best["checkpoint"],
        "selected_checkpoint_sha256": best["checkpoint_sha256"],
        "selected_step": best["step"],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"selected step={best['step']} -> {out}")


if __name__ == "__main__":
    main()
