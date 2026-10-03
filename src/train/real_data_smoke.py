"""Real-data verification smoke for the tokenizer-v2 sample (U7; R15, R17, KTD9).

Run on the governed node, after the reference build of a verification sample
(`python -m src.data.tokenize ... --build-vocab --sample-episodes N` and the GEM run with
the same vocabulary). It derives every tokenization-ablation arm's artifacts from that
sample and trains each arm for a few optimizer steps, then runs a few GEM next-event
steps and short rollouts to a disposition:

1. join the locally derived outcome labels into the clinical shard; fit value stats;
2. re-tokenize the same sample under ``scheme: decile_ablation`` (the decile arms);
3. derive the continuous-fused arm from the clinical shard;
4. each arm of ``configs/tokenization_ablation.yaml``: `setup_arm` (dry-run loaders: the
   sample vocabulary is smoke-only and training refuses it otherwise) + `train_arm` for
   ``--steps`` optimizer steps on a small trunk; pass = finite losses, the configured
   number of updates, and nonzero masked next-event targets;
5. GEM: ``--steps`` pure next-event steps on the train windows, then ``--rollouts``
   rollouts to a disposition from a validation stay's ICU-admit+24 h prefix.

Everything printed and written (``smoke_report.json`` beside the sample) is aggregate:
pass/fail, losses, update counts, booleans and rollout stop reasons. Derived artifacts
stay under the sample's governed PHI directory; checkpoints are not written.

    uv run python -m src.train.real_data_smoke --data-dir ~/Data/clif-source \\
        --sample-dir output/intermediate_phi/mimic_v2_sample \\
        --episodes output/intermediate_phi/episodes.parquet \\
        --labels output/intermediate_phi/mimic_v2_sample/labels.parquet
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import tempfile
from pathlib import Path

import polars as pl
import torch
import yaml

from src.data.segments import load_vocab_blob
from src.data.tokenize_continuous import REPRESENTATION as CONTINUOUS_FUSED

ROOT = Path(__file__).resolve().parents[2]
SMOKE_TRUNK = {"d_model": 64, "n_layers": 2, "n_heads": 2, "ffn_mult": 2, "dropout": 0.0}


def _smoke_configs(mcfg: dict, tcfg: dict, ckpt_dir: Path, *, steps: int,
                   batch: int) -> tuple[dict, dict]:
    mcfg = copy.deepcopy(mcfg)
    mcfg["trunk"].update(SMOKE_TRUNK)
    mcfg["compile"] = False
    tcfg = copy.deepcopy(tcfg)
    tcfg["batch"] = {"per_gpu": batch, "grad_accum": 1}
    tcfg["schedule"] = {"warmup_steps": 1, "total_steps": steps, "cosine_decay": True}
    tcfg["runtime"].update({"num_workers": 0, "log_every": 1000, "ckpt_every": 10**9,
                            "ckpt_dir": str(ckpt_dir), "token_budget": 0})
    tcfg["eval_schedule"] = {"val_every": 10**9}
    return mcfg, tcfg


def _join_and_stats(out: Path, labels: pl.DataFrame, cfg: dict, cohort: dict,
                    blob: dict | None = None) -> None:
    """Join the outcome labels into `out`'s shard (once) and fit its value stats;
    `blob` is `out`'s vocab.json when the caller already read it."""
    from src.data.outcome_join import join_outcomes

    blob = load_vocab_blob(out / "vocab.json") if blob is None else blob
    if not (out / "events_with_outcomes.parquet").exists():
        joined = join_outcomes(labels, pl.read_parquet(out / "events.parquet"), blob, cfg,
                               cohort)
        joined.write_parquet(out / "events_with_outcomes.parquet")
    _value_stats(out, blob)


def _value_stats(out: Path, blob: dict | None = None) -> None:
    from src.data.value_stats import compute_value_stats_from_events, write_value_stats

    blob = load_vocab_blob(out / "vocab.json") if blob is None else blob
    stats = compute_value_stats_from_events(out / "events_with_outcomes.parquet")
    write_value_stats(stats, out / "value_stats.json", vocab=blob["vocab"],
                      segments=blob["segments"], fit_partition_name="train")


def prepare_arms(sample_dir: Path, *, data_dir: Path, episodes: pl.DataFrame,
                 labels: pl.DataFrame, cfg: dict, cohort: dict, policy: dict,
                 site: str) -> dict[str, Path]:
    """Clinical (the sample itself), decile and continuous-fused arm directories."""
    from src.data.tokenize import tokenize_site
    from src.data.tokenize_continuous import write_continuous_fused_arm

    blob = load_vocab_blob(sample_dir / "vocab.json")
    size = blob["manifest"]["provenance"].get("sample_size")
    if not blob["manifest"]["provenance"].get("sample") or not size:
        raise SystemExit("the smoke runs on a verification-sample vocabulary "
                         "(tokenize with --sample-episodes)")
    _join_and_stats(sample_dir, labels, cfg, cohort, blob)

    decile = sample_dir.with_name(sample_dir.name + "_decile")
    if not (decile / "vocab.json").exists():
        dcfg = copy.deepcopy(cfg)
        dcfg["value_binning"]["scheme"] = "decile_ablation"
        tokenize_site(dcfg, site, data_dir, decile, None, episodes=episodes,
                      artifact_policy=policy, sample_episodes=int(size), workers=0)
    _join_and_stats(decile, labels, cfg, cohort)

    continuous = sample_dir.with_name(sample_dir.name + "_continuous")
    if not (continuous / "vocab.json").exists():
        write_continuous_fused_arm(sample_dir / "vocab.json",
                                   sample_dir / "events_with_outcomes.parquet", continuous,
                                   policy=policy)
    _value_stats(continuous)
    return {"clinical": sample_dir, "decile": decile, "continuous": continuous}


def _arm_dir(dirs: dict, arm: dict) -> Path:
    if arm["tokenizer"] == CONTINUOUS_FUSED:
        return dirs["continuous"]
    return dirs["decile"] if arm["scheme"] == "decile_ablation" else dirs["clinical"]


def _finite(losses: dict) -> bool:
    return all(math.isfinite(float(v)) for v in losses.values())


def run_arms(dirs: dict, *, abl: dict, mcfg: dict, tcfg: dict, n_targets: int, device,
             steps: int, text_encoder=None) -> dict[str, dict]:
    from src.train.engine import _prepare_batch
    from src.train.run_tokenization_ablation import (
        masked_target_counts,
        resolve_arm,
        setup_arm,
        train_arm,
    )

    results: dict[str, dict] = {}
    for name in abl["arms"]:
        d = _arm_dir(dirs, abl["arms"][name])
        try:
            arm = resolve_arm(abl, name, events=d / "events_with_outcomes.parquet",
                              vocab=d / "vocab.json", value_stats=d / "value_stats.json",
                              primary_vocab=dirs["clinical"] / "vocab.json")
            run = setup_arm(arm, mcfg=mcfg, tcfg=tcfg, n_targets=n_targets, device=device,
                            text_encoder=text_encoder, dry_run=True, seed=0)
            batch = _prepare_batch(next(iter(run.loaders.train)), device)
            run.model.eval()
            with torch.no_grad():
                first = {k: float(v) for k, v in run.model(batch).items()}
            run.model.train()
            _, manifest = train_arm(run, tcfg=tcfg, mcfg=mcfg, device=device,
                                    total_steps=steps, lr=float(arm["lr"]))
            masked = {k: v > 0 for k, v in masked_target_counts(batch).items()}
            updates = int(manifest.ledger.get("optimizer_updates", 0))
            passed = _finite(first) and updates == steps and masked["ntp"]
            results[name] = {"passed": passed, "optimizer_updates": updates,
                             "first_batch_losses": {k: round(v, 4) for k, v in first.items()},
                             "finite_loss": _finite(first), "nonzero_masked": masked,
                             "n_value_bins": run.n_value_bins}
        except Exception as exc:  # noqa: BLE001 - report the arm's failure, keep going
            results[name] = {"passed": False, "error": type(exc).__name__,
                             "detail": str(exc)[:300]}
        print(f"  arm {name}: {'PASS' if results[name]['passed'] else 'FAIL'}", flush=True)
    return results


def run_gem(sample_dir: Path, *, mcfg: dict, cfg: dict, device, steps: int,
            rollouts: int, max_new_tokens: int, batch: int) -> dict:
    """GEM next-event steps on train windows + rollouts from a validation prefix."""
    from src.data.collate import collate_model_samples
    from src.data.dataset import ModelDataset
    from src.data.segments import artifact_binding, n_value_bins
    from src.data.targets import TargetBuilder
    from src.model.generate import rollout_to_disposition
    from src.train.engine import _prepare_batch
    from src.train.pretrain import Model

    blob = load_vocab_blob(sample_dir / "vocab.json")
    vocab = blob["vocab"]
    gem = pl.read_parquet(sample_dir / "gem_events.parquet")
    vocab_size = int(mcfg["trunk"].get("target_vocab", 10000))
    gem_mcfg = copy.deepcopy(mcfg)
    for head in ("competing_risk", "threshold_hazard", "value_regression"):
        gem_mcfg["heads"][head]["weight"] = 0.0
    gem_mcfg["heads"]["next_event"]["weight"] = 1.0
    builder = TargetBuilder(vocab_size, mcfg["heads"]["competing_risk"]["n_time_bins"], 48,
                            {}, mode="gem")
    train_keys = sorted(gem.filter(pl.col("partition") == "train")["hosp_id"].unique())
    # A few complete train stays (every window) keep the step cheap.
    rows = [{**r, "value": [None] * len(r["token"])}
            for r in gem.filter(pl.col("hosp_id").is_in(train_keys[: 4 * batch])).to_dicts()]
    dataset = ModelDataset(rows, representation="gem", target_builder=builder,
                           expected_hashes=artifact_binding(blob))
    torch.manual_seed(0)
    model = Model(vocab_size, len(cfg["target_concepts"]), gem_mcfg,
                  n_value_bins=n_value_bins(blob)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    for step in range(steps):
        items = [dataset[(step * batch + i) % len(dataset)] for i in range(batch)]
        out = model(_prepare_batch(collate_model_samples(items), device))
        opt.zero_grad()
        out["total"].backward()
        opt.step()
        losses.append(float(out["ntp"].detach()))

    # Rollouts: a single-window validation stay, prefix through the ICU-admit+24 h anchor.
    val = (gem.filter((pl.col("partition") != "train") & (pl.col("n_windows") == 1))
           .sort("hosp_id"))
    if val.is_empty():
        return {"passed": False, "detail": "no single-window non-train stay for rollouts",
                "ntp_losses": losses}
    row = val.row(0, named=True)
    anchor = min(int(row["anchor_idx"]), len(row["token"]) - 3)
    model.eval()
    records = rollout_to_disposition(
        model.enc, row["token"][: anchor + 1], row["pos_min"][: anchor + 1], vocab=vocab,
        n_rollouts=rollouts, seed=0, max_new_tokens=max_new_tokens,
        dispositions=cfg["gem"]["dispositions"], device=device,
    )
    stops: dict[str, int] = {}
    for record in records:
        key = record["stop_reason"] + (f":{record['terminal_type']}"
                                       if record.get("terminal_type") else "")
        stops[key] = stops.get(key, 0) + 1
    passed = (len(losses) == steps and all(math.isfinite(v) for v in losses)
              and len(records) == rollouts)
    return {"passed": passed, "ntp_steps": len(losses),
            "ntp_losses": [round(v, 4) for v in losses], "rollouts": len(records),
            "rollout_stop_reasons": stops, "max_new_tokens": max_new_tokens}


def _text_encoder(kind: str):
    if kind == "real":
        return None  # tokenize_textcode loads the frozen encoder from the config
    import hashlib

    import numpy as np

    def stub(texts):
        rows = []
        for text in texts:
            seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
            rows.append(np.random.default_rng(seed).normal(size=64))
        return np.asarray(rows, dtype=np.float32)
    return stub


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", required=True, help="the site's CLIF 2.1 parquet dir")
    ap.add_argument("--sample-dir", required=True,
                    help="the verification-sample tokenization (events, vocab, GEM)")
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--labels", required=True, help="locally derived outcome labels")
    ap.add_argument("--site", default="mimic")
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--model-config", default="configs/model.yaml")
    ap.add_argument("--train-config", default="configs/train.yaml")
    ap.add_argument("--ablation-config", default="configs/tokenization_ablation.yaml")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--rollouts", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--textcode-encoder", choices=("real", "stub"), default="real",
                    help="real = the configured frozen clinical text encoder; stub = a "
                         "deterministic stand-in (no model download)")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text())
    cohort = yaml.safe_load((ROOT / cfg["cohort_contract"]).read_text())
    policy = yaml.safe_load((ROOT / cfg["artifact_policy"]).read_text())
    abl = yaml.safe_load(Path(args.ablation_config).read_text())
    device = torch.device(args.device)
    sample_dir = Path(args.sample_dir)
    episodes = pl.read_parquet(args.episodes)
    labels = pl.read_parquet(args.labels)

    print("preparing arm artifacts (clinical, decile, continuous-fused)", flush=True)
    dirs = prepare_arms(sample_dir, data_dir=Path(args.data_dir), episodes=episodes,
                        labels=labels, cfg=cfg, cohort=cohort, policy=policy,
                        site=args.site)
    with tempfile.TemporaryDirectory() as ckpt:
        mcfg, tcfg = _smoke_configs(yaml.safe_load(Path(args.model_config).read_text()),
                                    yaml.safe_load(Path(args.train_config).read_text()),
                                    Path(ckpt), steps=args.steps, batch=args.batch)
        print(f"training each ablation arm for {args.steps} steps on {device}", flush=True)
        arms = run_arms(dirs, abl=abl, mcfg=mcfg, tcfg=tcfg,
                        n_targets=len(cfg["target_concepts"]), device=device,
                        steps=args.steps, text_encoder=_text_encoder(args.textcode_encoder))
        print("GEM next-event steps + rollouts", flush=True)
        try:
            gem = run_gem(sample_dir, mcfg=mcfg, cfg=cfg, device=device, steps=args.steps,
                          rollouts=args.rollouts, max_new_tokens=args.max_new_tokens,
                          batch=args.batch)
        except Exception as exc:  # noqa: BLE001
            gem = {"passed": False, "error": type(exc).__name__, "detail": str(exc)[:300]}
    report = {"steps": args.steps, "device": str(device),
              "textcode_encoder": args.textcode_encoder, "arms": arms, "gem": gem,
              "passed": all(a["passed"] for a in arms.values()) and gem["passed"]}
    (sample_dir / "smoke_report.json").write_text(json.dumps(report, indent=2,
                                                             sort_keys=True))
    for name, result in arms.items():
        print(f"{'PASS' if result['passed'] else 'FAIL'}  arm {name}")
    print(f"{'PASS' if gem['passed'] else 'FAIL'}  gem ({gem.get('ntp_steps', 0)} steps, "
          f"{gem.get('rollouts', 0)} rollouts: {gem.get('rollout_stop_reasons')})")
    print("SMOKE PASS" if report["passed"] else "SMOKE FAIL")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
