"""Real-data verification smoke for the tokenizer-v2 sample (U7; R15, R17, KTD9; U6).

Run on the governed node, after the reference build of a verification sample
(`python -m src.data.tokenize ... --build-vocab --sample-episodes N` and the GEM run with
the same vocabulary). It derives every tokenization-ablation arm's FULL-HOSPITALIZATION
shard from that sample and trains each arm for a few optimizer steps through the U5 gem
path (`build_loaders(representation="gem")`, in-stream time-to-event labels), then runs a
few GEM next-event steps and short rollouts to a disposition:

1. one tokenization per distinct arm vocabulary, under ``--out-dir`` (the sample itself
   is only read): the clinical sample's windows; each decile arm re-tokenized on the
   same sample with its `binning` overrides (24 h build for the vocabulary, then the
   hospitalization trajectory with it; KTD11 matched granularity); the continuous-fused
   arm derived from the clinical ``gem_events.parquet``. ``--max-stays`` keeps the same
   N train stays (and N/4 validation stays) in every arm's shard;
2. value stats fit on each shard's train windows (``gem_value_stats.json``);
3. each arm of ``configs/tokenization_ablation.yaml``: `setup_arm` with the arm's sites
   (dry-run loaders: the sample vocabulary is smoke-only) + `train_arm` for ``--steps``
   optimizer steps on a small trunk; pass = finite losses, the configured number of
   updates and nonzero masked next-event targets. Label-status shares of the in-stream
   labels (threshold queries, cause labels, competing risk) are reported per arm;
4. GEM: ``--steps`` pure next-event steps on the train windows, then ``--rollouts``
   rollouts to a disposition from a validation stay's ICU-admit+24 h prefix.

Everything printed and written (``smoke_report.json`` in ``--out-dir``) is aggregate:
pass/fail, losses, update counts, label-status counts and shares, bin-count exceptions,
booleans and rollout stop reasons. Derived shards are patient-level and stay under the
governed ``output/intermediate_phi`` tree; checkpoints are not written.

    uv run python -m src.train.real_data_smoke --data-dir ~/Data/clif-source \\
        --sample-dir output/intermediate_phi/mimic_v2_sample \\
        --episodes output/intermediate_phi/episodes.parquet \\
        --out-dir output/intermediate_phi/smoke_u6 --max-stays 64 --textcode-encoder stub
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import tempfile
from pathlib import Path

import polars as pl
import torch
import yaml

from src.data.segments import load_vocab_blob
from src.data.tokenize_continuous import REPRESENTATION as CONTINUOUS_FUSED

ROOT = Path(__file__).resolve().parents[2]
SMOKE_TRUNK = {"d_model": 64, "n_layers": 2, "n_heads": 2, "ffn_mult": 2, "dropout": 0.0}
GEM_EVENTS = "gem_events.parquet"
GEM_VALUE_STATS = "gem_value_stats.json"


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


def arm_key(arm: dict) -> str:
    """Which tokenization an arm reads: clinical, decile, decile_forced or continuous."""
    if arm["tokenizer"] == CONTINUOUS_FUSED:
        return "continuous"
    if arm["scheme"] == "decile_ablation":
        forced = (arm.get("binning") or {}).get("decile_forced_edges", False)
        return "decile_forced" if forced else "decile"
    return "clinical"


def _keep_keys(gem: pl.DataFrame, max_stays: int | None) -> list[str] | None:
    """The same stays in every arm: the first N train and N/4 validation stays by key."""
    if max_stays is None:
        return None
    keys = []
    for partition, n in (("train", max_stays), ("validation", max(1, max_stays // 4))):
        keys += sorted(gem.filter(pl.col("partition") == partition)["hosp_id"]
                       .unique().to_list())[:n]
    return keys


def _write_shard(gem: pl.DataFrame, keys: list[str] | None, out: Path, policy: dict) -> None:
    from src.data.cohort import validate_artifact_destination

    path = out / GEM_EVENTS
    validate_artifact_destination(path, "patient_level_phi", policy)
    out.mkdir(parents=True, exist_ok=True)
    (gem if keys is None else gem.filter(pl.col("hosp_id").is_in(keys))).write_parquet(path)


def _gem_value_stats(out: Path) -> None:
    from src.data.value_stats import compute_value_stats_from_events, write_value_stats

    blob = load_vocab_blob(out / "vocab.json")
    stats = compute_value_stats_from_events(out / GEM_EVENTS)
    write_value_stats(stats, out / GEM_VALUE_STATS, vocab=blob["vocab"],
                      segments=blob["segments"], fit_partition_name="train")


def prepare_arms(sample_dir: Path, out_dir: Path, *, data_dir: Path,
                 episodes: pl.DataFrame, cfg: dict, policy: dict, site: str, abl: dict,
                 max_stays: int | None = None) -> dict[str, Path]:
    """``{tokenization key: dir}``, each dir holding vocab.json, gem_events.parquet and
    gem_value_stats.json (module docstring, steps 1-2). The sample is only read."""
    from src.data.tokenize import tokenize_site
    from src.data.tokenize_continuous import write_continuous_fused_arm
    from src.train.run_tokenization_ablation import arm_data_config

    blob = load_vocab_blob(sample_dir / "vocab.json")
    size = blob["manifest"]["provenance"].get("sample_size")
    if not blob["manifest"]["provenance"].get("sample") or not size:
        raise SystemExit("the smoke runs on a verification-sample vocabulary "
                         "(tokenize with --sample-episodes)")
    sample_gem = pl.read_parquet(sample_dir / GEM_EVENTS)
    keys = _keep_keys(sample_gem, max_stays)
    dirs: dict[str, Path] = {}
    clinical = out_dir / "clinical"
    if not (clinical / GEM_EVENTS).exists():
        _write_shard(sample_gem, keys, clinical, policy)
        shutil.copyfile(sample_dir / "vocab.json", clinical / "vocab.json")
    dirs["clinical"] = clinical
    del sample_gem

    for arm in abl["arms"].values():
        key = arm_key(arm)
        if key in dirs or key == "continuous":
            continue
        out = out_dir / key
        if not (out / GEM_EVENTS).exists():
            arm_cfg = arm_data_config(cfg, arm)
            build = out / "build"
            tokenize_site(arm_cfg, site, data_dir, build, None, episodes=episodes,
                          artifact_policy=policy, sample_episodes=int(size), workers=0)
            arm_blob = json.loads((build / "vocab.json").read_text())
            tokenize_site(arm_cfg, site, data_dir, build, arm_blob, episodes=episodes,
                          artifact_policy=policy, sample_episodes=int(size), workers=0,
                          trajectory="hospitalization")
            _write_shard(pl.read_parquet(build / GEM_EVENTS), keys, out, policy)
            shutil.copyfile(build / "vocab.json", out / "vocab.json")
            for report in ("tokenization_report.json", "gem_tokenization_report.json"):
                if (build / report).exists():
                    shutil.copyfile(build / report, out / report)
            shutil.rmtree(build)
        dirs[key] = out

    continuous = out_dir / "continuous"
    if not (continuous / GEM_EVENTS).exists():
        write_continuous_fused_arm(clinical / "vocab.json", clinical / GEM_EVENTS,
                                   continuous, policy=policy)
    dirs["continuous"] = continuous
    for out in dirs.values():
        if not (out / GEM_VALUE_STATS).exists():
            _gem_value_stats(out)
    return dirs


def stay_targets(frame: pl.DataFrame) -> list[dict]:
    """Whole stays (windows concatenated in order) in the shape `TargetBuilder.build`
    reads, with each window's span: what the gem dataset labels per stay."""
    stays: dict[str, dict] = {}
    for row in frame.sort(["hosp_id", "continuation_index"]).iter_rows(named=True):
        stay = stays.setdefault(str(row["hosp_id"]), {
            "episode_key": str(row["hosp_id"]), "token": [], "pos_min": [], "value": [],
            "target_eligible": [], "anchor_idx": None, "outcomes": [], "windows": []})
        lo = len(stay["token"])
        n = len(row["token"])
        stay["token"] += list(row["token"])
        stay["pos_min"] += list(row["pos_min"])
        stay["value"] += list(row.get("value") or [None] * n)
        stay["target_eligible"] += list(row["target_eligible"])
        stay["windows"].append((lo, lo + n))
    return list(stays.values())


def label_shares(builder, gem_path: Path, *, max_stays: int = 64) -> dict:
    """Aggregate label-status counts and shares of the in-stream labels over (up to)
    `max_stays` train stays (`targets.anchor_status_shares`)."""
    from src.data.targets import anchor_status_shares

    frame = pl.read_parquet(gem_path).filter(pl.col("partition") == "train")
    keys = sorted(frame["hosp_id"].unique().to_list())[:max_stays]
    stays = stay_targets(frame.filter(pl.col("hosp_id").is_in(keys)))
    return anchor_status_shares(builder.build(stay) for stay in stays)


def _finite(losses: dict) -> bool:
    return all(math.isfinite(float(v)) for v in losses.values())


def run_arms(dirs: dict, *, abl: dict, mcfg: dict, tcfg: dict, dcfg: dict, site: str,
             device, steps: int, seed: int = 0, text_encoder=None,
             label_stays: int = 64, thresholds: dict | None = None) -> dict[str, dict]:
    """Every ablation arm on its full-hospitalization shard: `setup_arm` through the gem
    path, one forward pass, `train_arm` for `steps` updates, and the label-status shares
    of its in-stream labels. Aggregate results only."""
    from src.train.engine import _prepare_batch
    from src.train.run_tokenization_ablation import (
        masked_target_counts,
        resolve_arm,
        setup_arm,
        train_arm,
    )

    results: dict[str, dict] = {}
    for name in abl["arms"]:
        d = dirs[arm_key(abl["arms"][name])]
        try:
            arm = resolve_arm(abl, name, events=d / GEM_EVENTS, vocab=d / "vocab.json",
                              value_stats=d / GEM_VALUE_STATS,
                              primary_vocab=dirs["clinical"] / "vocab.json")
            run = setup_arm(arm, mcfg=mcfg, tcfg=tcfg, n_targets=len(dcfg["target_concepts"]),
                            device=device, text_encoder=text_encoder, dry_run=True,
                            seed=seed, sites={site: d / GEM_EVENTS}, dcfg=dcfg,
                            thresholds=thresholds)
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
            provenance = run.vocab_blob["manifest"].get("provenance") or {}
            granularity = provenance.get("matched_granularity")
            results[name] = {
                "passed": passed, "optimizer_updates": updates,
                "first_batch_losses": {k: round(v, 4) for k, v in first.items()},
                "finite_loss": _finite(first), "nonzero_masked": masked,
                "n_value_bins": run.n_value_bins, "embedding_rows": run.vocab_size,
                "parameters": manifest.parameters,
                "label_status": label_shares(
                    run.loaders.train_dataset.target_builder, d / GEM_EVENTS,
                    max_stays=label_stays),
                "matched_granularity": None if granularity is None else {
                    "matched": granularity["matched"],
                    "exceptions": granularity["exceptions"],
                    "forced_edges": granularity["forced_edges"]},
            }
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
    from src.train.pretrain import Model, embedding_vocab_size

    blob = load_vocab_blob(sample_dir / "vocab.json")
    vocab = blob["vocab"]
    gem = pl.read_parquet(sample_dir / GEM_EVENTS)
    vocab_size = embedding_vocab_size(blob, mcfg)
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
    return {"passed": passed, "ntp_steps": len(losses), "embedding_rows": vocab_size,
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
    ap.add_argument("--out-dir", required=True,
                    help="governed directory (under output/intermediate_phi) for the arms' "
                         "derived shards and smoke_report.json; the sample is only read")
    ap.add_argument("--max-stays", type=int, default=None,
                    help="keep the same N train (+N/4 validation) stays in every arm")
    ap.add_argument("--seed", type=int, default=0)
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
    policy = yaml.safe_load((ROOT / cfg["artifact_policy"]).read_text())
    abl = yaml.safe_load(Path(args.ablation_config).read_text())
    device = torch.device(args.device)
    sample_dir = Path(args.sample_dir)
    out_dir = Path(args.out_dir)
    episodes = pl.read_parquet(args.episodes)

    print("preparing full-hospitalization arm shards (clinical, deciles, continuous-fused)",
          flush=True)
    dirs = prepare_arms(sample_dir, out_dir, data_dir=Path(args.data_dir),
                        episodes=episodes, cfg=cfg, policy=policy, site=args.site, abl=abl,
                        max_stays=args.max_stays)
    with tempfile.TemporaryDirectory() as ckpt:
        mcfg, tcfg = _smoke_configs(yaml.safe_load(Path(args.model_config).read_text()),
                                    yaml.safe_load(Path(args.train_config).read_text()),
                                    Path(ckpt), steps=args.steps, batch=args.batch)
        print(f"training each ablation arm for {args.steps} steps on {device}", flush=True)
        arms = run_arms(dirs, abl=abl, mcfg=mcfg, tcfg=tcfg, dcfg=cfg, site=args.site,
                        device=device, steps=args.steps, seed=args.seed,
                        text_encoder=_text_encoder(args.textcode_encoder))
        print("GEM next-event steps + rollouts", flush=True)
        try:
            gem = run_gem(dirs["clinical"], mcfg=mcfg, cfg=cfg, device=device, steps=args.steps,
                          rollouts=args.rollouts, max_new_tokens=args.max_new_tokens,
                          batch=args.batch)
        except Exception as exc:  # noqa: BLE001
            gem = {"passed": False, "error": type(exc).__name__, "detail": str(exc)[:300]}
    report = {"steps": args.steps, "device": str(device), "seed": args.seed,
              "max_stays": args.max_stays,
              "textcode_encoder": args.textcode_encoder, "arms": arms, "gem": gem,
              "passed": all(a["passed"] for a in arms.values()) and gem["passed"]}
    (out_dir / "smoke_report.json").write_text(json.dumps(report, indent=2,
                                                             sort_keys=True))
    for name, result in arms.items():
        print(f"{'PASS' if result['passed'] else 'FAIL'}  arm {name}")
    print(f"{'PASS' if gem['passed'] else 'FAIL'}  gem ({gem.get('ntp_steps', 0)} steps, "
          f"{gem.get('rollouts', 0)} rollouts: {gem.get('rollout_stop_reasons')})")
    print("SMOKE PASS" if report["passed"] else "SMOKE FAIL")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
