"""One-command synthetic reproduction of the federated-validation loop (U15/U19).

    python -m src.eval.reproduce_synthetic

Runs the whole releaser -> site -> aggregator path on SYNTHETIC fixtures, data-free and CPU:
a releaser signs a bundle (U11 Ed25519), two synthetic sites validate it and emit signed,
disclosure-controlled reports via the real site CLI (U9, draft -> approve), and the
coordinating-center aggregator (U15) verifies both, enforces the releaser signature, the
content-hash approval, anti-rollback, HMAC report authentication, and cross-release
differencing, then returns the multi-site aggregate panel.

This is NOT a new trusted code path — it composes the landed pieces unchanged, so a reviewer
can reproduce the synthetic result in one command. It never touches real data or real paths:
everything runs under a throwaway working directory of synthetic fixtures.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

from src.eval import aggregator as _agg
from src.eval import schema as _schema
from src.eval.synthetic_bundle import build_synthetic_bundle, build_synthetic_site

_SITE_KEYS = {"SITE-A": b"synthetic-site-a-report-secret",
              "SITE-B": b"synthetic-site-b-report-secret"}
_ACCESS_KEY = b"synthetic-access-chain-key"


def _reset_policy_pin() -> None:
    os.environ.pop(_schema.POLICY_OVERRIDE_ENV, None)
    _schema.min_cell_size.cache_clear()
    _schema.max_dropped_fraction.cache_clear()


def _run_site(bundle_dir: Path, site: Path, episodes: Path, trust_roles: Path,
              access_key: Path, keyfiles: dict, site_id: str, release_id: str,
              tag: str) -> Path:
    """Drive the real site CLI through the governed draft -> approve ceremony."""
    from src.eval.clif_validate import main

    out = f"output/final_no_phi/{tag}.json"
    base = [
        "clif_validate",
        "--checkpoint", str(bundle_dir),
        "--data", str(site),
        "--episode-artifact", str(episodes),
        "--site-id", site_id,
        "--release-id", release_id,
        "--out", out,
        "--ledger", f"output/intermediate_phi/{tag}_ledger.jsonl",
        "--access-log", f"output/intermediate_phi/{tag}_access.jsonl",
        "--shard-dir", f"output/intermediate_phi/{tag}_shards",
        "--signing-key-file", str(keyfiles[site_id]),
        "--access-log-key-file", str(access_key),
        "--trust-roles", str(trust_roles),
        "--rollback-state", f"output/intermediate_phi/{tag}_rollback.json",
    ]
    saved = sys.argv
    try:
        sys.argv = base  # draft: writes <out>.draft + <out>.draft.sha256
        main()
        draft_hash = Path(out + ".draft.sha256").read_text().strip()
        # approve: bound to the reviewed draft hash — the GOVERNED path, never the unsigned escape hatch.
        sys.argv = base + ["--approved", "--approved-hash", draft_hash]
        main()
    finally:
        sys.argv = saved
        _reset_policy_pin()
    return Path(out)


def run_synthetic_federation(work_dir: str | Path | None = None) -> dict:
    """Run the full synthetic loop and return the aggregate panel. Data-free, CPU.

    Builds synthetic fixtures + a signed bundle, runs two synthetic sites through the real
    governed site CLI, and aggregates their signed reports. Returns the aggregator's panel.
    """
    tmp = tempfile.TemporaryDirectory() if work_dir is None else None
    root = Path(work_dir) if work_dir is not None else Path(tmp.name)
    old_cwd = os.getcwd()
    try:
        os.chdir(root)  # the artifact policy classifies shards relative to CWD
        site = root / "site"
        episodes = build_synthetic_site(site)
        bundle_dir = build_synthetic_bundle(root / "bundle", site, episodes)
        trust_roles = root / "trust_roles.yaml"  # written beside the bundle by the releaser
        access_key = root / "access.key"
        access_key.write_bytes(_ACCESS_KEY)
        keyfiles = {}
        for sid, secret in _SITE_KEYS.items():
            kf = root / f"{sid}.key"
            kf.write_text(secret.hex())
            keyfiles[sid] = kf

        reports = []
        for sid, rel, tag in (("SITE-A", "rel-a", "site_a"), ("SITE-B", "rel-b", "site_b")):
            reports.append(_run_site(bundle_dir, site, episodes, trust_roles, access_key,
                                     keyfiles, sid, rel, tag))

        panel = _agg.aggregate_site_reports(
            [str(p) for p in reports], signing_keys=_SITE_KEYS,
            cumulative_ledger_path="output/intermediate_phi/aggregate_ledger.jsonl")
        return panel
    finally:
        os.chdir(old_cwd)
        _reset_policy_pin()
        if tmp is not None:
            tmp.cleanup()


def main() -> None:
    panel = run_synthetic_federation()
    print("=" * 70)
    print("CLIF federated-validation — synthetic reproduction (data-free, CPU)")
    print("=" * 70)
    print(f"sites aggregated : {panel['n_sites']}  {panel['site_ids']}")
    print("aggregate panel (summary statistics only — no patient-level data):")
    print(json.dumps(panel["table"], indent=2, sort_keys=True))
    print("-" * 70)
    print("All trust gates passed on the governed path: Ed25519 releaser signature, "
          "content-hash approval, anti-rollback, HMAC report authentication, and "
          "cross-release differencing.")
    print("No raw data, labels, or patient-level counts left the synthetic node.")


if __name__ == "__main__":
    main()
