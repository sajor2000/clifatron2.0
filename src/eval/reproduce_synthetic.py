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
import subprocess
import sys
import tempfile
from pathlib import Path

from src.eval import aggregator as _agg
from src.eval import schema as _schema
from src.eval.synthetic_bundle import build_synthetic_bundle, build_synthetic_site

_SITE_KEYS = {"SITE-A": b"synthetic-site-a-report-secret",
              "SITE-B": b"synthetic-site-b-report-secret"}
_ACCESS_KEY = b"synthetic-access-chain-key"


# The directory that contains the `src` package, so a child process can `-m src.eval...`
# regardless of its working directory.
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_site(root: Path, bundle_dir: Path, site: Path, episodes: Path, trust_roles: Path,
              access_key: Path, keyfiles: dict, site_id: str, release_id: str,
              tag: str) -> Path:
    """Drive the real site CLI as a CHILD PROCESS through the governed draft -> approve ceremony.

    Running the CLI out-of-process is both faithful (a real site IS a separate process) and the
    clean isolation boundary: the CLI's `sys.argv` and the environment variables it sets (e.g.
    CLIF_ACCESS_LOG_KEY_FILE) live and die in the child, never touching this process.
    """
    out = f"output/final_no_phi/{tag}.json"
    base = [
        sys.executable, "-m", "src.eval.clif_validate",
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
    # cwd=root so the CLI's relative output paths land under the throwaway dir; PYTHONPATH so
    # `-m src.eval.clif_validate` resolves against the repo whatever the cwd is.
    env = {**os.environ,
           "PYTHONPATH": (str(_REPO_ROOT) + os.pathsep
                          + os.environ.get("PYTHONPATH", "")).rstrip(os.pathsep)}
    subprocess.run(base, check=True, cwd=root, env=env,  # draft
                   capture_output=True, text=True)
    draft_hash = (root / (out + ".draft.sha256")).read_text().strip()
    # approve: bound to the reviewed draft hash — the GOVERNED path, never the unsigned escape hatch.
    subprocess.run(base + ["--approved", "--approved-hash", draft_hash],
                   check=True, cwd=root, env=env, capture_output=True, text=True)
    return root / out


def run_synthetic_federation() -> dict:
    """Run the full synthetic loop and return the aggregate panel. Data-free, CPU.

    Builds synthetic fixtures + a signed bundle, runs two synthetic sites through the real
    governed site CLI (each as a child process), and aggregates their signed reports.

    Everything runs inside a throwaway `TemporaryDirectory` — never a caller-supplied path, so it
    cannot overwrite real files. The two site CLIs run out-of-process, so their `sys.argv` and env
    mutations never reach this process. The in-process fixture builders write intermediate
    artifacts under `output/` relative to the cwd, so this briefly changes the process cwd and the
    `POLICY_OVERRIDE_ENV` pin and restores both on every exit path. It is a single-shot
    reproduction entrypoint, not intended for concurrent in-process use.
    """
    saved_cwd = os.getcwd()
    saved_policy = os.environ.get(_schema.POLICY_OVERRIDE_ENV)
    with tempfile.TemporaryDirectory() as name:
        root = Path(name)
        try:
            os.chdir(root)  # the fixture builders write output/ relative to cwd
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

            reports = [_run_site(root, bundle_dir, site, episodes, trust_roles, access_key,
                                 keyfiles, sid, rel, tag)
                       for sid, rel, tag in (("SITE-A", "rel-a", "site_a"),
                                             ("SITE-B", "rel-b", "site_b"))]

            # The aggregator reads the disclosure floor (min_cell_size) from the pinned policy.
            # The site CLIs pinned it in their own processes; pin the same bundle policy here so
            # the aggregate uses the bundle's floor, not whatever POLICY_OVERRIDE_ENV the caller
            # happened to have. Restored in the finally below.
            os.environ[_schema.POLICY_OVERRIDE_ENV] = str(bundle_dir / "artifact_policy.yaml")
            _schema.min_cell_size.cache_clear()
            _schema.max_dropped_fraction.cache_clear()

            return _agg.aggregate_site_reports(
                [str(p) for p in reports], signing_keys=_SITE_KEYS,
                cumulative_ledger_path=str(root / "output/intermediate_phi/aggregate_ledger.jsonl"))
        finally:
            os.chdir(saved_cwd)
            if saved_policy is None:
                os.environ.pop(_schema.POLICY_OVERRIDE_ENV, None)
            else:
                os.environ[_schema.POLICY_OVERRIDE_ENV] = saved_policy
            _schema.min_cell_size.cache_clear()
            _schema.max_dropped_fraction.cache_clear()


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
