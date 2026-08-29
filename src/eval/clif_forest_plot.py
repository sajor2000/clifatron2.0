"""Forest/box plot generator for multi-site external validation.

Reads per-site result JSONs from clif_validate.py and produces:
  1. Forest plot data (JSON) — per-outcome AUROC/AUPRC/ECE across sites
  2. Summary statistics — mean, std, range per outcome
  3. Ready for matplotlib/seaborn rendering

Usage:
    python -m src.eval.clif_forest_plot \
        --results results/MIMIC.json results/Rush.json results/UChicago.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.eval.attestation import AuthenticationError, verify_report
from src.eval.schema import EVALUABLE, validate_export


def load_site_results(paths: list[str], *, signing_keys: dict[str, bytes] | None = None,
                      require_signatures: bool = True) -> list[dict]:
    """Load site artifacts through the allow-listed schema (U5 D9).

    This was a bare `json.loads` with no schema check at all, feeding a table builder
    that treated every key except three literals as an outcome — so any field a site
    JSON happened to carry became a reported "outcome". That is an accept-anything
    loader where an allow-list is required, and it is the reader-side twin of the
    writer-side gate in `src/eval/schema.py`.

    `signing_keys` maps site_id -> shared secret. Verification is REQUIRED by default
    (U5 review #12): the previous behaviour verified only when keys happened to be
    supplied, so both ends of the trust boundary defaulted to unauthenticated. Pass
    `require_signatures=False` only for a local dry run over synthetic fixtures.

    Duplicate `release_id` values are rejected (U5 review #13). A valid signed report can
    otherwise be supplied N times and each copy counts as another site, biasing the mean
    and inflating n_sites in the headline figure.
    """
    if require_signatures and not signing_keys:
        raise AuthenticationError(
            "no signing keys supplied. Aggregating unauthenticated site reports lets a "
            "forged or altered report enter the cross-site comparison unattributed. Pass "
            "signing_keys={site_id: secret}, or require_signatures=False for a synthetic "
            "dry run."
        )
    results, seen_releases = [], {}
    for p in paths:
        blob = json.loads(Path(p).read_text())
        validate_export(blob)
        if signing_keys is not None:
            site = blob.get("site_id")
            if site not in signing_keys:
                raise AuthenticationError(
                    f"no signing key registered for site {site!r}; an unattributable "
                    "report cannot enter the aggregate"
                )
            verify_report(blob, signing_keys[site])
        release = blob.get("release_id")
        if release in seen_releases:
            raise AuthenticationError(
                f"release_id {release!r} appears twice (already loaded from "
                f"{seen_releases[release]!r}). A replayed report would be counted as an "
                "additional site."
            )
        seen_releases[release] = Path(p).name
        results.append(blob)
    return results


def build_forest_table(results: list[dict], metrics: list[str] | None = None):
    """Build (site × outcome) table for each metric.

    Outcomes come from the schema's `outcomes` block, not from whatever keys happen to
    be present. Non-evaluable outcomes contribute their status, never a number: a
    suppressed or unsupported cell must not be silently read as a missing value and
    then averaged away.
    """
    metrics = metrics or ["auroc", "auprc", "ece"]
    outcomes = sorted({name for r in results for name in (r.get("outcomes") or {})})

    table = []
    for outcome in outcomes:
        row = {"outcome": outcome}
        statuses = []
        for metric in metrics:
            values = []
            for r in results:
                block = (r.get("outcomes") or {}).get(outcome)
                if block is None:
                    values.append(float("nan"))
                    continue
                if block.get("status") != EVALUABLE:
                    statuses.append(block.get("status"))
                    values.append(float("nan"))
                    continue
                values.append((block.get("metrics") or {}).get(metric, float("nan")))
            finite = [v for v in values if v is not None and v == v]
            row[metric] = {
                "values": values,
                "mean": float(np.mean(finite)) if finite else float("nan"),
                "std": float(np.std(finite)) if len(finite) > 1 else 0.0,
                "min": float(np.min(finite)) if finite else float("nan"),
                "max": float(np.max(finite)) if finite else float("nan"),
                "n_sites": len(finite),
            }
        # Surfaced so a reader can tell "no sites could evaluate this" apart from
        # "every site scored it badly" -- the same distinction the status field exists for.
        row["non_evaluable_statuses"] = sorted(set(statuses))
        table.append(row)
    return table


def forest_plot_data(results: list[dict],
                     primary_metric: str = "auroc") -> dict:
    """Generate forest-plot-ready data: (outcome, site_name, value, ci_lower, ci_upper).

    This is the HEADLINE FIGURE from NEXT_STEPS.md §5:
    "does the model travel across N external CLIF sites?"
    """
    table = build_forest_table(results, [primary_metric])

    forest = []
    for row in table:
        outcome = row["outcome"]
        stats = row[primary_metric]
        # site_id is the OPAQUE identifier from the schema. `site_name` and `site` are
        # gone deliberately: the latter used to carry str(data_path).
        # Pair each site with ITS OWN value, then sort the pairs (U5 review #1).
        # Sorting the site list independently of `stats["values"]` -- which stays in
        # `results` order -- attributed every point in the headline figure to the wrong
        # hospital whenever --results were passed non-alphabetically.
        pairs = sorted(zip((r["site_id"] for r in results), stats["values"]),
                       key=lambda sv: sv[0])
        for site_idx, (site, val) in enumerate(pairs):
            ci = 1.96 * stats["std"] / max(stats["n_sites"], 1) ** 0.5
            forest.append({
                "outcome": outcome,
                "site": site,
                "value": val,
                "ci_lower": val - ci,
                "ci_upper": val + ci,
            })
    return forest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--out", default="output/final_no_phi/forest_plot.json")
    ap.add_argument("--signing-keys", default=None,
                    help="JSON file mapping site_id -> hex shared secret. Required "
                         "unless --unsigned-dry-run is passed.")
    ap.add_argument("--unsigned-dry-run", action="store_true",
                    help="skip signature verification; synthetic fixtures only")
    args = ap.parse_args()

    keys = None
    if args.signing_keys:
        keys = {k: bytes.fromhex(v)
                for k, v in json.loads(Path(args.signing_keys).read_text()).items()}
    results = load_site_results(args.results, signing_keys=keys,
                                require_signatures=not args.unsigned_dry_run)
    sites = [r["site_id"] for r in results]
    print(f"Loaded {len(results)} sites: {', '.join(sites)}")

    forest = forest_plot_data(results)
    table = build_forest_table(results)

    report = {
        "sites": sites,
        "n_sites": len(sites),
        "primary_metric": "auroc",
        "forest": forest,
        "summary_table": table,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nForest plot data written to {out_path}")


if __name__ == "__main__":
    main()