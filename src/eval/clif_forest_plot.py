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


def load_site_results(paths: list[str]) -> list[dict]:
    results = []
    for p in paths:
        blob = json.loads(Path(p).read_text())
        results.append(blob)
    return results


def build_forest_table(results: list[dict], metrics: list[str] | None = None):
    """Build (site × outcome) table for each metric."""
    metrics = metrics or ["auroc", "auprc", "ece"]
    outcomes = sorted(set().union(*(
        k for r in results for k in r if k not in ("site", "site_name", "n_stays")
    )))

    table = []
    for outcome in outcomes:
        row = {"outcome": outcome}
        for metric in metrics:
            values = [
                r.get(outcome, {}).get(metric, float("nan"))
                for r in results
            ]
            finite = [v for v in values if v is not None and v == v]
            row[metric] = {
                "values": values,
                "mean": float(np.mean(finite)) if finite else float("nan"),
                "std": float(np.std(finite)) if len(finite) > 1 else 0.0,
                "min": float(np.min(finite)) if finite else float("nan"),
                "max": float(np.max(finite)) if finite else float("nan"),
                "n_sites": len(finite),
            }
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
        sites = sorted(r.get("site_name", r.get("site", "?")) for r in results)
        for site_idx, (site, val) in enumerate(zip(sites, stats["values"])):
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
    ap.add_argument("--out", default="results/forest_plot.json")
    args = ap.parse_args()

    results = load_site_results(args.results)
    sites = [r.get("site_name", r.get("site", "?")) for r in results]
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