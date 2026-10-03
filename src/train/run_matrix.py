"""Experiment matrix -> run specifications and launch commands (plan U6; R34, R36, R37;
KTD3, KTD11, KTD12). NEVER starts training.

`configs/experiment_matrix.yaml` lists the rows of the minimum experiment table
(notes/ai-novelty-audit.md section 5): tokenization arms, objective arms, the order-only
time arm, the language-grounded (TextCode) arm, the size sweep and the seeds, each with a
screening and a full budget (passes), plus the optional claim 1 attribution arm (off by
default) and the claim-bearing runs listed by arm and seed. `expand_matrix` validates
every referenced arm and key and expands the rows to one run per (configuration, seed,
budget) with a unique identifier; rows naming the same configuration share its runs.

    uv run python -m src.train.run_matrix                  # runs + launch commands
    uv run python -m src.train.run_matrix --edge-check     # KTD3 edge-distance table
    uv run python -m src.train.run_matrix --write          # + <run_root>/<run_id>/run_spec.json

`--write` writes each run's `run_spec.json` (exactly the `src/eval/claims_report.py`
schema, validated by its `validate_run_spec`) and a `matrix_entry.json` beside it (size,
trunk, positions, budget passes, table rows, launch commands). The vocabulary hash of
each run is its tokenization arm's frozen vocab.json (`arm_data`, first site; override
with `--vocab ARM=PATH`).

`--edge-check` prints, for every arm's frozen vocabulary, the edge-distance table of the
registered decision and control thresholds (configs/thresholds.yaml) and REFUSES, naming
the arm, a control threshold that sits on a bin edge in any arm (KTD3). The table holds
thresholds, edges and hashes only - no patient data.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = ROOT / "configs/experiment_matrix.yaml"
ABLATION_PATH = ROOT / "configs/tokenization_ablation.yaml"
OBJECTIVE_ARMS_PATH = ROOT / "configs/objective_arms.yaml"
DATA_PATH = ROOT / "configs/data.yaml"
POSITION_TAGS = {"admission_minutes": "time", "token_index": "order"}
MIN_SEEDS = 3


class MatrixError(ValueError):
    """The experiment matrix cannot be expanded into valid run specifications."""


def load_matrix(path: str | Path = MATRIX_PATH) -> dict:
    return yaml.safe_load(Path(path).read_text())


def _as_list(value) -> list:
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _check_matrix(matrix: Mapping, abl: Mapping, objective_arms: Mapping) -> None:
    from src.eval.claims_report import BUDGETS
    from src.train.pretrain import ROPE_POSITIONS, TRUNK_OVERRIDES

    seeds = matrix.get("seeds") or []
    if (len(set(seeds)) < MIN_SEEDS or len(set(seeds)) != len(seeds)
            or any(not isinstance(s, int) or isinstance(s, bool) for s in seeds)):
        raise MatrixError(f"the matrix needs at least three seeds (distinct integers), got "
                          f"{seeds}")
    budgets = matrix.get("budgets") or {}
    if set(budgets) != set(BUDGETS):
        raise MatrixError(f"budgets must be exactly {BUDGETS}, got {sorted(budgets)}")
    for name, passes in budgets.items():
        if not isinstance(passes, (int, float)) or isinstance(passes, bool) or passes <= 0:
            raise MatrixError(f"budget {name} must be a positive number of passes")
    if budgets["screening"] >= budgets["full"]:
        raise MatrixError("the screening budget must be smaller than the full budget")
    for size, trunk in (matrix.get("sizes") or {}).items():
        unknown = set(trunk) - set(TRUNK_OVERRIDES)
        if unknown:
            raise MatrixError(f"size {size} overrides unknown trunk key(s) {sorted(unknown)}")
    for position in matrix.get("positions") or ():
        if position not in ROPE_POSITIONS:
            raise MatrixError(f"unknown positions {position!r}; expected {ROPE_POSITIONS}")
    reference = matrix.get("reference") or {}
    if reference.get("size") not in (matrix.get("sizes") or {}):
        raise MatrixError(f"reference size {reference.get('size')!r} is not a listed size")
    if reference.get("positions") not in (matrix.get("positions") or ()):
        raise MatrixError("reference positions are not a listed positions value")
    launch = matrix.get("launch") or {}
    for key in ("module", "trajectory", "nproc_per_node", "sites", "run_root",
                "value_stats_file"):
        if key not in launch:
            raise MatrixError(f"launch.{key} is missing")
    if launch["module"] != "src.train.run_tokenization_ablation":
        raise MatrixError(f"launch.module {launch['module']!r} is not the arm runner")
    for arm, sites in (matrix.get("arm_data") or {}).items():
        if arm not in abl["arms"]:
            raise MatrixError(f"arm_data names unknown tokenization arm {arm!r}")
        if list(sites) != list(launch["sites"]):
            raise MatrixError(f"arm_data.{arm} must give a directory for each launch site "
                              f"{launch['sites']}")
    del objective_arms  # validated per row


def _row_configs(matrix: Mapping, abl: Mapping, objective_arms: Mapping) -> list[dict]:
    """One dict per (row, tokenization arm, objective arm, size, positions)."""
    reference = matrix["reference"]
    out = []
    ids = set()
    for row in matrix.get("rows") or ():
        rid = row.get("id")
        if not rid or rid in ids:
            raise MatrixError(f"every row needs a unique id; got {rid!r}")
        ids.add(rid)
        if not row.get("table_row"):
            raise MatrixError(f"row {rid} names no table_row")
        for key in ("tokenization_arm", "objective_arm"):
            if key not in row:
                raise MatrixError(f"row {rid} has no {key}")
        for tok in _as_list(row["tokenization_arm"]):
            if tok not in abl["arms"]:
                raise MatrixError(f"row {rid}: unknown tokenization arm {tok!r} "
                                  "(configs/tokenization_ablation.yaml)")
            if tok not in (matrix.get("arm_data") or {}):
                raise MatrixError(f"row {rid}: tokenization arm {tok!r} has no arm_data")
            for obj in _as_list(row["objective_arm"]):
                if obj not in objective_arms:
                    raise MatrixError(f"row {rid}: unknown objective arm {obj!r} "
                                      "(configs/objective_arms.yaml)")
                for size in _as_list(row.get("size", reference["size"])):
                    if size not in matrix["sizes"]:
                        raise MatrixError(f"row {rid}: unknown size {size!r}")
                    for pos in _as_list(row.get("positions", reference["positions"])):
                        if pos not in matrix["positions"]:
                            raise MatrixError(f"row {rid}: unknown positions {pos!r}")
                        out.append({"row": rid, "table_row": row["table_row"],
                                    "tokenization_arm": tok, "objective_arm": obj,
                                    "size": size, "positions": pos})
    attribution = matrix.get("attribution") or {}
    if attribution.get("enabled"):
        raise MatrixError(
            f"attribution arm {attribution.get('name')!r} is enabled, but the launcher "
            "does not implement a clinical query grid over decile input tokens yet; keep "
            "it off (plan open item: claim 1 attribution)")
    return out


def expand_matrix(matrix: Mapping, *, abl: Mapping | None = None,
                  objective_arms: Mapping | None = None) -> list[dict]:
    """Validate the matrix and expand it to run specifications (module docstring).

    Each run: ``run_id``, ``table_rows``, ``tokenization_arm``, ``objective_arm``,
    ``size``, ``trunk`` (the overrides: size dims, plus ``rope_position`` for the
    order-only arm), ``positions``, ``seed``, ``budget``, ``passes`` and
    ``claim_bearing``."""
    from src.train.curriculum import load_objective_arms

    abl = abl if abl is not None else yaml.safe_load(ABLATION_PATH.read_text())
    objective_arms = (objective_arms if objective_arms is not None
                      else load_objective_arms(OBJECTIVE_ARMS_PATH))
    _check_matrix(matrix, abl, objective_arms)
    configs: dict[tuple, dict] = {}
    for cfg in _row_configs(matrix, abl, objective_arms):
        key = (cfg["tokenization_arm"], cfg["objective_arm"], cfg["size"], cfg["positions"])
        entry = configs.setdefault(key, {"table_rows": []})
        if cfg["table_row"] not in entry["table_rows"]:
            entry["table_rows"].append(cfg["table_row"])

    reference = (matrix["reference"]["size"], matrix["reference"]["positions"])
    claim = {}
    for entry in matrix.get("claim_bearing") or ():
        if entry.get("budget") != "full":
            raise MatrixError(
                f"claim-bearing entry {entry.get('tokenization_arm')}/"
                f"{entry.get('objective_arm')} has budget {entry.get('budget')!r}: claims "
                "are read only from full-budget runs (KTD12); a screening budget is refused")
        for seed in entry.get("seeds") or ():
            key = (entry["tokenization_arm"], entry["objective_arm"], *reference)
            if key not in configs or seed not in matrix["seeds"]:
                raise MatrixError(
                    f"claim-bearing entry {entry['tokenization_arm']}/"
                    f"{entry['objective_arm']} seed {seed} matches no run at the reference "
                    "size and positions")
            claim[(entry["tokenization_arm"], entry["objective_arm"], seed)] = True

    runs = []
    for (tok, obj, size, pos), entry in configs.items():
        trunk = dict(matrix["sizes"][size])
        if pos != "admission_minutes":
            trunk["rope_position"] = pos
        for seed in matrix["seeds"]:
            for budget, passes in matrix["budgets"].items():
                runs.append({
                    "run_id": f"{tok}.{obj}.{size}.{POSITION_TAGS[pos]}.s{seed}.{budget}",
                    "table_rows": list(entry["table_rows"]),
                    "tokenization_arm": tok, "objective_arm": obj, "size": size,
                    "trunk": trunk, "positions": pos, "seed": int(seed),
                    "budget": budget, "passes": passes,
                    "claim_bearing": (budget == "full" and (size, pos) == reference
                                      and claim.get((tok, obj, seed), False)),
                })
    if len({r["run_id"] for r in runs}) != len(runs):  # pragma: no cover - by construction
        raise MatrixError("run identifiers are not unique")
    return runs


def run_dir(run: Mapping, matrix: Mapping, root: str | Path | None = None) -> Path:
    return Path(root if root is not None else matrix["launch"]["run_root"]) / run["run_id"]


def launch_commands(run: Mapping, matrix: Mapping) -> dict[str, str]:
    """The exact commands for one run: ``torchrun`` (the L40 node, one process per GPU,
    launched as ``uv run torchrun`` so the locked environment's torchrun and torch run)
    and ``cpu`` (a single process forced onto the CPU), and their recovery forms
    ``torchrun_resume`` / ``cpu_resume`` (the same command + ``--resume latest``: continue
    from the newest checkpoint in the run directory after a crash or a clean stop)."""
    launch = matrix["launch"]
    dirs = [matrix["arm_data"][run["tokenization_arm"]][site] for site in launch["sites"]]
    args = ["-m", launch["module"], "--arm", run["tokenization_arm"],
            "--objective-arm", run["objective_arm"], "--seed", str(run["seed"]),
            "--trajectory", launch["trajectory"], "--data", *dirs,
            "--site", *launch["sites"],
            "--value-stats", str(Path(dirs[0]) / launch["value_stats_file"]),
            "--passes", str(run["passes"])]
    for key, value in run["trunk"].items():
        args += ["--trunk", f"{key}={value}"]
    args += ["--run-dir", str(run_dir(run, matrix))]
    tail = " ".join(shlex.quote(a) for a in args)
    # `uv run`: the project environment's torchrun, never one found first on PATH.
    torchrun = f"uv run torchrun --nproc_per_node={int(launch['nproc_per_node'])} {tail}"
    cpu = f"uv run python {tail} --device cpu"
    return {"torchrun": torchrun, "cpu": cpu,
            "torchrun_resume": f"{torchrun} --resume latest",
            "cpu_resume": f"{cpu} --resume latest"}


def vocab_hashes(matrix: Mapping, overrides: Mapping[str, str] | None = None) -> dict[str, str]:
    """``{tokenization arm: vocabulary hash}`` of every arm whose frozen vocab.json exists
    (`arm_data`, first site; `overrides` ``{arm: path}``)."""
    from src.data.segments import artifact_binding, load_vocab_blob

    out = {}
    for arm, sites in matrix["arm_data"].items():
        path = Path((overrides or {}).get(arm) or Path(next(iter(sites.values()))) / "vocab.json")
        if path.exists():
            out[arm] = artifact_binding(load_vocab_blob(path))["vocabulary"]
    return out


def write_run_specs(runs: Sequence[Mapping], hashes: Mapping[str, str],
                    root: str | Path, matrix: Mapping | None = None) -> list[Path]:
    """Write ``<root>/<run_id>/run_spec.json`` (validated by `claims_report`) and
    ``matrix_entry.json`` for every run; refuses when an arm's vocabulary hash is
    unknown (its vocabulary is not built yet)."""
    from src.eval.claims_report import RUN_SPEC_FILE, validate_run_spec

    missing = sorted({r["tokenization_arm"] for r in runs} - set(hashes))
    if missing:
        raise MatrixError(f"no frozen vocabulary for tokenization arm(s) {missing}: build "
                          "them (or pass --vocab ARM=PATH) before writing run specs")
    matrix = matrix if matrix is not None else load_matrix()
    dirs = []
    for run in runs:
        directory = Path(root) / run["run_id"]
        spec = validate_run_spec({
            "run_id": run["run_id"], "tokenization_arm": run["tokenization_arm"],
            "objective_arm": run["objective_arm"], "seed": int(run["seed"]),
            "budget": run["budget"], "claim_bearing": bool(run["claim_bearing"]),
            "vocab_hash": hashes[run["tokenization_arm"]],
            "checkpoint": str(directory / "checkpoints"),
        })
        directory.mkdir(parents=True, exist_ok=True)
        (directory / RUN_SPEC_FILE).write_text(json.dumps(spec, indent=2, sort_keys=True))
        entry = {key: run[key] for key in ("run_id", "table_rows", "size", "trunk",
                                           "positions", "budget", "passes")}
        entry["launch"] = launch_commands(run, matrix)
        (directory / "matrix_entry.json").write_text(json.dumps(entry, indent=2,
                                                                sort_keys=True))
        dirs.append(directory)
    return dirs


# ------------------------------------------------------------------ edge distance (KTD3)

def _arm_rows(blob: Mapping, thresholds: Mapping, target_concepts) -> list[dict]:
    from src.data.threshold_grid import edge_distance
    from src.data.tokenize_continuous import REPRESENTATION, ContinuousThresholdGrid

    if blob.get("representation") == REPRESENTATION:
        # Continuous-fused: its threshold bins are the primary clinical segments'.
        return ContinuousThresholdGrid(blob, target_concepts, thresholds).edge_distance()
    return edge_distance(thresholds, blob)


def edge_distance_table(vocabs: Mapping[str, Mapping], *, thresholds: Mapping | None = None,
                        target_concepts=None) -> list[dict]:
    """Per arm, the edge-distance rows of the registered decision and control thresholds
    (`threshold_grid.edge_distance`). Refuses, naming every offending arm, a control
    threshold on a bin edge in any arm (KTD3)."""
    from src.data.threshold_grid import load_thresholds

    thresholds = load_thresholds() if thresholds is None else thresholds
    if target_concepts is None:
        target_concepts = yaml.safe_load(DATA_PATH.read_text())["target_concepts"]
    rows, refused = [], []
    for arm, blob in vocabs.items():
        for row in _arm_rows(blob, thresholds, target_concepts):
            if row["kind"] not in ("decision", "control"):
                continue
            rows.append({"arm": arm, **{k: row[k] for k in (
                "kind", "concept", "value", "direction", "binned", "on_edge",
                "nearest_edge", "distance", "threshold_bin")}})
            if row["kind"] == "control" and row["on_edge"]:
                refused.append(f"control {row['concept']} {row['value']:g} is on a bin edge "
                               f"in arm {arm!r}")
    if refused:
        error = MatrixError("; ".join(refused) + ": a control threshold must be off-edge "
                            "in every arm (KTD3); remove it from configs/thresholds.yaml")
        error.rows = rows       # the full table, printed with the refusal
        raise error
    return rows


def format_edge_table(rows: Sequence[Mapping]) -> str:
    arms = list(dict.fromkeys(r["arm"] for r in rows))
    keys = list(dict.fromkeys((r["kind"], r["concept"], r["value"], r["direction"])
                              for r in rows))
    cell = {(r["arm"], r["kind"], r["concept"], r["value"], r["direction"]): r for r in rows}
    width = max([12, *(len(a) for a in arms)])
    lines = [f"{'threshold':<34}" + "".join(f"{a:>{width + 2}}" for a in arms)]
    for kind, concept, value, direction in keys:
        label = f"{kind} {concept} {'<' if direction == 'below' else '>'} {value:g}"
        out = []
        for arm in arms:
            r = cell.get((arm, kind, concept, value, direction))
            if r is None or not r["binned"]:
                text = "not binned"
            elif r["on_edge"]:
                text = "ON edge"
            else:
                text = f"off {r['distance']:.3g}"
            out.append(f"{text:>{width + 2}}")
        lines.append(f"{label:<34}" + "".join(out))
    return "\n".join(lines)


def _arm_vocabs(matrix: Mapping, overrides: Mapping[str, str]) -> dict[str, dict]:
    from src.data.segments import load_vocab_blob

    unknown = sorted(set(overrides) - set(matrix["arm_data"]))
    if unknown:
        raise MatrixError(f"--vocab names unknown arm(s) {unknown}; matrix arms are "
                          f"{sorted(matrix['arm_data'])}")
    out, missing = {}, []
    for arm, sites in matrix["arm_data"].items():
        path = Path(overrides.get(arm) or Path(next(iter(sites.values()))) / "vocab.json")
        if path.exists():
            out[arm] = load_vocab_blob(path)
        else:
            missing.append(arm)
    if missing:
        print(f"no frozen vocabulary yet for arm(s) {missing}; their edges are unchecked",
              file=sys.stderr)
    return out


def _parse_vocab_overrides(items: Sequence[str]) -> dict[str, str]:
    out = {}
    for item in items:
        arm, sep, path = item.partition("=")
        if not sep:
            raise SystemExit(f"--vocab takes ARM=PATH, got {item!r}")
        out[arm] = path
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--matrix", default=str(MATRIX_PATH))
    ap.add_argument("--write", action="store_true",
                    help="write run_spec.json + matrix_entry.json per run under run_root")
    ap.add_argument("--run-root", default=None, help="override launch.run_root")
    ap.add_argument("--edge-check", action="store_true",
                    help="print the edge-distance table of every arm's frozen vocabulary "
                         "and refuse a control threshold on an edge in any arm")
    ap.add_argument("--vocab", action="append", default=[], metavar="ARM=PATH",
                    help="an arm's frozen vocab.json (default: its arm_data dir)")
    args = ap.parse_args(argv)

    matrix = load_matrix(args.matrix)
    if args.run_root:
        matrix["launch"]["run_root"] = args.run_root
    overrides = _parse_vocab_overrides(args.vocab)
    try:
        runs = expand_matrix(matrix)
        if args.edge_check:
            vocabs = _arm_vocabs(matrix, overrides)
            if not vocabs:
                print("edge check NOT run: no frozen vocabulary for any matrix arm (build the "
                      "vocabularies first, or pass --vocab ARM=PATH)", file=sys.stderr)
                return 2
            rows = edge_distance_table(vocabs)
            print(format_edge_table(rows))
            print(f"edge check passed: no control threshold on an edge in "
                  f"{sorted(vocabs)}")
            return 0
    except MatrixError as exc:
        if getattr(exc, "rows", None):
            print(format_edge_table(exc.rows))
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2

    by_budget = {b: sum(r["budget"] == b for r in runs) for b in matrix["budgets"]}
    print(f"{len(runs)} runs ({', '.join(f'{n} {b}' for b, n in by_budget.items())}); "
          f"budgets in passes: {matrix['budgets']}; seeds {matrix['seeds']}")
    for run in runs:
        flag = "claim-bearing" if run["claim_bearing"] else "not claim-bearing"
        print(f"\n[{run['run_id']}] {flag}; table rows: {'; '.join(run['table_rows'])}")
        commands = launch_commands(run, matrix)
        print(f"  L40:  {commands['torchrun']}")
        print(f"  CPU:  {commands['cpu']}")
    bearing = [r for r in runs if r["claim_bearing"]]
    print(f"\nclaim-bearing (full budget, listed before launch): {len(bearing)} runs")
    for run in bearing:
        print(f"  {run['tokenization_arm']} / {run['objective_arm']} seed {run['seed']}")
    print("\nrecovery: after a crash or a clean stop (SIGTERM), rerun a run's command with "
          "--resume latest (matrix_entry.json: launch.torchrun_resume)")
    for row in matrix.get("not_trained") or ():
        print(f"not trained here: {row['table_row']} -> {row['source']}")
    if args.write:
        try:
            dirs = write_run_specs(runs, vocab_hashes(matrix, overrides),
                                   matrix["launch"]["run_root"], matrix)
        except MatrixError as exc:
            print(f"REFUSED: {exc}", file=sys.stderr)
            return 2
        print(f"wrote {len(dirs)} run_spec.json under {matrix['launch']['run_root']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
