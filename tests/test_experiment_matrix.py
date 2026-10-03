"""Experiment matrix (plan U6; R34, R36, R37; KTD11, KTD12): `configs/experiment_matrix.yaml`
expanded by `src.train.run_matrix` into run specifications, run_spec.json files, launch
commands and the edge-distance check. Nothing here trains."""

from __future__ import annotations

import copy
import io
import json
import re
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MATRIX = yaml.safe_load((ROOT / "configs/experiment_matrix.yaml").read_text())
HEX = "ab" * 32


def expand(matrix=None):
    from src.train.run_matrix import expand_matrix

    return expand_matrix(copy.deepcopy(MATRIX if matrix is None else matrix))


def audit_table_rows() -> list[str]:
    text = (ROOT / "notes/ai-novelty-audit.md").read_text()
    section = text.split("## 5. Minimum experiment table", 1)[1].split("\n## ", 1)[0]
    rows = []
    for line in section.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if line.startswith("|") and cells[0] not in ("Row", "---") and not set(cells[0]) <= {"-"}:
            rows.append(cells[0])
    return rows


class ExpansionTest(unittest.TestCase):
    def test_expands_to_the_expected_runs_with_unique_identifiers(self):
        runs = expand()
        configs = {(r["tokenization_arm"], r["objective_arm"], r["size"], r["positions"])
                   for r in runs}
        # 12 rows -> 16 configurations (4 objective ablations, 2 extra sizes).
        self.assertEqual(len(configs), 16)
        self.assertEqual(len(runs), 16 * len(MATRIX["seeds"]) * len(MATRIX["budgets"]))
        self.assertEqual(len({r["run_id"] for r in runs}), len(runs))
        self.assertGreaterEqual(len(MATRIX["seeds"]), 3)
        for run in runs:
            self.assertRegex(run["run_id"], r"\A[a-z0-9_.\-]+\Z")

    def test_every_referenced_config_key_exists(self):
        from src.train.curriculum import load_objective_arms
        from src.train.pretrain import ROPE_POSITIONS, TRUNK_OVERRIDES

        abl = yaml.safe_load((ROOT / "configs/tokenization_ablation.yaml").read_text())
        objective = load_objective_arms()
        for run in expand():
            self.assertIn(run["tokenization_arm"], abl["arms"])
            self.assertIn(run["objective_arm"], objective)
            self.assertIn(run["positions"], ROPE_POSITIONS)
            self.assertLessEqual(set(run["trunk"]), set(TRUNK_OVERRIDES))
            self.assertIn(run["budget"], MATRIX["budgets"])

    def test_unknown_arm_size_or_objective_is_refused_by_name(self):
        from src.train.run_matrix import MatrixError

        for field, value in (("tokenization_arm", "no_such_arm"),
                             ("objective_arm", "no_such_objective"), ("size", "7m"),
                             ("positions", "wall_clock")):
            bad = copy.deepcopy(MATRIX)
            bad["rows"][2][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(MatrixError, value):
                expand(bad)
        bad = copy.deepcopy(MATRIX)
        bad["seeds"] = [1, 2]
        with self.assertRaisesRegex(MatrixError, "three seeds"):
            expand(bad)
        bad = copy.deepcopy(MATRIX)
        del bad["arm_data"]["textcode"]
        with self.assertRaisesRegex(MatrixError, "textcode"):
            expand(bad)

    def test_size_sweep_rows_differ_only_in_trunk_size(self):
        from src.train.run_matrix import launch_commands

        runs = [r for r in expand() if r["tokenization_arm"] == "clinical_soft"
                and r["objective_arm"] == "full" and r["positions"] == "admission_minutes"
                and r["seed"] == 1 and r["budget"] == "full"]
        self.assertEqual(sorted(r["size"] for r in runs), ["10m", "30m", "3m"])
        ignore = {"run_id", "size", "trunk", "table_rows", "claim_bearing"}
        base = {k: v for k, v in runs[0].items() if k not in ignore}
        for run in runs[1:]:
            self.assertEqual({k: v for k, v in run.items() if k not in ignore}, base)
        trunks = [r["trunk"] for r in runs]
        self.assertEqual(len({json.dumps(t, sort_keys=True) for t in trunks}), 3)
        for run in runs:
            self.assertEqual(set(run["trunk"]), {"d_model", "n_layers", "n_heads"})
        # The launch commands differ only in their --trunk values and run directory.
        strip = lambda cmd: re.sub(r"--trunk \S+|--run-dir \S+", "", cmd)
        commands = {strip(launch_commands(r, MATRIX)["torchrun"]) for r in runs}
        self.assertEqual(len(commands), 1)

    def test_order_only_rows_differ_from_the_primary_only_in_positions(self):
        runs = {r["positions"]: r for r in expand()
                if r["tokenization_arm"] == "clinical_soft" and r["objective_arm"] == "full"
                and r["size"] == "30m" and r["seed"] == 2 and r["budget"] == "screening"}
        self.assertEqual(set(runs), {"admission_minutes", "token_index"})
        self.assertEqual(runs["token_index"]["trunk"]["rope_position"], "token_index")

    def test_every_trainable_audit_table_row_maps_to_runs(self):
        runs = expand()
        served = {row for r in runs for row in r["table_rows"]}
        untrained = {row["table_row"] for row in MATRIX["not_trained"]}
        for row in audit_table_rows():
            with self.subTest(row=row):
                self.assertTrue(row in served or row in untrained, row)
        self.assertEqual(untrained & served, set())


class ClaimBearingTest(unittest.TestCase):
    def test_claim_bearing_runs_are_full_budget_reference_runs_listed_by_arm_and_seed(self):
        runs = expand()
        bearing = [r for r in runs if r["claim_bearing"]]
        listed = {(e["tokenization_arm"], e["objective_arm"], s)
                  for e in MATRIX["claim_bearing"] for s in e["seeds"]}
        self.assertEqual({(r["tokenization_arm"], r["objective_arm"], r["seed"])
                          for r in bearing}, listed)
        for run in bearing:
            self.assertEqual(run["budget"], "full")
            self.assertEqual((run["size"], run["positions"]), ("30m", "admission_minutes"))
        # One claim-bearing run per (arm, seed): the claims report refuses a repeat.
        keys = [(r["tokenization_arm"], r["objective_arm"], r["seed"]) for r in bearing]
        self.assertEqual(len(keys), len(set(keys)))

    def test_a_claim_bearing_row_with_a_screening_budget_is_refused(self):
        from src.train.run_matrix import MatrixError

        bad = copy.deepcopy(MATRIX)
        bad["claim_bearing"][0]["budget"] = "screening"
        with self.assertRaisesRegex(MatrixError, "screening"):
            expand(bad)

    def test_a_claim_bearing_entry_that_matches_no_run_is_refused(self):
        from src.train.run_matrix import MatrixError

        bad = copy.deepcopy(MATRIX)
        bad["claim_bearing"][0]["seeds"] = [99]
        with self.assertRaisesRegex(MatrixError, "99"):
            expand(bad)

    def test_attribution_arm_is_off_by_default_and_refused_until_implemented(self):
        from src.train.run_matrix import MatrixError

        self.assertFalse(MATRIX["attribution"]["enabled"])
        self.assertNotIn("deciles_clinical_query",
                         {r["tokenization_arm"] for r in expand()})
        on = copy.deepcopy(MATRIX)
        on["attribution"]["enabled"] = True
        with self.assertRaisesRegex(MatrixError, "deciles_clinical_query"):
            expand(on)


class RunSpecTest(unittest.TestCase):
    def test_every_written_run_spec_passes_the_claims_report_schema(self):
        from src.eval.claims_report import load_run_spec, validate_run_spec
        from src.train.run_matrix import write_run_specs

        runs = expand()
        hashes = {arm: HEX for arm in MATRIX["arm_data"]}
        with tempfile.TemporaryDirectory() as td:
            dirs = write_run_specs(runs, hashes, Path(td))
            self.assertEqual(len(dirs), len(runs))
            for run, run_dir in zip(runs, dirs):
                spec = load_run_spec(run_dir)
                validate_run_spec(spec)
                self.assertEqual(spec["run_id"], run["run_id"])
                self.assertEqual(spec["claim_bearing"], run["claim_bearing"])
                self.assertEqual(spec["vocab_hash"], HEX)
                entry = json.loads((run_dir / "matrix_entry.json").read_text())
                self.assertEqual(entry["size"], run["size"])

    def test_writing_without_an_arm_vocabulary_hash_is_refused(self):
        from src.train.run_matrix import MatrixError, write_run_specs

        with tempfile.TemporaryDirectory() as td, self.assertRaisesRegex(MatrixError,
                                                                         "textcode"):
            write_run_specs(expand(), {"clinical_soft": HEX}, Path(td))


class LaunchCommandTest(unittest.TestCase):
    def test_each_run_has_a_torchrun_and_a_single_process_cpu_command(self):
        from src.train.run_matrix import launch_commands

        run = next(r for r in expand() if r["claim_bearing"])
        cmds = launch_commands(run, MATRIX)
        self.assertTrue(cmds["torchrun"].startswith("torchrun --nproc_per_node=2 -m "
                                                    "src.train.run_tokenization_ablation"))
        self.assertIn("--device cpu", cmds["cpu"])
        self.assertNotIn("torchrun", cmds["cpu"])
        for cmd in cmds.values():
            for flag in (f"--arm {run['tokenization_arm']}",
                         f"--objective-arm {run['objective_arm']}", f"--seed {run['seed']}",
                         f"--passes {MATRIX['budgets'][run['budget']]}",
                         "--trajectory hospitalization", "--site mimic"):
                self.assertIn(flag, cmd)

    def test_each_run_has_a_resume_form_of_its_commands(self):
        from src.train.run_matrix import launch_commands

        run = next(r for r in expand() if r["claim_bearing"])
        cmds = launch_commands(run, MATRIX)
        for key in ("torchrun", "cpu"):
            self.assertEqual(cmds[f"{key}_resume"], cmds[key] + " --resume latest")
            self.assertNotIn("--resume", cmds[key])

    def test_main_prints_every_run_and_starts_no_training(self):
        from src.train import run_matrix

        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(run_matrix.main([]), 0)
        text = out.getvalue()
        self.assertEqual(text.count("torchrun --nproc_per_node=2"), len(expand()))
        self.assertIn("claim-bearing", text)


# ------------------------------------------------------------------ edge distance

def _blob(segments: dict) -> dict:
    from src.data.segments import SPECIAL

    vocab = dict(SPECIAL)
    for concept, segs in segments.items():
        for b in range(len(segs)):
            vocab[f"{concept}={b}"] = len(vocab)
    return {"vocab": vocab, "segments": segments, "manifest": {"tokenizer_version": 2},
            "concept_sources": {"tables": {}, "treatment_sources": []}}


def _segments(edges: dict, forced: bool) -> dict:
    from src.data.segments import segments_from_edges

    data = yaml.safe_load((ROOT / "configs/data.yaml").read_text())
    directions = {t["name"]: t["direction"] for t in data["target_concepts"]}
    pinned = data["value_binning"]["forced_edges"] if forced else {}
    return {c: segments_from_edges(e, pinned.get(c, ()), directions[c])
            for c, e in edges.items()}


class EdgeDistanceTest(unittest.TestCase):
    EDGES = {"map": [55.0, 60.5, 66.0, 72.0, 80.0], "lactate": [1.1, 1.7, 2.2, 3.3, 5.0],
             "spo2": [86.5, 91.5, 94.5, 97.5], "creatinine": [0.8, 1.1, 1.7, 2.6, 4.1]}

    def test_the_table_has_a_row_per_registered_threshold_and_arm(self):
        from src.data.threshold_grid import load_thresholds
        from src.train.run_matrix import edge_distance_table

        vocabs = {"plain": _blob(_segments(self.EDGES, False)),
                  "forced": _blob(_segments(self.EDGES, True))}
        rows = edge_distance_table(vocabs)
        thresholds = load_thresholds()
        n = len(thresholds["decision"]) + len(thresholds["control"])
        self.assertEqual(len([r for r in rows if r["arm"] == "plain"]), n)
        forced_decisions = [r for r in rows if r["arm"] == "forced" and r["kind"] == "decision"]
        self.assertTrue(all(r["on_edge"] for r in forced_decisions))
        for row in rows:
            self.assertEqual(set(row) & {"hosp_id", "patient_id"}, set())

    def test_a_control_on_an_edge_in_one_arm_is_refused_with_the_arm_named(self):
        from src.train.run_matrix import MatrixError, edge_distance_table

        edges = copy.deepcopy(self.EDGES)
        edges["map"] = [55.0, 60.5, 62.5, 72.0, 80.0]        # control MAP 62.5 on an edge
        vocabs = {"clean_arm": _blob(_segments(self.EDGES, False)),
                  "edgy_arm": _blob(_segments(edges, False))}
        with self.assertRaisesRegex(MatrixError, "edgy_arm") as caught:
            edge_distance_table(vocabs)
        self.assertIn("62.5", str(caught.exception))
        self.assertNotIn("clean_arm", str(caught.exception))


class EdgeCheckCommandTest(unittest.TestCase):
    """`run_matrix --edge-check` crashed with a TypeError when no --vocab matched a matrix
    arm (no vocabulary loaded). It now reports that nothing was checked, and refuses an
    unknown arm name."""

    def _main(self, *argv):
        from contextlib import redirect_stderr

        from src.train.run_matrix import main

        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_no_vocabulary_for_any_arm_is_reported_not_a_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            matrix = copy.deepcopy(MATRIX)
            matrix["arm_data"] = {arm: {site: str(Path(tmp) / arm) for site in sites}
                                  for arm, sites in matrix["arm_data"].items()}
            path = Path(tmp) / "matrix.yaml"
            path.write_text(yaml.safe_dump(matrix))
            code, out, err = self._main("--matrix", str(path), "--edge-check")
        self.assertEqual(code, 2)
        self.assertIn("NOT run", err)
        self.assertNotIn("edge check passed", out)

    def test_an_unknown_vocab_arm_is_refused_by_name(self):
        code, _, err = self._main("--edge-check", "--vocab", "no_such_arm=/tmp/vocab.json")
        self.assertEqual(code, 2)
        self.assertIn("no_such_arm", err)

    def test_one_arm_vocabulary_is_checked_and_the_table_prints(self):
        with tempfile.TemporaryDirectory() as tmp:
            arm = next(iter(MATRIX["arm_data"]))
            vocab = Path(tmp) / "vocab.json"
            vocab.write_text(json.dumps(_blob(_segments(EdgeDistanceTest.EDGES, False))))
            code, out, _ = self._main("--edge-check", "--vocab", f"{arm}={vocab}")
        self.assertEqual(code, 0, out)
        self.assertIn("edge check passed", out)
        self.assertIn("control map < 62.5", out)

    def test_format_edge_table_with_no_rows_does_not_crash(self):
        from src.train.run_matrix import format_edge_table

        self.assertIn("threshold", format_edge_table([]))


class OrderOnlyArmTest(unittest.TestCase):
    """R36: `trunk.rope_position: token_index` rotates by event order, not minutes."""

    MCFG = {"trunk": {"d_model": 16, "n_layers": 1, "n_heads": 2, "ffn_mult": 2,
                      "dropout": 0.0, "tied_embeddings": False, "rope_position": "token_index"},
            "heads": {"next_event": {"enabled": True, "weight": 0.2},
                      "competing_risk": {"enabled": True, "weight": 1.0, "n_time_bins": 4},
                      "threshold_hazard": {"enabled": True, "weight": 1.0, "n_time_bins": 4,
                                           "threshold_embed_dim": 4},
                      "value_regression": {"enabled": True, "weight": 0.5}}}

    def test_order_only_trunk_ignores_minutes_and_generation_matches_it(self):
        import torch

        from src.model.generate import KVCache, _cached_forward
        from src.train.pretrain import Model, OrderOnlyEncoder

        torch.manual_seed(0)
        model = Model(30, 2, self.MCFG, n_value_bins=4).eval()
        self.assertIsInstance(model.enc, OrderOnlyEncoder)
        token = torch.randint(4, 30, (1, 6))
        minutes = torch.tensor([[0, 5, 9, 60, 61, 300]])
        with torch.no_grad():
            full = model.enc(token, minutes)
            self.assertTrue(torch.allclose(full, model.enc(token, minutes * 7 + 3)))
            cache = KVCache(1)
            prefix = _cached_forward(model.enc, token[:, :5], minutes[:, :5], cache)
            step = _cached_forward(model.enc, token[:, 5:], minutes[:, 5:], cache)
        self.assertTrue(torch.allclose(prefix, full[:, :5], atol=1e-5))
        self.assertTrue(torch.allclose(step[:, 0], full[:, 5], atol=1e-5))

    def test_unknown_position_mode_is_refused(self):
        from src.train.pretrain import Model

        bad = copy.deepcopy(self.MCFG)
        bad["trunk"]["rope_position"] = "wall_clock"
        with self.assertRaisesRegex(ValueError, "wall_clock"):
            Model(30, 2, bad, n_value_bins=4)


if __name__ == "__main__":
    unittest.main()
