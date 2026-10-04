"""U3 (KTD3): the threshold registry and the per-vocabulary threshold grid.

`configs/thresholds.yaml` registers thresholds as (target concept, value, direction);
`src/data/threshold_grid.py` maps them to (concept index, value bin under ONE
vocabulary's own segments, direction), bound to that vocabulary's hash. All vocabularies
here are hand-built from the physician segment CSV or from synthetic quantile edges.
"""

import copy
import random
import unittest
from pathlib import Path

import yaml

from src.data.segments import (
    SPECIAL,
    ArtifactBindingError,
    artifact_binding,
    bin_index,
    contains,
    load_csv_segments,
    segments_from_edges,
)
from src.data.segments import threshold_bin as threshold_bin_of
from src.data.threshold_grid import (
    DEATH_TOKEN,
    THRESHOLD_KINDS,
    ThresholdGrid,
    ThresholdGridError,
    edge_distance,
    load_thresholds,
    segment_edges,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_CFG = yaml.safe_load((ROOT / "configs/data.yaml").read_text())
COHORT_CFG = yaml.safe_load((ROOT / "configs/cohort.yaml").read_text())
TARGETS = DATA_CFG["target_concepts"]
DIRECTIONS = {c["name"]: c["direction"] for c in TARGETS}
TERMINALS = ("DISCHARGE//home", DEATH_TOKEN)


def vocab_blob(segments: dict, *, tables: dict | None = None,
               treatment_sources=("meds", "resp_support"), extra_tokens=()) -> dict:
    """A tokenizer-v2 vocabulary artifact with one `concept=bin` token per segment."""
    vocab = dict(SPECIAL)
    for concept, segs in segments.items():
        for b in range(len(segs)):
            vocab[f"{concept}={b}"] = len(vocab)
    for token in (*extra_tokens, "ADMISSION//ed", *TERMINALS):
        vocab[token] = len(vocab)
    source = {c["name"]: [c["source"]] for c in TARGETS}
    return {
        "vocab": vocab,
        "segments": segments,
        "manifest": {"tokenizer_version": 2},
        "concept_sources": {"tables": {**source, **(tables or {})},
                            "treatment_sources": sorted(treatment_sources)},
    }


def clinical_segments() -> dict:
    """The primary arm: physician segments for the ten target concepts, forced edges in."""
    binning = DATA_CFG["value_binning"]
    return load_csv_segments(ROOT / binning["segment_source"], list(DIRECTIONS),
                             binning["forced_edges"], DIRECTIONS)


def decile_segments(*, forced: bool) -> dict:
    """A decile-style arm for MAP and lactate: quantile edges that are NOT the clinical
    ones; `forced` pins the decision thresholds as extra edges (KTD11's second arm)."""
    forced_edges = DATA_CFG["value_binning"]["forced_edges"] if forced else {}
    return {
        "map": segments_from_edges([58.0, 62.0, 68.0, 74.0, 81.0, 90.0],
                                   forced_edges.get("map", ()), "below"),
        "lactate": segments_from_edges([0.8, 1.1, 1.7, 2.6, 4.4],
                                       forced_edges.get("lactate", ()), "above"),
    }


class ThresholdRegistryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = load_thresholds()

    def test_every_decision_threshold_is_a_forced_edge_and_every_forced_edge_is_registered(self):
        forced = {(concept, float(value))
                  for concept, values in DATA_CFG["value_binning"]["forced_edges"].items()
                  for value in values}
        decision = {(t.concept, t.value) for t in self.registry["decision"]}
        self.assertEqual(decision, forced)
        self.assertEqual(len(self.registry["decision"]), len(decision), "duplicate entry")

    def test_every_threshold_uses_its_target_concepts_direction(self):
        for kind in THRESHOLD_KINDS:
            for threshold in self.registry[kind]:
                with self.subTest(kind=kind, concept=threshold.concept):
                    self.assertEqual(threshold.direction, DIRECTIONS[threshold.concept])

    def test_each_target_concept_has_exactly_one_competing_risk_cause(self):
        causes = [t.concept for t in self.registry["competing_risk_cause"]]
        self.assertEqual(sorted(causes), sorted(DIRECTIONS))
        self.assertEqual(len(causes), len(set(causes)))

    def test_contract_causes_repeat_the_outcome_contract_and_the_rest_are_confirmed(self):
        """Product authority, 2026-10-03 (docs/decisions/2026-10-03-clinical-decisions.md
        items 1-7): no cause is left `proposed`."""
        contract = {(spec["concept"], float(spec["threshold"]), spec["direction"])
                    for spec in COHORT_CFG["outcomes"].values()}
        by_status = {"contract": set(), "confirmed": set(), "proposed": set()}
        for t in self.registry["competing_risk_cause"]:
            by_status[t.status].add((t.concept, t.value, t.direction))
        self.assertEqual(by_status["contract"], contract)
        self.assertEqual(by_status["proposed"], set())
        self.assertEqual(len(by_status["confirmed"]), len(DIRECTIONS) - len(contract))

    def test_confirmed_cause_values_are_the_product_authoritys(self):
        causes = {t.concept: t for t in self.registry["competing_risk_cause"]}
        expected = {"respiratory_rate": 24.0, "bilirubin_total": 2.0, "platelet_count": 100.0,
                    "heart_rate": 130.0, "sbp": 90.0, "temp_c": 39.166}
        self.assertEqual({c: causes[c].value for c in expected}, expected)
        creatinine = causes["creatinine"]
        self.assertEqual((creatinine.rule, creatinine.value), ("kdigo_aki", None))
        self.assertEqual(creatinine.params, {"absolute_rise": 0.3, "absolute_window_hours": 48.0,
                                             "relative_rise": 1.5, "relative_window_hours": 168.0})

    def test_temperature_cause_is_a_physician_csv_edge_and_the_top_bin(self):
        """Item 7 (final): above 39.166 C, exactly a physician-CSV temp_c edge, so the
        event is the top bin (39.166, 44.0] and readable from the token stream."""
        cause = next(t for t in self.registry["competing_risk_cause"] if t.concept == "temp_c")
        segs = clinical_segments()["temp_c"]
        self.assertEqual(cause.value, 39.166)
        self.assertIn(cause.value, segment_edges(segs))
        top = segs[threshold_bin_of(cause.value, segs, cause.direction)]
        self.assertEqual((top["lo"], top["hi"]), (39.166, segs[-1]["hi"]))

    def test_controls_are_the_provisional_off_edge_values(self):
        controls = {(t.concept, t.paired_decision): t.value for t in self.registry["control"]}
        self.assertEqual(controls[("map", 65.0)], 62.5)
        self.assertEqual(controls[("creatinine", 1.5)], 1.35)

    def test_controls_are_candidates_paired_with_a_decision_threshold_of_their_concept(self):
        decision = {(t.concept, t.value) for t in self.registry["decision"]}
        self.assertTrue(self.registry["control"])
        for control in self.registry["control"]:
            with self.subTest(concept=control.concept, value=control.value):
                self.assertEqual(control.status, "candidate")
                self.assertIn((control.concept, control.paired_decision), decision)
                self.assertNotIn((control.concept, control.value), decision)

    def test_control_candidates_are_off_edge_in_the_physician_segments(self):
        """MAP 60 and 61 are physician edges, so the MAP control is 62.5; SpO2 89 sits
        inside [88, 90)."""
        rows = {(r["kind"], r["concept"], r["value"]): r
                for r in edge_distance(self.registry, vocab_blob(clinical_segments()))}
        for control in self.registry["control"]:
            row = rows[("control", control.concept, control.value)]
            with self.subTest(concept=control.concept, value=control.value):
                self.assertFalse(row["on_edge"])
                self.assertGreater(row["distance"], 0.0)
        self.assertEqual(rows[("control", "map", 62.5)]["distance"], 1.5)
        self.assertEqual(rows[("control", "map", 62.5)]["nearest_edge"], 61.0)
        edges = segment_edges(clinical_segments()["map"])
        self.assertTrue({59.0, 60.0, 61.0, 65.0, 67.0} <= set(edges))
        self.assertNotIn(62.5, edges)

    def test_label_rule_repeats_the_outcome_contracts_windows(self):
        rule = self.registry["label_rule"]
        for spec in COHORT_CFG["outcomes"].values():
            self.assertEqual(rule["baseline_lookback_hours"], spec["baseline_lookback_hours"])
            self.assertEqual(rule["required_measurement_within_hours_of_horizon"],
                             spec["required_measurement_within_hours_of_horizon"])
        self.assertEqual(rule["horizon_hours"],
                         COHORT_CFG["windows"]["prediction"]["horizon_hours"])

    def test_malformed_registries_are_refused(self):
        raw = yaml.safe_load((ROOT / "configs/thresholds.yaml").read_text())

        def refused(mutate, pattern):
            broken = copy.deepcopy(raw)
            mutate(broken)
            path = Path(self.enterContext(_tempdir())) / "thresholds.yaml"
            path.write_text(yaml.safe_dump(broken))
            with self.assertRaisesRegex(ThresholdGridError, pattern):
                load_thresholds(path)

        refused(lambda r: r.pop("decision"), "decision")
        refused(lambda r: r["decision"][0].update(direction="sideways"), "direction")
        refused(lambda r: r["decision"][0].update(value=float("nan")), "finite")
        refused(lambda r: r["control"][0].update(paired_decision=9.9), "paired")
        refused(lambda r: r["control"][0].update(status="registered"), "status")
        refused(lambda r: r["competing_risk_cause"].append(
            dict(r["competing_risk_cause"][0], value=60.0)), "one competing-risk cause")
        refused(lambda r: r["decision"].append(dict(r["decision"][0])), "repeats")
        refused(lambda r: r["label_rule"].update(horizon_hours=0), "label_rule")
        cr = next(i for i, e in enumerate(raw["competing_risk_cause"])
                  if e["concept"] == "creatinine")
        refused(lambda r: r["competing_risk_cause"][cr].update(rule="rifle"), "unknown rule")
        refused(lambda r: r["competing_risk_cause"][cr].update(value=2.0), "no `value`")
        refused(lambda r: r["competing_risk_cause"][cr].pop("absolute_rise"), "absolute_rise")
        refused(lambda r: r["competing_risk_cause"][cr].update(relative_rise=0.9), "exceed 1")
        refused(lambda r: r["decision"][0].update(rule="kdigo_aki"), "only a competing-risk")
        refused(lambda r: r.pop("units"), "units")
        refused(lambda r: r["units"].pop("lactate"), "no CLIF 2.1 unit")
        refused(lambda r: r["label_rule"].pop("required_measurement_within_hours_of_horizon"),
                "label_rule")


def _tempdir():
    import tempfile

    return tempfile.TemporaryDirectory()


class ClinicalGridTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = load_thresholds()
        cls.segments = clinical_segments()
        cls.blob = vocab_blob(cls.segments)
        cls.grid = ThresholdGrid(cls.blob, TARGETS, cls.registry)

    def test_grid_is_bound_to_the_vocabulary_hash(self):
        self.assertEqual(self.grid.binding, artifact_binding(self.blob))
        self.grid.check_binding(artifact_binding(self.blob), what="shard")
        other = vocab_blob(decile_segments(forced=False))
        with self.assertRaisesRegex(ArtifactBindingError, "mismatch"):
            self.grid.check_binding(artifact_binding(other), what="shard")
        with self.assertRaises(ArtifactBindingError):
            self.grid.check_binding(None, what="shard")

    def test_a_pre_v2_vocabulary_is_refused(self):
        with self.assertRaises(ArtifactBindingError):
            ThresholdGrid({"vocab": {}, "edges": {}}, TARGETS, self.registry)

    def test_concept_index_is_the_target_concept_order(self):
        self.assertEqual(self.grid.concepts, tuple(c["name"] for c in TARGETS))
        self.assertEqual(self.grid.query("map", 65.0).target_idx, 0)
        self.assertEqual(self.grid.query("temp_c", 39.166).target_idx, 9)

    def test_decision_thresholds_map_to_the_bin_on_their_event_side(self):
        """An on-edge threshold names the bin adjacent to the edge on the event side —
        the 24 h outcome join's rule (`segments.threshold_bin`), via `bin_index`."""
        below = self.grid.query("map", 65.0)
        self.assertEqual((below.direction, below.direction_id), ("below", 0))
        self.assertEqual(below.threshold_bin, bin_index(64.9, self.segments["map"]))
        self.assertEqual(self.segments["map"][below.threshold_bin]["hi"], 65.0)
        above = self.grid.query("lactate", 4.0)
        self.assertEqual((above.direction, above.direction_id), ("above", 1))
        self.assertEqual(above.threshold_bin, bin_index(4.01, self.segments["lactate"]))
        self.assertEqual(self.segments["lactate"][above.threshold_bin]["lo"], 4.0)

    def test_an_off_edge_threshold_maps_to_the_bin_that_contains_it(self):
        control = self.grid.query("map", 63.0)
        self.assertTrue(contains(self.segments["map"][control.threshold_bin], 63.0))
        self.assertEqual(control.threshold_bin, bin_index(63.0, self.segments["map"]))
        # ... which is the bin the MAP < 65 decision threshold names: the model cannot
        # tell the two queries apart, which is what the off-edge control measures.
        self.assertEqual(control.threshold_bin, self.grid.query("map", 65.0).threshold_bin)
        self.assertEqual(control.value, 63.0)   # labels still use the exact value

    def test_registered_thresholds_are_mapped_per_kind(self):
        for kind in THRESHOLD_KINDS:
            mapped = self.grid.registered(kind)
            self.assertEqual([(q.concept, q.value, q.direction) for q in mapped],
                             [(t.concept, t.value, t.direction) for t in self.registry[kind]])
        self.assertEqual(sorted(self.grid.causes), list(range(len(TARGETS))))
        self.assertEqual(self.grid.causes[0].value, 65.0)

    def test_every_decision_threshold_is_one_of_the_arms_own_sampling_edges(self):
        for query in self.grid.registered("decision"):
            pool = {(q.value, q.threshold_bin) for q in self.grid.sampling_edges(query.concept)}
            with self.subTest(concept=query.concept, value=query.value):
                self.assertIn((query.value, query.threshold_bin), pool)

    def test_sampling_edges_are_the_arms_own_interior_edges_one_per_bin(self):
        for concept in self.grid.concepts:
            segs = self.segments[concept]
            edges = segment_edges(segs)
            pool = self.grid.sampling_edges(concept)
            with self.subTest(concept=concept):
                self.assertTrue(pool)
                self.assertTrue({q.value for q in pool} <= set(edges))
                # Never the outer floor or ceiling: nothing is beyond them.
                self.assertNotIn(segs[0]["lo"], {q.value for q in pool})
                self.assertNotIn(segs[-1]["hi"], {q.value for q in pool})
                # One threshold per (concept, bin, direction): two label definitions
                # never share one query embedding.
                bins = [q.threshold_bin for q in pool]
                self.assertEqual(len(bins), len(set(bins)))
                self.assertTrue(all(q.direction == DIRECTIONS[concept] for q in pool))
                self.assertTrue(all(0 <= b < len(segs) for b in bins))

    def test_sampler_is_deterministic_and_only_draws_target_concepts_own_edges(self):
        first = [self.grid.sample(random.Random(11)) for _ in range(3)]
        self.assertEqual(first, [self.grid.sample(random.Random(11)) for _ in range(3)])
        rng = random.Random(5)
        seen = set()
        for _ in range(400):
            query = self.grid.sample(rng)
            self.assertIn(query, self.grid.sampling_edges(query.concept))
            seen.add(query.concept)
        self.assertEqual(seen, set(DIRECTIONS))

    def test_token_map_covers_target_measurement_tokens_only(self):
        vocab = self.blob["vocab"]
        self.assertEqual(self.grid.token_target[vocab["map=0"]], 0)
        self.assertEqual(self.grid.token_target[vocab["lactate=3"]], 1)
        self.assertEqual(len(self.grid.token_target),
                         sum(len(self.segments[c]) for c in self.grid.concepts))
        self.assertEqual(self.grid.death_token, vocab[DEATH_TOKEN])
        self.assertEqual(self.grid.terminal_tokens, {vocab[t] for t in TERMINALS})
        self.assertEqual(self.grid.death_cause, len(TARGETS))

    def test_edge_distance_reports_on_edge_and_distance_for_every_registered_threshold(self):
        rows = self.grid.edge_distance()
        # A rule cause (KDIGO creatinine) has no value and so no edge-distance row.
        valued = sum(t.rule is None for k in THRESHOLD_KINDS for t in self.registry[k])
        self.assertEqual(len(rows), valued)
        self.assertNotIn(("competing_risk_cause", "creatinine"),
                         {(r["kind"], r["concept"]) for r in rows})
        for row in rows:
            self.assertEqual(row["vocabulary"], self.grid.binding["vocabulary"])
            self.assertEqual(row["numeric_edges"], self.grid.binding["numeric_edges"])
            if row["kind"] == "decision":
                self.assertTrue(row["on_edge"], row)
                self.assertEqual(row["distance"], 0.0)
                self.assertEqual(row["nearest_edge"], row["value"])
        self.assertEqual(rows, edge_distance(self.registry, self.blob))


class RefusalTest(unittest.TestCase):
    """Hard rule 1: a treatment / input-only concept is never a threshold target."""

    @classmethod
    def setUpClass(cls):
        cls.registry = load_thresholds()
        cls.segments = clinical_segments()

    def test_a_concept_outside_the_target_map_is_refused(self):
        segments = dict(self.segments,
                        norepinephrine_mcg_kg_min=segments_from_edges([0.05, 0.1, 0.3]))
        grid = ThresholdGrid(vocab_blob(segments, tables={"norepinephrine_mcg_kg_min": ["meds"]}),
                             TARGETS, self.registry)
        with self.assertRaisesRegex(ThresholdGridError, "not a target concept"):
            grid.query("norepinephrine_mcg_kg_min", 0.1, "above")
        self.assertNotIn("norepinephrine_mcg_kg_min", grid.concepts)
        vocab = vocab_blob(segments)["vocab"]
        self.assertNotIn(vocab["norepinephrine_mcg_kg_min=0"], grid.token_target)

    def test_a_registered_threshold_on_an_input_only_concept_is_refused(self):
        registry = dict(self.registry)
        registry["decision"] = (*registry["decision"],
                                registry["decision"][0].__class__(
                                    "decision", "norepinephrine_mcg_kg_min", 0.1, "above"))
        with self.assertRaisesRegex(ThresholdGridError, "not a target concept"):
            ThresholdGrid(vocab_blob(self.segments), TARGETS, registry)

    def test_a_target_concept_charted_by_a_treatment_table_is_refused(self):
        """Even a concept listed under `target_concepts` is refused when the vocabulary
        says a treatment (input-only) table charts it."""
        targets = [*TARGETS, {"name": "fio2_set", "source": "resp_support",
                              "direction": "above", "unit": "fraction"}]
        segments = dict(self.segments, fio2_set=segments_from_edges([0.3, 0.4, 0.6]))
        blob = vocab_blob(segments, tables={"fio2_set": ["resp_support"]})
        with self.assertRaisesRegex(ThresholdGridError, "input-only"):
            ThresholdGrid(blob, targets, self.registry)

    def test_a_direction_other_than_the_target_concepts_is_refused(self):
        grid = ThresholdGrid(vocab_blob(self.segments), TARGETS, self.registry)
        with self.assertRaisesRegex(ThresholdGridError, "direction"):
            grid.query("map", 65.0, "above")

    def test_a_target_concept_this_vocabulary_does_not_bin_cannot_be_queried(self):
        blob = vocab_blob({"map": self.segments["map"]})
        registry = {**self.registry,
                    **{k: tuple(t for t in self.registry[k] if t.concept == "map")
                       for k in THRESHOLD_KINDS}}
        grid = ThresholdGrid(blob, TARGETS, registry)
        self.assertEqual(grid.binned, ("map",))
        with self.assertRaisesRegex(ThresholdGridError, "no segments"):
            grid.query("lactate", 4.0)
        # ... and a registry naming it fails closed rather than dropping the threshold.
        with self.assertRaisesRegex(ThresholdGridError, "no segments"):
            ThresholdGrid(blob, TARGETS, self.registry)

    def test_unknown_tau_sampling_is_refused(self):
        with self.assertRaisesRegex(ThresholdGridError, "tau_sampling"):
            ThresholdGrid(vocab_blob(self.segments), TARGETS, self.registry,
                          tau_sampling="uniform_values")

    def test_non_finite_threshold_is_refused(self):
        grid = ThresholdGrid(vocab_blob(self.segments), TARGETS, self.registry)
        with self.assertRaisesRegex(ThresholdGridError, "finite"):
            grid.query("map", float("inf"))


class DecileArmGridTest(unittest.TestCase):
    """A decile-arm vocabulary yields a grid on ITS OWN edges."""

    @classmethod
    def setUpClass(cls):
        full = load_thresholds()
        cls.registry = {**full, **{k: tuple(t for t in full[k]
                                            if t.concept in ("map", "lactate"))
                                   for k in THRESHOLD_KINDS}}
        cls.plain = ThresholdGrid(vocab_blob(decile_segments(forced=False)), TARGETS,
                                  cls.registry)
        cls.forced = ThresholdGrid(vocab_blob(decile_segments(forced=True)), TARGETS,
                                   cls.registry)

    def test_sampling_edges_are_the_decile_edges(self):
        self.assertEqual([q.value for q in self.plain.sampling_edges("map")],
                         [58.0, 62.0, 68.0, 74.0, 81.0, 90.0])
        self.assertEqual([q.value for q in self.plain.sampling_edges("lactate")],
                         [0.8, 1.1, 1.7, 2.6, 4.4])
        self.assertEqual(self.plain.binned, ("map", "lactate"))
        self.assertNotIn(65.0, [q.value for q in self.plain.sampling_edges("map")])

    def test_a_decision_threshold_maps_to_the_bin_that_contains_it(self):
        segments = decile_segments(forced=False)
        for concept, value in (("map", 65.0), ("lactate", 2.0), ("lactate", 4.0)):
            query = self.plain.query(concept, value)
            with self.subTest(concept=concept, value=value):
                self.assertTrue(contains(segments[concept][query.threshold_bin], value))
                self.assertEqual(query.threshold_bin, bin_index(value, segments[concept]))

    def test_the_same_threshold_names_a_different_bin_in_each_arm(self):
        clinical = ThresholdGrid(vocab_blob(clinical_segments()), TARGETS, load_thresholds())
        self.assertNotEqual(clinical.query("map", 65.0).threshold_bin,
                            self.plain.query("map", 65.0).threshold_bin)
        self.assertNotEqual(clinical.binding, self.plain.binding)

    def test_edge_distance_separates_the_plain_and_forced_edge_decile_arms(self):
        def row(grid, concept, value):
            return next(r for r in grid.edge_distance()
                        if (r["kind"], r["concept"], r["value"]) == ("decision", concept, value))

        plain = row(self.plain, "map", 65.0)
        self.assertFalse(plain["on_edge"])
        self.assertEqual((plain["nearest_edge"], plain["distance"]), (62.0, 3.0))
        forced = row(self.forced, "map", 65.0)
        self.assertTrue(forced["on_edge"])
        self.assertEqual(forced["distance"], 0.0)
        self.assertIn(65.0, [q.value for q in self.forced.sampling_edges("map")])

    def test_a_control_can_be_on_edge_in_a_decile_arm(self):
        """Why controls are only candidates: a quantile edge can land on one."""
        segments = decile_segments(forced=False)
        segments["map"] = segments_from_edges([58.0, 62.5, 68.0], (), "below")
        rows = edge_distance(self.registry, vocab_blob(segments))
        control = next(r for r in rows if (r["kind"], r["concept"]) == ("control", "map"))
        self.assertTrue(control["on_edge"])

    def test_edge_distance_marks_a_concept_the_vocabulary_does_not_bin(self):
        rows = edge_distance(load_thresholds(), vocab_blob(decile_segments(forced=False)))
        spo2 = next(r for r in rows if r["concept"] == "spo2")
        self.assertFalse(spo2["binned"])
        self.assertIsNone(spo2["on_edge"])
        self.assertIsNone(spo2["distance"])


class ClifUnitsTest(unittest.TestCase):
    """Product authority (2026-10-03): every threshold is in the CLIF 2.1 unit of its
    concept and every concept is a CLIF 2.1 category. Read from the mCIDE 2.1.1 snapshot
    (configs/clif_mcide_2.1.1/, else output/final_no_phi/clif_spec_v2.1.1/)."""

    @classmethod
    def setUpClass(cls):
        candidates = [ROOT / "configs/clif_mcide_2.1.1", ROOT / "output/final_no_phi/clif_spec_v2.1.1"]
        cls.snapshot = next((p for p in candidates
                             if (p / "mCIDE/labs/clif_lab_categories.csv").is_file()), None)
        if cls.snapshot is None:
            raise unittest.SkipTest("no CLIF 2.1.1 mCIDE snapshot in the repository")
        import csv

        with open(cls.snapshot / "mCIDE/labs/clif_lab_categories.csv", encoding="utf-8") as fh:
            cls.labs = {row["lab_category"]: row["reference_unit"] for row in csv.DictReader(fh)}
        with open(cls.snapshot / "mCIDE/vitals/clif_vitals_categories.csv", encoding="utf-8") as fh:
            cls.vitals = {row["vital_category"] for row in csv.DictReader(fh) if row["vital_category"]}
        ddl = next((cls.snapshot / "ddl").rglob("*.sql"))
        line = next(l for l in ddl.read_text(encoding="utf-8").splitlines()
                    if l.strip().startswith("vital_value"))
        permissible = line.split('"permissible": "', 1)[1].split('"', 1)[0]
        declared, _, no_unit = permissible.partition("No unit for ")
        cls.vital_units = {k.strip(): v.strip() for k, v in
                           (part.split("=") for part in declared.strip(" .").split(",") if "=" in part)}
        cls.vital_no_unit = {name.strip() for name in no_unit.split(",") if name.strip()}
        cls.registry = load_thresholds()

    @staticmethod
    def _norm(unit: str) -> str:
        return unit.replace("µ", "u").replace("μ", "u").strip().lower()

    def test_every_threshold_concept_is_a_clif_lab_or_vital_category(self):
        for kind in THRESHOLD_KINDS:
            for t in self.registry[kind]:
                with self.subTest(kind=kind, concept=t.concept):
                    self.assertIn(t.concept, set(self.labs) | self.vitals)

    def test_every_threshold_unit_is_the_clif_2_1_unit(self):
        units = self.registry["units"]
        for concept, spec in units.items():
            with self.subTest(concept=concept):
                if concept in self.labs:
                    self.assertTrue(spec["clif_declared"])
                    self.assertEqual(self._norm(spec["unit"]), self._norm(self.labs[concept]))
                elif concept in self.vital_units:
                    self.assertTrue(spec["clif_declared"])
                    self.assertEqual(self._norm(spec["unit"]), self._norm(self.vital_units[concept]))
                else:
                    # CLIF 2.1 states "no unit" for this vital: the value is used as charted.
                    self.assertIn(concept, self.vital_no_unit)
                    self.assertFalse(spec["clif_declared"])

    def test_registry_units_agree_with_the_target_concept_contract(self):
        declared = {c["name"]: c.get("unit") for c in TARGETS}
        aliases = {"celsius": "celsius", "breaths per minute": "breaths/min",
                   "beats per minute": "beats/min"}
        for concept, spec in self.registry["units"].items():
            with self.subTest(concept=concept):
                want = self._norm(declared[concept])
                self.assertEqual(self._norm(spec["unit"]), aliases.get(want, want))


if __name__ == "__main__":
    unittest.main()
