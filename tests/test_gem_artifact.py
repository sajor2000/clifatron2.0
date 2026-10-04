"""U8 (R17, R18; KTD10): the full-hospitalization GEM artifact with terminal tokens.

A synthetic site is tokenized twice from the same input: the 24 h prediction artifact
(`trajectory="icu_24h"`, the default) and the GEM artifact
(`trajectory="hospitalization"`, the SAME frozen vocabulary). The GEM stream runs from
hospital admission to discharge, framed `<bos> ADMISSION//x ... DISCHARGE//y <eos>`, with
long stays split into continuation windows. All data is synthetic.
"""

import copy
import hashlib
import json
import os
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

import polars as pl
import yaml

MAX_TOKENS = 64            # small window so the 20k-event stay splits fast in tests
LONG_STAY = "synth-005"
LONG_EVENTS = 20_000
STATIC = ["age_decile", "admission_type"]
DISPOSITIONS = {           # raw discharge_category per stay; the rest are Home
    "synth-000": "Expired",
    "synth-001": "Missing",
    "synth-002": "Skilled Nursing Facility (SNF)",
    "synth-003": None,
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reseal(hosp: pl.DataFrame, adt: pl.DataFrame) -> pl.DataFrame:
    """Rebuild the canonical episode artifact like `build_synthetic_site` does."""
    from src.data.cohort import build_cohort
    from src.data.splits import content_manifest
    from src.eval.synthetic_bundle import FIXTURE_COHORT

    episodes = build_cohort(hosp, adt, {
        "anchor_hours": 24, "prediction_horizon_hours": 48, "minimum_age": 18,
        "icu_location_category": "icu",
    }).with_columns(pl.lit("train").alias("partition"))
    split_hash = content_manifest(
        episodes, columns=["hospitalization_id", "patient_id", "partition"])["sha256"]
    episode_hash = content_manifest(
        episodes, columns=["hospitalization_id", "patient_id", "eligible", "partition"]
    )["sha256"]
    return episodes.with_columns(
        pl.lit(FIXTURE_COHORT["contract_version"]).alias("cohort_contract_version"),
        pl.lit(split_hash).alias("split_sha256"),
        pl.lit(episode_hash).alias("episode_sha256"),
        pl.lit("{}").alias("source_provenance_json"),
    )


def _build_site(work: Path):
    """Synthetic site: synth-000 has an ED phase (hospital admission 6 h before ICU), a
    post-ICU ward phase, and events before admission / after discharge that must be
    excluded; synth-005 carries 20k MAP rows. Returns (site dir, episodes, cfg)."""
    from src.data.tokenize import ROOT
    from src.eval.synthetic_bundle import (
        FIXTURE_COHORT,
        FIXTURE_DATA_CONFIG,
        FIXTURE_POLICY,
        build_synthetic_site,
    )

    site = work / "site"
    build_synthetic_site(site)
    hosp = pl.read_parquet(site / "clif_hospitalization.parquet")
    hosp = hosp.with_columns(
        pl.when(pl.col("hospitalization_id") == "synth-000")
        .then(pl.col("admission_dttm") - pl.duration(hours=6))
        .otherwise(pl.col("admission_dttm")).alias("admission_dttm"),
        pl.col("hospitalization_id").replace_strict(
            DISPOSITIONS, default="Home", return_dtype=pl.String).alias("discharge_category"),
        pl.when(pl.col("hospitalization_id") == "synth-001").then(pl.lit("Elective"))
        .otherwise(pl.lit("ed")).alias("admission_type_category"),
    )
    hosp.write_parquet(site / "clif_hospitalization.parquet")
    row0 = hosp.filter(pl.col("hospitalization_id") == "synth-000").row(0, named=True)
    adt = pl.read_parquet(site / "clif_adt.parquet")
    icu0 = adt.filter(pl.col("hospitalization_id") == "synth-000").row(0, named=True)
    extra_adt = pl.DataFrame({
        "hospitalization_id": ["synth-000", "synth-000"],
        "in_dttm": [row0["admission_dttm"], icu0["out_dttm"]],
        "out_dttm": [icu0["in_dttm"], row0["discharge_dttm"]],
        "location_category": ["ed", "ward"],
        "hospital_id": [icu0["hospital_id"]] * 2,
    }, schema=adt.schema)
    adt = pl.concat([adt, extra_adt])
    adt.write_parquet(site / "clif_adt.parquet")

    vitals = pl.read_parquet(site / "clif_vitals.parquet")
    icu_in = icu0["in_dttm"]
    extra = [
        ("synth-000", row0["admission_dttm"] - timedelta(hours=1), 71.25),   # before admit
        ("synth-000", row0["admission_dttm"] + timedelta(hours=3), 72.25),   # ED, pre-ICU
        ("synth-000", icu_in + timedelta(hours=100), 73.25),                 # ward, post-ICU
        ("synth-000", row0["discharge_dttm"] + timedelta(hours=1), 74.25),   # after discharge
    ]
    long_in = (hosp.filter(pl.col("hospitalization_id") == LONG_STAY)
               .row(0, named=True)["admission_dttm"])
    extra += [(LONG_STAY, long_in + timedelta(seconds=20 * k + 30), 60.0 + (k % 40))
              for k in range(LONG_EVENTS)]
    vitals = pl.concat([vitals, pl.DataFrame(
        {"hospitalization_id": [e[0] for e in extra],
         "recorded_dttm": [e[1] for e in extra],
         "vital_category": ["map"] * len(extra),
         "vital_value": [e[2] for e in extra],
         "vital_unit": ["mmHg"] * len(extra)}, schema=vitals.schema)])
    vitals.write_parquet(site / "clif_vitals.parquet")
    episodes = _reseal(hosp, adt)

    (work / "cohort.yaml").write_text(yaml.safe_dump(FIXTURE_COHORT))
    (work / "artifact_policy.yaml").write_text(yaml.safe_dump(FIXTURE_POLICY))
    data_cfg = yaml.safe_load((ROOT / "configs/data.yaml").read_text())
    cfg = copy.deepcopy(FIXTURE_DATA_CONFIG)
    cfg["cohort_contract"] = str((work / "cohort.yaml").resolve())
    cfg["artifact_policy"] = str((work / "artifact_policy.yaml").resolve())
    cfg["tables"]["adt"] = copy.deepcopy(data_cfg["tables"]["adt"])
    cfg["static_tokens"] = list(STATIC)
    cfg["static_source"] = copy.deepcopy(data_cfg["static_source"])
    cfg["gem"] = copy.deepcopy(data_cfg["gem"])
    return site, episodes, cfg


class GemArtifactTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.data.tokenize import tokenize_site
        from src.eval.synthetic_bundle import FIXTURE_POLICY, SYNTHETIC_SITE

        cls._td = tempfile.TemporaryDirectory()
        work = Path(cls._td.name)
        old_cwd = os.getcwd()
        os.chdir(work)
        try:
            site, episodes, cfg = _build_site(work)
            cls.cfg, cls.episodes = cfg, episodes
            kw = {"episodes": episodes, "artifact_policy": FIXTURE_POLICY}
            # Run A: the 24 h artifact alone.
            only = work / "output/intermediate_phi/only_24h"
            tokenize_site(cfg, SYNTHETIC_SITE, site, only, None, **kw)
            # Run B: the 24 h artifact, then GEM beside it with the same frozen vocab.
            both = work / "output/intermediate_phi/with_gem"
            tokenize_site(cfg, SYNTHETIC_SITE, site, both, None, **kw)
            cls.events_sha_before_gem = _sha(both / "events.parquet")
            blob = json.loads((both / "vocab.json").read_text())
            tokenize_site(cfg, SYNTHETIC_SITE, site, both, blob, trajectory="hospitalization",
                          max_tokens=MAX_TOKENS, **kw)
            # Run C: no `gem` block -> no terminal allowlist in the vocab.
            nogem_cfg = {k: v for k, v in cfg.items() if k != "gem"}
            nogem = work / "output/intermediate_phi/no_gem_block"
            tokenize_site(nogem_cfg, SYNTHETIC_SITE, site, nogem, None, **kw)

            cls.only_sha = _sha(only / "events.parquet")
            cls.both_sha = _sha(both / "events.parquet")
            cls.events = pl.read_parquet(both / "events.parquet")
            cls.nogem_events = pl.read_parquet(nogem / "events.parquet")
            cls.nogem_vocab = json.loads((nogem / "vocab.json").read_text())["vocab"]
            cls.gem = pl.read_parquet(both / "gem_events.parquet")
            cls.blob = blob
            cls.site, cls.both, cls.work, cls.kw = site, both, work, kw
            cls.gem_path = both / "gem_events.parquet"
        finally:
            os.chdir(old_cwd)
        cls.vocab = cls.blob["vocab"]
        cls.inv = {i: t for t, i in cls.vocab.items()}

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    # --- helpers -------------------------------------------------------------------
    def _windows(self, stay):
        return (self.gem.filter(pl.col("hosp_id") == stay)
                .sort("continuation_index").to_dicts())

    def _stay(self, stay):
        """The stay's full GEM stream, windows concatenated in order."""
        rows = self._windows(stay)
        self.assertTrue(rows, stay)
        out = {k: [] for k in ("token", "pos_min", "value", "target_eligible")}
        for row in rows:
            for k in out:
                out[k] += list(row[k])
        out["tokens"] = [self.inv[t] for t in out["token"]]
        out["anchor_idx"], out["anchor_min"] = rows[0]["anchor_idx"], rows[0]["anchor_min"]
        return out

    def _minutes(self, stay, column):
        ep = self.episodes.filter(pl.col("hospitalization_id") == stay).row(0, named=True)
        return int((ep[column] - ep["admission_dttm"]).total_seconds() // 60)

    # --- framing and dispositions ----------------------------------------------------
    def test_expired_stay_ends_with_discharge_expired_then_eos_at_discharge_position(self):
        stay = self._stay("synth-000")
        self.assertEqual(stay["tokens"][-2:], ["DISCHARGE//expired", "<eos>"])
        discharge_min = self._minutes("synth-000", "discharge_dttm")
        self.assertEqual(stay["pos_min"][-2:], [discharge_min, discharge_min])
        self.assertEqual(stay["tokens"][:2], ["<bos>", "ADMISSION//ed"])
        self.assertEqual(stay["pos_min"][:2], [0, 0])

    def test_missing_and_null_disposition_map_to_unknown(self):
        self.assertEqual(self._stay("synth-001")["tokens"][-2], "DISCHARGE//unknown")
        self.assertEqual(self._stay("synth-003")["tokens"][-2], "DISCHARGE//unknown")

    def test_snf_maps_to_facility_and_home_to_home(self):
        self.assertEqual(self._stay("synth-002")["tokens"][-2], "DISCHARGE//facility")
        self.assertEqual(self._stay("synth-004")["tokens"][-2], "DISCHARGE//home")

    def test_admission_type_is_normalized(self):
        self.assertEqual(self._stay("synth-001")["tokens"][1], "ADMISSION//elective")

    def test_terminal_and_framing_tokens_appear_only_at_their_slots(self):
        for stay in self.gem["hosp_id"].unique().to_list():
            tokens = self._stay(stay)["tokens"]
            terminal = [i for i, t in enumerate(tokens)
                        if t.startswith("DISCHARGE//") or t == "<eos>"]
            self.assertEqual(terminal, [len(tokens) - 2, len(tokens) - 1], stay)
            self.assertEqual([i for i, t in enumerate(tokens) if t.startswith("ADMISSION//")],
                             [1], stay)
            self.assertEqual([i for i, t in enumerate(tokens) if t == "<bos>"], [0], stay)

    def test_static_tokens_follow_the_admission_token_at_admission(self):
        stay = self._stay("synth-000")
        head = [t.split("=")[0] for t in stay["tokens"][2:2 + len(STATIC)]]
        self.assertEqual(head, STATIC)
        self.assertEqual(stay["pos_min"][2:2 + len(STATIC)], [0] * len(STATIC))

    def test_no_terminal_token_or_disposition_in_the_24h_artifact(self):
        inv = {i: t for t, i in self.vocab.items()}
        for row in self.events.iter_rows(named=True):
            tokens = [inv[t] for t in row["token"]]
            self.assertFalse([t for t in tokens if t.startswith(("DISCHARGE//", "ADMISSION//"))
                              or t in ("<bos>", "<eos>")])

    # --- the 24 h artifact is unchanged ------------------------------------------------
    def test_24h_events_are_byte_identical_with_and_without_the_gem_run(self):
        self.assertEqual(self.both_sha, self.events_sha_before_gem)
        self.assertEqual(self.both_sha, self.only_sha)

    def test_terminal_allowlist_does_not_shift_data_token_ids(self):
        # The allowlist is appended after every data-derived token, so the 24 h token
        # ids equal those of a config with no `gem` block (only the vocab hash differs).
        self.assertEqual(self.events["token"].to_list(), self.nogem_events["token"].to_list())
        for token, idx in self.nogem_vocab.items():
            self.assertEqual(self.vocab[token], idx)
        self.assertFalse(any(t.startswith("DISCHARGE//") for t in self.nogem_vocab))
        # Concepts charted only outside the 24 h window (ED / ward ADT locations) join
        # the one shared vocabulary after the 24 h tokens instead of becoming <unk>.
        for location in ("ed", "ward"):
            self.assertNotIn(location, self.nogem_vocab)
            self.assertGreaterEqual(self.vocab[location], len(self.nogem_vocab))

    # --- hospitalization window --------------------------------------------------------
    def test_pre_and_post_icu_events_are_included_relative_to_hospital_admission(self):
        stay = self._stay("synth-000")
        maps = [(p, v) for t, p, v in zip(stay["tokens"], stay["pos_min"], stay["value"])
                if t.startswith("map=")]
        values = {v: p for p, v in maps}
        self.assertEqual(values[72.25], 180)                     # ED, adm + 3 h
        self.assertEqual(values[73.25], 360 + 100 * 60)          # ward, ICU + 100 h
        self.assertNotIn(71.25, values)                          # before admission
        self.assertNotIn(74.25, values)                          # after discharge
        locations = [(t, p) for t, p in zip(stay["tokens"], stay["pos_min"])
                     if t in ("ed", "icu", "ward")]
        self.assertEqual(locations, [("ed", 0), ("icu", 360), ("ward", 360 + 96 * 60)])
        self.assertEqual(stay["pos_min"], sorted(stay["pos_min"]))
        # The 24 h artifact keeps ICU-relative positions and no ED / ward events.
        row = self.events.filter(pl.col("hosp_id") == "synth-000").row(0, named=True)
        self.assertNotIn(72.25, row["value"])
        self.assertNotIn(73.25, row["value"])

    def test_anchor_idx_is_the_last_event_at_or_before_icu_admit_plus_24h(self):
        stay = self._stay("synth-000")
        anchor_min = self._minutes("synth-000", "anchor_dttm")
        self.assertEqual(stay["anchor_min"], anchor_min)
        idx = stay["anchor_idx"]
        self.assertLessEqual(stay["pos_min"][idx], anchor_min)
        self.assertGreater(stay["pos_min"][idx + 1], anchor_min)

    def test_treatment_static_and_framing_tokens_are_never_target_eligible(self):
        stay = self._stay("synth-000")
        for token, eligible in zip(stay["tokens"], stay["target_eligible"]):
            if token.startswith(("DISCHARGE//", "map=")):
                self.assertTrue(eligible, token)
            else:   # <bos>, ADMISSION//, static, adt (input-only), <eos>
                self.assertFalse(eligible, token)

    # --- windowing -----------------------------------------------------------------------
    def test_long_stay_splits_into_consistent_windows(self):
        rows = self._windows(LONG_STAY)
        self.assertGreater(len(rows), 1)
        total = sum(len(r["token"]) for r in rows)
        self.assertGreaterEqual(total, LONG_EVENTS + 4)
        start = 0
        for i, row in enumerate(rows):
            n = len(row["token"])
            self.assertLessEqual(n, MAX_TOKENS)
            self.assertEqual(row["n_events"], n)
            for key in ("pos_min", "value", "target_eligible", "soft_token", "soft_weight"):
                self.assertEqual(len(row[key]), n, key)
            self.assertEqual(row["continuation_index"], i)
            self.assertEqual(row["continues_from_previous"], i > 0)
            self.assertEqual(row["continues_to_next"], i < len(rows) - 1)
            self.assertEqual((row["source_start"], row["source_end"]), (start, start + n))
            self.assertEqual(row["n_windows"], len(rows))
            start += n
        self.assertEqual(start, total)
        first = [self.inv[t] for t in rows[0]["token"][:2]]
        last = [self.inv[t] for t in rows[-1]["token"][-2:]]
        self.assertEqual(first[0], "<bos>")
        self.assertTrue(first[1].startswith("ADMISSION//"))
        self.assertTrue(last[0].startswith("DISCHARGE//"))
        self.assertEqual(last[1], "<eos>")

    def test_continuation_header_leaves_room_for_the_stay_header(self):
        # trunk.continuation_header (configs/model.yaml): continuation windows are cut
        # max_tokens - header_len long so the loader can re-insert the header; with the
        # flag off the windows are byte-identical to gem_window_bounds.
        from src.data.dataset import header_token_ids, stay_header_length
        from src.data.tokenize import tokenize_site
        from src.eval.synthetic_bundle import SYNTHETIC_SITE

        old_cwd = os.getcwd()
        os.chdir(self.work)
        try:
            out = {}
            for flag in (False, True):
                target = self.work / f"output/intermediate_phi/header_{flag}"
                target.mkdir(parents=True, exist_ok=True)
                (target / "vocab.json").write_text((self.both / "vocab.json").read_text())
                tokenize_site(self.cfg, SYNTHETIC_SITE, self.site, target, self.blob,
                              trajectory="hospitalization", max_tokens=MAX_TOKENS,
                              continuation_header=flag, **self.kw)
                out[flag] = target / "gem_events.parquet"
        finally:
            os.chdir(old_cwd)
        from src.data.tokenize import gem_window_bounds
        off = pl.read_parquet(out[False]).filter(pl.col("hosp_id") == LONG_STAY) \
            .sort("continuation_index")
        total = int(off["n_events"].sum())
        self.assertEqual(list(zip(off["source_start"], off["source_end"])),
                         gem_window_bounds(total, MAX_TOKENS))
        on = pl.read_parquet(out[True]).filter(pl.col("hosp_id") == LONG_STAY) \
            .sort("continuation_index")
        header = header_token_ids(self.vocab)
        h = stay_header_length(on["token"][0].to_list(), header)
        self.assertEqual(h, 2 + len(STATIC))          # <bos>, ADMISSION//, statics
        lengths = on["n_events"].to_list()
        self.assertEqual(lengths[0], MAX_TOKENS)
        for n in lengths[1:]:
            self.assertLessEqual(n + h, MAX_TOKENS)
        self.assertEqual(sum(lengths), total)
        self.assertGreater(len(lengths), off.height)

    def test_short_stays_are_one_window(self):
        rows = self._windows("synth-001")
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]["continues_from_previous"], rows[0]["continues_to_next"]),
                         (False, False))

    def test_every_window_is_bound_to_the_frozen_vocabulary(self):
        from src.data.segments import artifact_binding

        binding = artifact_binding(self.blob)
        for hashes in self.gem["artifact_hashes"].to_list():
            # The vocabulary binding, plus the site binding (tokenize.site_binding): the
            # site and the SHA-256 of its declarations.
            self.assertEqual({k: hashes[k] for k in binding}, binding)
            self.assertEqual(set(hashes) - set(binding), {"site", "site_declarations"})
            self.assertEqual(len(hashes["site_declarations"]), 64)
        self.assertEqual(set(self.gem["trajectory"].to_list()), {"hospitalization"})

    # --- vocabulary ------------------------------------------------------------------------
    def test_terminal_and_admission_allowlist_is_in_the_vocab_even_if_absent(self):
        gem = self.cfg["gem"]
        for label in gem["dispositions"]:
            self.assertIn(f"DISCHARGE//{label}", self.vocab)
        for label in gem["admission_types"]:
            self.assertIn(f"ADMISSION//{label}", self.vocab)
        self.assertIn("DISCHARGE//hospice", self.vocab)        # no hospice stay here

    def test_gem_requires_the_frozen_vocab_with_the_allowlist(self):
        from src.data.cohort import QualificationError
        from src.data.tokenize import tokenize_site
        from src.eval.synthetic_bundle import SYNTHETIC_SITE

        old_cwd = os.getcwd()
        os.chdir(self.work)
        try:
            out = self.work / "output/intermediate_phi/gem_reject"
            with self.assertRaisesRegex(QualificationError, "frozen vocabulary"):
                tokenize_site(self.cfg, SYNTHETIC_SITE, self.site, out, None,
                              trajectory="hospitalization", **self.kw)
            nogem_cfg = {k: v for k, v in self.cfg.items() if k != "gem"}
            with self.assertRaisesRegex(QualificationError, "gem"):
                tokenize_site(nogem_cfg, SYNTHETIC_SITE, self.site, out, self.blob,
                              trajectory="hospitalization", **self.kw)
            with self.assertRaisesRegex(ValueError, "trajectory"):
                tokenize_site(self.cfg, SYNTHETIC_SITE, self.site, out, self.blob,
                              trajectory="icu_episode", **self.kw)
        finally:
            os.chdir(old_cwd)

    # --- ModelDataset ----------------------------------------------------------------------
    def _dataset(self, records=None, expected=None):
        from src.data.dataset import ModelDataset
        from src.data.segments import artifact_binding
        from src.data.targets import TargetBuilder

        builder = TargetBuilder(len(self.vocab), 4, 48, {}, mode="gem")
        rows = records if records is not None else [
            {**r, "value": [None] * len(r["token"])}
            for r in pl.read_parquet(self.gem_path).to_dicts()
        ]
        return ModelDataset(rows, representation="gem", target_builder=builder,
                            expected_hashes=expected or artifact_binding(self.blob))

    def test_gem_windows_flow_through_model_dataset(self):
        dataset = self._dataset()
        self.assertEqual(len(dataset), len(self.gem))
        discharge = {i for t, i in self.vocab.items() if t.startswith("DISCHARGE//")}
        for index in range(len(dataset)):
            sample = dataset[index]
            (segment,) = sample["segments"]
            n = len(sample["input_ids"])
            self.assertLessEqual(n, MAX_TOKENS)
            self.assertEqual((segment["packed_start"], segment["packed_end"]), (0, n))
            self.assertEqual(segment["outcome_labels"], [])
            self.assertIsNone(segment["threshold_query"])
            if not segment["continues_to_next"]:
                # The last physiologic event predicts the terminal disposition.
                targets = [t for t, m in zip(sample["ntp_target"], sample["ntp_mask"]) if m]
                self.assertIn(targets[-1], discharge)
            else:
                # Targets are built on the whole stay, so a window's last eligible event
                # still predicts the first eligible event of the next window.
                self.assertTrue(any(sample["ntp_mask"]))

    def test_model_dataset_keeps_the_segments_hash_check_for_gem(self):
        from src.data.targets import TargetContractError

        rows = pl.read_parquet(self.gem_path).to_dicts()
        stale = [{**r, "artifact_hashes": {"vocabulary": "x"}} for r in rows]
        with self.assertRaisesRegex(TargetContractError, "tokenizer-v2"):
            self._dataset(stale)
        with self.assertRaisesRegex(TargetContractError, "numeric_edges"):
            self._dataset(expected={"numeric_edges": "0" * 64})

    def test_model_dataset_rejects_a_missing_window(self):
        from src.data.targets import TargetContractError

        rows = [r for r in pl.read_parquet(self.gem_path).to_dicts()
                if not (r["hosp_id"] == LONG_STAY and r["continuation_index"] == 1)]
        with self.assertRaisesRegex(TargetContractError, "window"):
            self._dataset([{**r, "value": [None] * len(r["token"])} for r in rows])


    def test_model_dataset_refuses_gem_rows_outside_the_gem_representation(self):
        """GEM rows through the 24 h path would skip the post-anchor relaxation's
        coupling (and a 24 h shard through the gem path would relax it)."""
        from src.data.dataset import ModelDataset
        from src.data.segments import artifact_binding
        from src.data.targets import TargetBuilder, TargetContractError

        rows = [{**r, "value": [None] * len(r["token"])}
                for r in pl.read_parquet(self.gem_path).to_dicts()]
        with self.assertRaisesRegex(TargetContractError, "representation='gem'"):
            ModelDataset(rows, representation="decile",
                         target_builder=TargetBuilder(len(self.vocab), 4, 48, {}),
                         expected_hashes=artifact_binding(self.blob))

    def test_the_gem_representation_requires_a_gem_mode_target_builder(self):
        from src.data.dataset import ModelDataset
        from src.data.segments import artifact_binding
        from src.data.targets import TargetBuilder, TargetContractError

        rows = [{**r, "value": [None] * len(r["token"])}
                for r in pl.read_parquet(self.gem_path).to_dicts()]
        with self.assertRaisesRegex(TargetContractError, "mode='gem'"):
            ModelDataset(rows, representation="gem",
                         target_builder=TargetBuilder(len(self.vocab), 4, 48, {}),
                         expected_hashes=artifact_binding(self.blob))


class GemWindowBoundsTest(unittest.TestCase):
    """The final `<eos>` is never alone in a window (`gem_window_bounds` tail branch)."""

    def test_a_one_token_tail_moves_one_token_into_the_last_window(self):
        from src.data.tokenize import gem_window_bounds

        self.assertEqual(gem_window_bounds(65, 64), [(0, 63), (63, 65)])
        self.assertEqual(gem_window_bounds(129, 64)[-2:], [(64, 127), (127, 129)])

    def test_exact_and_short_streams_are_not_adjusted(self):
        from src.data.tokenize import gem_window_bounds

        self.assertEqual(gem_window_bounds(64, 64), [(0, 64)])
        self.assertEqual(gem_window_bounds(1, 64), [(0, 1)])
        self.assertEqual(gem_window_bounds(66, 64), [(0, 64), (64, 66)])

    def test_a_window_must_hold_at_least_two_tokens(self):
        from src.data.tokenize import gem_window_bounds

        with self.assertRaises(ValueError):
            gem_window_bounds(10, 1)


class GemTargetBuilderTest(unittest.TestCase):
    """`TargetBuilder(mode="gem")` vs the default 24 h mode."""

    BOS, ADM, MAP1, NE, MAP2, DIS, EOS = 1, 10, 11, 12, 13, 14, 2

    def _episode(self, **overrides):
        record = {
            "episode_key": "opaque-gem",
            "token": [self.BOS, self.ADM, self.MAP1, self.NE, self.MAP2, self.DIS, self.EOS],
            "pos_min": [0, 0, 30, 40, 2000, 3000, 3000],
            "value": [None] * 7,
            "target_eligible": [False, False, True, False, True, True, False],
            "anchor_idx": 3,
            "anchor_min": 1440,
            "outcomes": [],
        }
        record.update(overrides)
        return record

    def _builder(self, mode=None):
        from src.data.targets import TargetBuilder

        kwargs = {} if mode is None else {"mode": mode}
        return TargetBuilder(32, 4, 48, {}, **kwargs)

    def test_gem_mode_targets_the_terminal_token_and_never_a_treatment(self):
        built = self._builder("gem").build(self._episode())
        self.assertEqual(built["ntp_target"], [0, 0, self.MAP2, 0, self.DIS, 0, 0])
        self.assertEqual(built["ntp_mask"], [False, False, True, False, True, False, False])
        self.assertNotIn(self.NE, built["ntp_target"])
        self.assertNotIn(self.BOS, built["ntp_target"])
        self.assertNotIn(self.ADM, built["ntp_target"])
        self.assertEqual(built["outcome_labels"], [])
        self.assertIsNone(built["threshold_query"])
        self.assertEqual(built["anchor_idx"], 3)

    def test_gem_mode_carries_no_tte_labels(self):
        from src.data.targets import TargetContractError

        outcome = {"target_idx": 1, "status": "negative", "time_from_anchor_hours": 4,
                   "threshold_bin": 2, "direction": "below"}
        with self.assertRaisesRegex(TargetContractError, "gem"):
            self._builder("gem").build(self._episode(outcomes=[outcome]))

    def test_default_mode_still_rejects_post_anchor_features(self):
        from src.data.targets import TargetContractError

        for builder in (self._builder(), self._builder("icu_24h")):
            with self.assertRaisesRegex(TargetContractError, "post-anchor"):
                builder.build(self._episode())

    def test_unknown_mode_is_rejected(self):
        from src.data.targets import TargetContractError

        with self.assertRaisesRegex(TargetContractError, "mode"):
            self._builder("hospitalization")

    def test_gem_window_without_the_anchor_is_allowed(self):
        built = self._builder("gem").build(self._episode(anchor_idx=None))
        self.assertIsNone(built["anchor_idx"])


class GemConfigTest(unittest.TestCase):
    def setUp(self):
        from src.data.tokenize import ROOT

        self.gem = yaml.safe_load((ROOT / "configs/data.yaml").read_text())["gem"]
        self.model = yaml.safe_load((ROOT / "configs/model.yaml").read_text())

    def test_window_matches_the_trunk_context(self):
        self.assertEqual(self.gem["max_tokens"], self.model["trunk"]["max_tokens"])

    def test_real_discharge_categories_map_as_specified(self):
        from src.data.tokenize import disposition_label

        expected = {
            "Home": "home", "Missing": "unknown", "Expired": "expired", "Hospice": "hospice",
            "Skilled Nursing Facility (SNF)": "facility",
            "Acute Inpatient Rehab Facility": "facility",
            "Long Term Care Hospital (LTACH)": "facility",
            "Psychiatric Hospital": "facility", "Acute Care Hospital": "facility",
            "Assisted Living": "facility", "Against Medical Advice (AMA)": "ama",
            "Other": "other", None: "unknown", "": "unknown", "Unknown": "unknown",
            "something unmapped": "unknown",
        }
        for raw, label in expected.items():
            self.assertEqual(disposition_label(raw, self.gem), label, raw)
        self.assertEqual(sorted(self.gem["dispositions"]),
                         sorted(["home", "facility", "hospice", "expired", "ama", "other",
                                 "unknown"]))

    def test_admission_types_map_into_the_allowlist(self):
        from src.data.tokenize import admission_label

        self.assertEqual(admission_label("ed", self.gem), "ed")
        self.assertEqual(admission_label("Elective", self.gem), "elective")
        self.assertEqual(admission_label("direct", self.gem), "direct")
        self.assertEqual(admission_label(None, self.gem), "unknown")
        self.assertIn(admission_label("osh", self.gem), self.gem["admission_types"])


if __name__ == "__main__":
    unittest.main()
