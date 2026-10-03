"""U1: closure-aware segments, precedence policy v1, and the single binning function.

Boundary table (physician CSV + configs/data.yaml forced edges and target directions):

| concept          | value   | expected segment                | rule                         |
|------------------|---------|---------------------------------|------------------------------|
| lactate          | 2.0     | (1.6, 2.0]                      | CSV flag; forced edge, above |
| lactate          | 2.01    | (2.0, 2.2]                      | CSV flag                     |
| lactate          | 4.0     | (3.4, 4.0]                      | forced split, direction above|
| map              | 65.0    | [65, 67]                        | forced edge, direction below |
| map              | 64.9    | (61, 65)                        | forced edge, direction below |
| spo2             | 88.0    | [88, 90)                        | forced edge, direction below |
| spo2             | 92.0    | [92, 92]                        | exact point row              |
| respiratory_rate | 9.2/9.8 | (6, 9] / [10, 10]               | gap -> nearest               |
| respiratory_rate | 9.5     | (6, 9]                          | gap, equidistant -> lower    |
| temp_c           | 37.4995 | (37.111, 37.5]                  | overlap: earlier owns        |
| fio2_set         | 0.2     | (0.2, 0.3]                      | gap -> nearest (distance 0)  |
| lactate          | 0.0/50  | first / last segment            | clamp                        |
"""

from __future__ import annotations

import json
import math

import polars as pl
import pytest
import yaml

from src.data import segments as S
from src.data.tokenize import ROOT

CSV = ROOT / "external/clifatron/tokenETL/config/critical_illness_tokenization_final_with_intervals.csv"
CFG = yaml.safe_load((ROOT / "configs/data.yaml").read_text())
FORCED = CFG["value_binning"]["forced_edges"]
DIRECTIONS = {t["name"]: t["direction"] for t in CFG["target_concepts"]}


@pytest.fixture(scope="module")
def segs() -> dict[str, list[dict]]:
    return S.load_csv_segments(CSV, forced_edges=FORCED, directions=DIRECTIONS)


def _seg_of(segs: dict, concept: str, value: float) -> tuple:
    idx = S.bin_index(value, segs[concept])
    assert idx is not None
    s = segs[concept][idx]
    return (s["lo"], s["hi"], s["lo_closed"], s["hi_closed"])


def _containing(segments: list[dict], value: float) -> list[int]:
    return [i for i, s in enumerate(segments) if S.contains(s, value)]


def _row(lo, hi, lo_flag, hi_flag, exact=False) -> dict:
    return {"lo": lo, "hi": hi, "lo_closed": lo_flag == "[", "hi_closed": hi_flag == "]",
            "exact": exact}


# ---- CSV interval flags and forced-edge direction closure -----------------------------

def test_lactate_threshold_value_stays_on_non_event_side(segs):
    assert _seg_of(segs, "lactate", 2.0) == (1.6, 2.0, False, True)
    assert _seg_of(segs, "lactate", 2.01) == (2.0, 2.2, False, True)


def test_lactate_forced_edge_splits_containing_segment_by_direction(segs):
    # (3.4, 5.4] is split at the forced 4.0 edge; direction "above" keeps 4.0 below.
    assert _seg_of(segs, "lactate", 4.0) == (3.4, 4.0, False, True)
    assert _seg_of(segs, "lactate", 4.01) == (4.0, 5.4, False, True)


def test_map_65_goes_above_the_edge_for_a_below_target(segs):
    lo, hi, lo_closed, _ = _seg_of(segs, "map", 65.0)
    assert (lo, lo_closed) == (65.0, True)
    assert _seg_of(segs, "map", 64.9) == (61.0, 65.0, False, False)


def test_spo2_88_goes_above_and_92_is_a_point_bin(segs):
    assert _seg_of(segs, "spo2", 88.0) == (88.0, 90.0, True, False)
    assert _seg_of(segs, "spo2", 87.99) == (49.9, 88.0, True, False)
    assert _seg_of(segs, "spo2", 90.0) == (90.0, 91.0, True, True)
    assert _seg_of(segs, "spo2", 92.0) == (92.0, 92.0, True, True)
    assert _seg_of(segs, "spo2", 92.5) == (92.0, 93.0, False, True)


def test_respiratory_rate_gap_goes_to_nearest_and_ties_go_lower(segs):
    assert _seg_of(segs, "respiratory_rate", 9.2) == (6.0, 9.0, False, True)
    assert _seg_of(segs, "respiratory_rate", 9.8) == (10.0, 10.0, True, True)
    assert _seg_of(segs, "respiratory_rate", 9.5) == (6.0, 9.0, False, True)
    assert _seg_of(segs, "respiratory_rate", 0.0) == (0.0, 0.0, True, True)


def test_temp_c_near_duplicate_boundary_is_resolved(segs):
    assert len(_containing(segs["temp_c"], 37.4995)) == 1
    assert _seg_of(segs, "temp_c", 37.4995) == (37.111, 37.5, False, True)
    assert _seg_of(segs, "temp_c", 37.55) == (37.5, 37.611, False, True)
    assert all(37.499 not in (s["lo"], s["hi"]) for s in segs["temp_c"])


def test_fio2_set_value_in_gap_uses_gap_rule(segs):
    assert _seg_of(segs, "fio2_set", 0.2) == (0.2, 0.3, False, True)
    assert _seg_of(segs, "fio2_set", 0.05) == (0.0, 0.0, True, True)


def test_out_of_range_values_clamp_to_end_segments(segs):
    assert S.bin_index(0.0, segs["lactate"]) == 0
    assert S.bin_index(50.0, segs["lactate"]) == len(segs["lactate"]) - 1


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), float("-inf")])
def test_missing_or_non_finite_values_get_no_bin(segs, value):
    assert S.bin_index(value, segs["lactate"]) is None


def test_exact_dose_rows_match_only_min_value(segs):
    # peak_inspiratory_pressure_set (5,10) carries exact_dose_token=1 -> point at 5.
    assert _seg_of(segs, "peak_inspiratory_pressure_set", 5.0) == (5.0, 5.0, True, True)
    assert _seg_of(segs, "peak_inspiratory_pressure_set", 4.0) == (0.0, 5.0, True, False)


def test_angiotension_csv_prefix_is_aliased_to_clif_angiotensin(segs):
    assert "angiotensin_ng_kg_min" in segs
    assert not any(name.startswith("angiotension") for name in segs)


# ---- precedence policy on synthetic rows ------------------------------------------------

@pytest.mark.parametrize("earlier_flags", [("(", "]"), ("[", ")")])
@pytest.mark.parametrize("later_flags", [("(", "]"), ("[", ")")])
def test_overlap_all_closure_combinations_give_a_strict_partition(earlier_flags, later_flags):
    earlier = _row(1.0, 3.0, *earlier_flags)
    later = _row(2.0, 5.0, *later_flags)
    out = S.build_segments([later, earlier])  # input order must not matter
    S.validate_partition(out)
    assert len(out) == 2
    first, second = out
    # Earlier segment keeps its upper endpoint and closure.
    assert (first["lo"], first["hi"]) == (1.0, 3.0)
    assert first["lo_closed"] == earlier["lo_closed"]
    assert first["hi_closed"] == earlier["hi_closed"]
    # Later starts there with the complementary closure.
    assert (second["lo"], second["hi"]) == (3.0, 5.0)
    assert second["lo_closed"] is (not earlier["hi_closed"])
    assert second["hi_closed"] == later["hi_closed"]
    # Shared endpoint and the overlapped region map to exactly one segment.
    for v in (2.0, 2.5, 3.0, 3.0001, 4.9):
        assert len(_containing(out, v)) == 1
    assert S.bin_index(2.5, out) == 0
    assert S.bin_index(3.0, out) == (0 if earlier["hi_closed"] else 1)


def test_boundaries_within_relative_tolerance_are_merged():
    out = S.build_segments([_row(0.0, 10.0, "[", "]"), _row(10.0 * (1 + 5e-7), 20.0, "(", "]")])
    assert [(s["lo"], s["hi"]) for s in out] == [(0.0, 10.0), (10.0, 20.0)]
    assert len(_containing(out, 10.0000001)) == 1


def test_point_rows_win_over_containing_intervals():
    out = S.build_segments([_row(0.0, 10.0, "[", "]"), _row(4.0, 4.0, "[", "]", exact=True)])
    S.validate_partition(out)
    assert [(s["lo"], s["hi"], s["lo_closed"], s["hi_closed"]) for s in out] == [
        (0.0, 4.0, True, False), (4.0, 4.0, True, True), (4.0, 10.0, False, True)]


def test_non_target_forced_edges_put_the_edge_value_above():
    out = S.build_segments([_row(0.0, 10.0, "[", "]")], forced_edges=[5.0], direction=None)
    assert S.bin_index(5.0, out) == 1
    assert out[1]["lo_closed"] is True


def test_forced_edge_outside_segment_range_fails_closed():
    with pytest.raises(ValueError, match="outside"):
        S.build_segments([_row(0.0, 10.0, "[", "]")], forced_edges=[12.0], direction="above")


def test_quantile_edges_become_half_open_segments_with_direction_closure():
    plain = S.segments_from_edges([1.0, 2.0, 3.0])
    assert len(plain) == 4
    assert plain[0]["lo"] is None and plain[-1]["hi"] is None
    assert S.bin_index(2.0, plain) == 2          # [a, b) semantics
    assert S.bin_index(-1e9, plain) == 0 and S.bin_index(1e9, plain) == 3
    above = S.segments_from_edges([1.0, 2.0, 3.0], forced_edges=[2.0], direction="above")
    assert S.bin_index(2.0, above) == 1          # threshold value on the non-event side
    S.validate_partition(above)


# ---- soft bins --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "concept,value",
    [("lactate", 1.0), ("lactate", 0.0), ("lactate", 50.0), ("spo2", 92.0),
     ("respiratory_rate", 10.0), ("map", 65.0)],
)
def test_soft_triple_sums_to_one_and_centers_on_the_hard_segment(segs, concept, value):
    triple = S.soft_bins(value, segs[concept], kernel_bins=1)
    assert len(triple) == 3
    assert math.isclose(sum(w for _, w in triple), 1.0)
    hard = S.bin_index(value, segs[concept])
    hard_weight = sum(w for b, w in triple if b == hard)
    # The hard segment carries the most mass (a value exactly on a boundary ties).
    assert hard_weight >= max(w for _, w in triple) - 1e-12
    assert all(0 <= b < len(segs[concept]) for b, _ in triple)


def test_soft_bins_for_missing_values_keep_the_fixed_width_contract(segs):
    assert S.soft_bins(None, segs["lactate"], kernel_bins=1) == [(None, 1.0), (None, 0.0), (None, 0.0)]
    assert S.soft_bins(1.0, segs["lactate"], kernel_bins=0) == [(S.bin_index(1.0, segs["lactate"]), 1.0)]


# ---- every CSV measurement ----------------------------------------------------------------

def test_every_csv_measurement_loads_into_a_strict_partition(segs):
    assert len(segs) == 92
    for concept, segments in segs.items():
        S.validate_partition(segments)
        bounds = sorted({b for s in segments for b in (s["lo"], s["hi"])})
        probes = set(bounds)
        for a, b in zip(bounds, bounds[1:]):
            probes.update({(a + b) / 2, a + (b - a) * 1e-6, b - (b - a) * 1e-6})
        probes.update({bounds[0] - 1.0, bounds[-1] + 1.0})
        for v in probes:
            hits = _containing(segments, v)
            assert len(hits) <= 1, (concept, v, hits)
            idx = S.bin_index(v, segments)
            assert idx is not None and 0 <= idx < len(segments), (concept, v)
            if hits:
                assert idx == hits[0], (concept, v)


def test_segments_and_policy_are_json_serializable(segs):
    # v2 (2026-10-02): dose concepts close the gap between the [0, 0] stop bin and
    # their first positive segment (policy step 8).
    assert S.POLICY_VERSION == 2
    assert json.loads(json.dumps(segs)) == segs


# ---- U2: bins for every numeric concept, zero-aware doses ---------------------------------

def _fit_events(concepts: dict) -> pl.DataFrame:
    """``{concept: (source, values)}`` -> a train-partition long event frame."""
    rows = {"concept": [], "value": [], "source": []}
    for concept, (source, values) in concepts.items():
        for v in values:
            rows["concept"].append(concept)
            rows["value"].append(None if v is None else float(v))
            rows["source"].append(source)
    return pl.DataFrame(rows, schema={"concept": pl.String, "value": pl.Float64,
                                      "source": pl.String})


def _build_all(fit, *, bin_overrides: dict | None = None, directions: dict | None = None,
               targets: list[str] | None = None):
    import copy

    from src.data.tokenize import build_segments

    bin_cfg = copy.deepcopy(CFG["value_binning"])
    bin_cfg.update(bin_overrides or {})
    targets = [t["name"] for t in CFG["target_concepts"]] if targets is None else targets
    return build_segments(bin_cfg, fit, targets, DIRECTIONS if directions is None else directions)


def test_config_declares_full_coverage_and_dose_sources():
    vb = CFG["value_binning"]
    assert vb["coverage"] == "all"
    assert vb["ordinal_max_distinct"] == 25
    assert vb["min_count"] == 20
    assert vb["quantile_n_bins"] == 10
    assert {"meds", "meds_intermittent"} <= set(vb["dose_sources"])


def test_csv_concept_uses_physician_segments():
    import numpy as np

    fit = _fit_events({"potassium": ("labs", np.linspace(2.5, 6.5, 200))})
    segs, sources = _build_all(fit)
    assert sources["potassium"] == "csv"
    assert segs["potassium"] == S.load_csv_segments(CSV, ["potassium"])["potassium"]


def test_integer_scale_with_few_distinct_values_gets_one_point_bin_per_value():
    fit = _fit_events({"gcs_total": ("assessments", list(range(3, 16)) * 10)})
    segs, sources = _build_all(fit)
    assert sources["gcs_total"] == "ordinal"
    gcs = segs["gcs_total"]
    assert len(gcs) == 13
    assert all(S.is_point(s) for s in gcs)
    assert [s["lo"] for s in gcs] == [float(v) for v in range(3, 16)]
    assert [S.bin_index(v, gcs) for v in range(3, 16)] == list(range(13))
    S.validate_partition(gcs)


def test_non_csv_continuous_concept_gets_quantile_bins_with_forced_edges_pinned():
    import numpy as np

    values = np.random.default_rng(0).lognormal(0.0, 0.6, 500)
    fit = _fit_events({"synth_lab": ("labs", values)})
    segs, sources = _build_all(
        fit, bin_overrides={"forced_edges": {**FORCED, "synth_lab": [1.0]}},
        directions={**DIRECTIONS, "synth_lab": "above"},
    )
    assert sources["synth_lab"] == "quantile"
    lab = segs["synth_lab"]
    assert 9 <= len(lab) <= 11
    S.validate_partition(lab)
    assert any(a["hi"] == 1.0 == b["lo"] for a, b in zip(lab, lab[1:]))
    # direction "above": the threshold value stays on the non-event (lower) side.
    assert S.bin_index(1.0, lab) == S.bin_index(0.999, lab)
    assert S.bin_index(1.0, lab) != S.bin_index(1.001, lab)


def test_concept_with_fewer_than_min_count_values_gets_one_reported_bin():
    fit = _fit_events({"rare_lab": ("labs", [0.3, 1.7, 2.2, 9.1, 40.5])})
    segs, sources = _build_all(fit)
    assert sources["rare_lab"] == "single"
    assert len(segs["rare_lab"]) == 1
    assert {S.bin_index(v, segs["rare_lab"]) for v in (0.3, 1.7, 40.5, -5.0)} == {0}


def test_non_csv_dose_with_many_zeros_keeps_zero_distinct_and_fits_on_positive_values():
    import numpy as np

    rng = np.random.default_rng(1)
    # A dose concept with no physician-CSV or literature bins falls back to quantiles.
    doses = [0.0] * 200 + list(rng.uniform(100.0, 2500.0, 300))   # 40% zeros
    fit = _fit_events({"synthdrug_u_hr": ("meds", doses)})
    segs, sources = _build_all(fit)
    hep = segs["synthdrug_u_hr"]
    assert sources["synthdrug_u_hr"] == "quantile"
    S.validate_partition(hep)
    assert hep[0] == S.make_segment(0.0, 0.0, True, True)
    assert S.bin_index(0.0, hep) == 0
    assert S.bin_index(1.0, hep) not in (None, 0)
    # Quantiles were fit on the positive doses: the zeros did not collapse edges.
    assert len(hep) >= 10
    assert sum(1 for s in hep if s["lo"] == 0.0 and s["hi"] == 0.0) == 1


def test_unconverted_dose_stop_lands_in_the_zero_bin_distinct_from_a_running_dose():
    import numpy as np

    rng = np.random.default_rng(2)
    doses = [0.0] * 40 + list(rng.uniform(10.0, 200.0, 160))
    fit = _fit_events({"fentanyl_mcg_hr": ("meds", doses)})
    segs, _ = _build_all(fit)
    fent = segs["fentanyl_mcg_hr"]
    assert S.bin_index(0.0, fent) == 0 and S.is_point(fent[0])
    assert S.bin_index(25.0, fent) != S.bin_index(0.0, fent)


def test_dose_concepts_are_config_driven_and_csv_dose_zero_rows_are_not_duplicated():
    import numpy as np

    rng = np.random.default_rng(3)
    vals = [0.0] * 30 + list(rng.uniform(0.01, 0.5, 70))
    fit = _fit_events({
        "norepinephrine_mcg_kg_min": ("meds", vals),     # CSV dose: has its own [0,0] row
        "zeroish_lab": ("labs", vals),                   # same values, NOT a dose source
    })
    segs, sources = _build_all(fit)
    assert sources["norepinephrine_mcg_kg_min"] == "csv"
    ne = segs["norepinephrine_mcg_kg_min"]
    assert sum(1 for s in ne if S.is_point(s) and s["lo"] == 0.0) == 1
    assert not any(S.is_point(s) for s in segs["zeroish_lab"])
    # A table flagged `dose: true` is a dose source too, independent of dose_sources.
    import copy

    from src.data.tokenize import build_segments

    bin_cfg = copy.deepcopy(CFG["value_binning"])
    bin_cfg["dose_sources"] = []
    segs3, _ = build_segments(bin_cfg, fit, [], {}, tables={"labs": {"dose": True}})
    assert segs3["zeroish_lab"][0] == S.make_segment(0.0, 0.0, True, True)


def test_running_csv_dose_below_the_first_positive_segment_is_never_the_stop_bin():
    """Policy v2 step 8: the physician CSV gives lorazepam_mg_hr only [0, 0] and [1, 1]
    below 1 mg/hr. Under v1 the gap rule sent a running 0.25-0.5 mg/hr infusion to the
    nearest segment, the stop bin; v2 inserts an open (0, 1) running-dose segment."""
    fit = _fit_events({"lorazepam_mg_hr": ("meds", [0.0, 0.25, 0.5, 1.0, 2.0] * 10)})
    segs, sources = _build_all(fit)
    assert sources["lorazepam_mg_hr"] == "csv"
    lzp = segs["lorazepam_mg_hr"]
    S.validate_partition(lzp)
    stop = S.bin_index(0.0, lzp)
    assert lzp[stop] == S.make_segment(0.0, 0.0, True, True)
    for running in (1e-9, 0.25, 0.5, 0.999):
        assert S.bin_index(running, lzp) != stop, running
    assert lzp[S.bin_index(0.25, lzp)] == S.make_segment(0.0, 1.0, False, False)
    # The CSV's own [1, 1] point bin is unchanged, and so is every segment above it.
    assert lzp[S.bin_index(1.0, lzp)] == S.make_segment(1.0, 1.0, True, True)
    csv = S.load_csv_segments(CSV, ["lorazepam_mg_hr"])["lorazepam_mg_hr"]
    assert lzp[S.bin_index(1.0, lzp):] == csv[S.bin_index(1.0, csv):]


@pytest.mark.parametrize("values, source", [
    ([0.0] * 30 + [0.25 * k for k in range(1, 200)], "quantile"),   # quantile-binned dose
    ([0.0, 1.0, 2.0, 3.0, 4.0] * 10, "ordinal"),                     # integer-valued dose
    ([0.0, 0.3, 0.6], "single"),                                    # too few to fit
])
def test_no_strictly_positive_dose_reaches_the_stop_bin(values, source):
    fit = _fit_events({"synth_med_u_hr": ("meds", values)})
    segs, sources = _build_all(fit)
    assert sources["synth_med_u_hr"] == source
    med = segs["synth_med_u_hr"]
    S.validate_partition(med)
    stop = S.bin_index(0.0, med)
    assert med[stop] == S.make_segment(0.0, 0.0, True, True)
    for running in (1e-9, 0.1, 0.25, 0.4, 0.5, 0.75, 1.0, 1.5):
        assert S.bin_index(running, med) != stop, running


def test_with_zero_point_closes_the_gap_to_the_first_positive_segment():
    closed = S.with_zero_point([S.make_segment(1.0, 1.0, True, True),
                                S.make_segment(1.0, 2.0, False, True)])
    assert closed[:2] == [S.make_segment(0.0, 0.0, True, True),
                          S.make_segment(0.0, 1.0, False, False)]
    # The open segment takes the complementary closure of the next segment's lower end.
    closed = S.with_zero_point([S.make_segment(0.0, 0.0, True, True),
                                S.make_segment(2.0, 5.0, False, True)])
    assert closed[1] == S.make_segment(0.0, 2.0, False, True)
    # Already contiguous: unchanged.
    tight = [S.make_segment(0.0, 0.0, True, True), S.make_segment(0.0, 3.0, False, True)]
    assert S.with_zero_point(tight) == tight


def test_decile_arm_forces_every_concept_to_quantile():
    import numpy as np

    fit = _fit_events({
        "potassium": ("labs", np.linspace(2.5, 6.5, 200)),
        "gcs_total": ("assessments", list(range(3, 16)) * 10),
    })
    segs, sources = _build_all(fit, bin_overrides={"scheme": "decile_ablation"})
    assert sources == {"potassium": "quantile", "gcs_total": "quantile"}
    assert segs["potassium"] != S.load_csv_segments(CSV, ["potassium"])["potassium"]


def test_targets_only_coverage_keeps_the_legacy_behavior():
    import numpy as np

    fit = _fit_events({
        "potassium": ("labs", np.linspace(2.5, 6.5, 200)),
        "map": ("vitals", np.linspace(40, 120, 200)),
    })
    segs, sources = _build_all(fit, bin_overrides={"coverage": "targets_only"})
    assert "potassium" not in segs
    assert sources["map"] == "csv"


def test_unknown_coverage_fails_closed():
    fit = _fit_events({"map": ("vitals", [70.0] * 30)})
    with pytest.raises(ValueError, match="coverage"):
        _build_all(fit, bin_overrides={"coverage": "most"})


def test_every_numeric_concept_gets_a_binning_source():
    import numpy as np

    rng = np.random.default_rng(4)
    fit = _fit_events({
        "potassium": ("labs", np.linspace(2.5, 6.5, 200)),
        "gcs_total": ("assessments", list(range(3, 16)) * 10),
        "synth_lab": ("labs", rng.lognormal(0.0, 0.6, 300)),
        "rare_lab": ("labs", [1.0, 2.0]),
        "heparin_u_hr": ("meds", [0.0] * 20 + list(rng.uniform(1, 9, 50))),
        "cam_total": ("assessments", [None] * 10),          # no numeric value -> not binned
    })
    segs, sources = _build_all(fit)
    numeric = {"potassium", "gcs_total", "synth_lab", "rare_lab", "heparin_u_hr"}
    assert numeric <= set(sources)
    assert "cam_total" not in sources and "cam_total" not in segs
    assert set(sources.values()) <= {"csv", "literature", "ordinal", "quantile", "single"}
    for concept in sources:
        S.validate_partition(segs[concept])


# ---- U2: fused categorical values -----------------------------------------------------------

def test_categorical_values_are_normalized_into_fused_tokens():
    from src.data.tokenize import categorical_token

    assert categorical_token("cam_total", "Negative") == "cam_total=negative"
    assert categorical_token("braden_mobility", "  Very Limited ") == "braden_mobility=very_limited"
    # A bare-integer category would collide with a bin index token; it is disambiguated.
    assert categorical_token("rass", "2") != "rass=2"


# ---- categorical small-cell floor (the vocabulary ships to every site) ----------------------

def _categorical_rows(spec: dict) -> pl.DataFrame:
    """``{(concept, raw value): [stay ids]}`` -> fit rows, three charted events per stay."""
    rows = {"hosp_id": [], "concept": [], "value": [], "cat_value": []}
    for (concept, raw), stays in spec.items():
        for stay in stays:
            for _ in range(3):
                rows["hosp_id"].append(stay)
                rows["concept"].append(concept)
                rows["value"].append(None)
                rows["cat_value"].append(raw)
    return pl.DataFrame(rows, schema={"hosp_id": pl.String, "concept": pl.String,
                                      "value": pl.Float64, "cat_value": pl.String})


def test_a_categorical_value_needs_ten_distinct_fit_stays_to_enter_the_vocabulary():
    """vocab.json is a bundle file shipped to every site: a value charted for fewer than
    `minimum_cell_size` patients must not leave the node as a verbatim token."""
    from src.data.tokenize import build_vocab

    stays = [f"s{i:02d}" for i in range(20)]
    fit = _categorical_rows({
        ("cam_total", "Rare Finding"): stays[:9],          # 9 stays (27 events) -> <unk>
        ("cam_total", "Negative"): stays[:10],             # 10 stays -> a token
        # Raw spellings normalizing to one token pool their stays: 5 + 5 = 10.
        ("position", "Prone"): stays[:5],
        ("position", " prone "): stays[5:10],
        ("sex", "Other"): stays[:9],                       # controlled categories too
        ("sex", "Female"): stays[:15],
    })
    vocab = build_vocab(fit, {})
    assert "cam_total=negative" in vocab
    assert "cam_total=rare_finding" not in vocab
    assert "position=prone" in vocab
    assert "sex=female" in vocab and "sex=other" not in vocab
    # The concepts themselves stay; a pruned value encodes as <unk>.
    assert {"cam_total", "position", "sex"} <= set(vocab)
    assert "cam_total=rare_finding" in build_vocab(fit, {}, min_category_stays=9)
