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
    assert S.POLICY_VERSION == 1
    assert json.loads(json.dumps(segs)) == segs
