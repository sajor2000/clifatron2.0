"""Go/no-go design audit: export type, suppression over declared cells, stages (plan U13).

Synthetic data only. The cohort and labels come from `tests.fixtures_extubation` through
the real cohort builder and labeler; nothing here reads a real site.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest
import yaml

from src.eval import attestation as attest
from src.eval import schema
from src.eval.schema import DisclosureError

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "configs/extubation_benchmarks.yaml"
COHORT_CONFIG_PATH = ROOT / "configs/extubation.yaml"
PROTOCOL_HASH = hashlib.sha256(b"synthetic registered protocol").hexdigest()
FAST = ["--n-folds", "3", "--max-iter", "30", "--max-depth", "2"]


# ---------------------------------------------------------------------------
# helpers: a hand-built audit payload
# ---------------------------------------------------------------------------

def cell(n, *, table="arm_sizes", site="mimic", parents=(), dims=None):
    return {"table": table, "site": site, "dimensions": dict(dims or {}), "parents": list(parents), "n": n}


def payload(cells, partitions=(), *, stage="blind", results=None, **extra):
    return {
        "export_type": schema.AUDIT_EXPORT_TYPE, "schema_version": schema.AUDIT_SCHEMA_VERSION,
        "stage": stage, "sites": ["mimic"], "release_id": "audit-test-1",
        "disclosure_status": "pending_review", "generated_by": "test",
        "cells": cells, "partitions": [dict(p) for p in partitions], "results": results or {},
        "tables": {}, "stop_rules": {}, "triggers": {}, **extra,
    }


def released(cells):
    return {key for key, value in cells.items() if value["status"] == schema.EVALUABLE}


# ---------------------------------------------------------------------------
# suppression over declared cells (KTD10, R6)
# ---------------------------------------------------------------------------

def test_counts_under_ten_are_suppressed_and_carry_no_exact_count():
    out = schema.suppress_audit_cells({"a": cell(25), "b": cell(9), "c": cell(0)}, [])
    assert out["a"]["status"] == schema.EVALUABLE and out["a"]["n"] == 25
    for key in ("b", "c"):
        assert out[key]["status"] != schema.EVALUABLE
        assert "n" not in out[key] and out[key]["n_band"] == "<10"


def test_pooled_minus_a_public_site_cannot_recover_a_suppressed_site_cell():
    cells = {
        "pooled|all|eligible": cell(25, site="pooled"),
        "mimic|all|eligible": cell(20, parents=["pooled|all|eligible"]),
        "rush|all|eligible": cell(5, site="rush", parents=["pooled|all|eligible"]),
    }
    partitions = [{"total": "pooled|all|eligible", "parts": ["mimic|all|eligible", "rush|all|eligible"]}]
    out = schema.suppress_audit_cells(cells, partitions)
    assert "rush|all|eligible" not in released(out)
    # Pooled minus the site cell would give the suppressed cell back: one more is hidden.
    assert len({"pooled|all|eligible", "mimic|all|eligible"} & released(out)) <= 1
    schema.validate_audit_export(payload(out, partitions))


def test_validator_refuses_a_partition_with_a_single_hidden_member():
    cells = schema.suppress_audit_cells({"t": cell(25), "a": cell(20), "b": cell(5)}, [])
    partitions = [{"total": "t", "parts": ["a", "b"]}]
    # Suppressed without declaring the partition: the gate re-derives it and refuses.
    with pytest.raises(DisclosureError, match="differencing"):
        schema.validate_audit_export(payload(cells, partitions))


def test_parent_and_child_whose_difference_is_small_are_both_suppressed():
    cells = {"all": cell(30), "trial": cell(25, parents=["all"])}
    out = schema.suppress_audit_cells(cells, [])
    assert released(out) == set()
    assert "differencing" in out["trial"]["reason"] and "differencing" in out["all"]["reason"]
    schema.validate_audit_export(payload(out))


def test_identical_parent_and_child_and_large_differences_are_released():
    cells = {"all": cell(300), "casey": cell(300, parents=["all"]), "subset": cell(100, parents=["all"])}
    out = schema.suppress_audit_cells(cells, [])
    assert released(out) == {"all", "casey", "subset"}


def test_validator_refuses_a_released_pair_whose_difference_is_small():
    cells = {"all": dict(cell(30), status=schema.EVALUABLE), "trial": dict(cell(25, parents=["all"]), status=schema.EVALUABLE)}
    with pytest.raises(DisclosureError, match="differencing"):
        schema.validate_audit_export(payload(cells))


def test_validator_refuses_a_released_count_under_ten_and_a_suppressed_cell_with_n():
    small = {"a": dict(cell(7), status=schema.EVALUABLE)}
    with pytest.raises(DisclosureError, match="below"):
        schema.validate_audit_export(payload(small))
    leaky = {"a": dict(cell(7), status=schema.SMALL_CELL_SUPPRESSED)}
    with pytest.raises(DisclosureError, match="exact n"):
        schema.validate_audit_export(payload(leaky))


def test_complementary_suppression_cascades_across_overlapping_partitions():
    # unit total = arms; all-comer total = units. Hiding one arm in one unit must not
    # leave any partition with a single unknown.
    cells = {
        "all": cell(200),
        "u1": cell(120, parents=["all"]), "u2": cell(80, parents=["all"]),
        "u1|o": cell(100, parents=["u1"]), "u1|h": cell(15, parents=["u1"]), "u1|n": cell(5, parents=["u1"]),
        "u2|o": cell(40, parents=["u2"]), "u2|h": cell(20, parents=["u2"]), "u2|n": cell(20, parents=["u2"]),
    }
    partitions = [
        {"total": "all", "parts": ["u1", "u2"]},
        {"total": "u1", "parts": ["u1|o", "u1|h", "u1|n"]},
        {"total": "u2", "parts": ["u2|o", "u2|h", "u2|n"]},
    ]
    out = schema.suppress_audit_cells(cells, partitions)
    assert {"u1|n", "u1|h"} <= set(out) - released(out)
    schema.validate_audit_export(payload(out, partitions))


def test_results_on_a_suppressed_basis_carry_no_statistics():
    cells = schema.suppress_audit_cells({"t": cell(100), "h": cell(4, parents=["t"])}, [])
    results = {"r": {"kind": "feasibility", "basis": ["t", "h"], "status": schema.EVALUABLE,
                     "effective_sample_size": {"treated": 3.2}}}
    with pytest.raises(DisclosureError, match="basis"):
        schema.validate_audit_export(payload(cells, results=results))
    results = {"r": schema.finalize_audit_result(results["r"], cells)}
    assert results["r"]["status"] != schema.EVALUABLE and "effective_sample_size" not in results["r"]
    schema.validate_audit_export(payload(cells, results=results))


def test_shares_are_released_only_with_both_counts():
    cells = schema.suppress_audit_cells({
        "u": cell(100), "u|h": dict(cell(65, parents=["u"]), share_of="u"),
        "u|n": dict(cell(30, parents=["u"]), share_of="u"), "u|o": dict(cell(5, parents=["u"]), share_of="u"),
    }, [{"total": "u", "parts": ["u|h", "u|n", "u|o"]}])
    assert cells["u|h"]["share"] == 0.65
    assert "share" not in cells["u|o"] and "share" not in cells["u|n"]


# ---------------------------------------------------------------------------
# the audit export type and the untouched prediction export
# ---------------------------------------------------------------------------

def test_audit_payload_with_a_field_outside_the_allow_list_is_refused():
    base = payload(schema.suppress_audit_cells({"a": cell(50)}, []))
    schema.validate_audit_export(base)
    for where, mutate in (
        ("envelope", lambda p: p.update(debug_rows=[1, 2])),
        ("cell", lambda p: p["cells"]["a"].update(patient_count=3)),
        ("result", lambda p: p["results"].update(r={"kind": "x", "basis": ["a"], "status": "evaluable", "raw": 1})),
        ("bundle envelope", lambda p: p.update(model_bundle_id="bundle")),
    ):
        bad = copy.deepcopy(base)
        mutate(bad)
        with pytest.raises(DisclosureError):
            schema.validate_audit_export(bad)


def test_blind_payload_may_not_cross_outcome_with_arm():
    cells = schema.suppress_audit_cells({"a": cell(50, dims={"arm": "hfnc", "outcome": "reintubation"})}, [])
    with pytest.raises(DisclosureError, match="outcome"):
        schema.validate_audit_export(payload(cells))
    cells = schema.suppress_audit_cells({"a": cell(50, dims={"arm": "hfnc"})}, [])
    estimate = {"r": {"kind": "effect_estimate", "basis": ["a"], "status": "evaluable", "estimates": {}}}
    with pytest.raises(DisclosureError, match="outcome"):
        schema.validate_audit_export(payload(cells, results=estimate))
    schema.validate_audit_export(payload(cells, stage="unblinded", results=estimate,
                                         authorization={"basis": "pre_registration_audit"}))


def test_prediction_export_still_refuses_crosstab_keys_and_audit_payloads():
    prediction = {
        "schema_version": schema.METRIC_SCHEMA_VERSION, "metric_version": "m", "model_bundle_id": "b",
        "model_version": "v", "vocab_hash": "h", "outcome_spec_hash": "o", "clif_version": "2.1.0",
        "site_id": "s", "site_role": "development", "partition_role": "test", "outcomes": {
            "mort": {"status": "single_class", "label_validity": {
                "outcome_definition_id": "d", "outcome_definition_version": "1",
                "status_counts": {s: 0 for s in schema.U1_OUTCOME_STATES},
                "evaluable_denominator_fraction": 1.0},
                "subgroups": {}}},
        "disclosure_status": "pending_review", "release_id": "r1",
    }
    schema.validate_export(prediction)
    crosstab = copy.deepcopy(prediction)
    crosstab["outcomes"]["mort"] = {**crosstab["outcomes"]["mort"], "status": "evaluable",
                                    "metrics": {"n": 100, "prevalence": 0.3},
                                    "subgroups": {"sex_x_race": {}}}
    with pytest.raises(DisclosureError, match="crosstab"):
        schema.validate_export(crosstab)
    with pytest.raises(DisclosureError):
        schema.validate_export(payload(schema.suppress_audit_cells({"a": cell(50)}, [])))
    with pytest.raises(DisclosureError):
        schema.validate_audit_export(prediction)


# ---------------------------------------------------------------------------
# ledger (R6): every release adds entries; differencing across releases
# ---------------------------------------------------------------------------

def _release(p, ledger):
    with attest.ledger_lock(ledger):
        attest.check_cross_release_differencing(p, ledger)
        attest.append_to_ledger(p, ledger)
        attest.confirm_publication(p, ledger)


def test_audit_releases_are_recorded_in_the_ledger(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    p = payload(schema.suppress_audit_cells({"a": cell(50), "b": cell(3, parents=["a"])}, []))
    _release(p, ledger)
    entries = [e for e in attest.read_ledger(ledger) if "cell" in e]
    assert {e["cell"] for e in entries} == {"extubation_audit|a", "extubation_audit|b"}
    assert all("dimensions" not in e and "share" not in e for e in entries)
    suppressed = next(e for e in entries if e["cell"].endswith("|b"))
    assert suppressed["n"] is None


def test_a_child_released_later_cannot_difference_against_an_earlier_parent(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    _release(payload(schema.suppress_audit_cells({"all": cell(30)}, [])), ledger)
    later = payload(schema.suppress_audit_cells({"all": cell(30), "trial": cell(25, parents=["all"])}, []))
    later["cells"]["all"] = {**cell(30), "status": schema.EVALUABLE}
    later["cells"]["trial"] = {**cell(25, parents=["all"]), "status": schema.EVALUABLE}
    later["release_id"] = "audit-test-2"
    with pytest.raises(DisclosureError):
        _release(later, ledger)
    # Seeded with the ledger, suppression keeps the public parent and hides the child.
    prior_released, prior_suppressed = attest.prior_audit_cells(ledger)
    out = schema.suppress_audit_cells({"all": cell(30), "trial": cell(25, parents=["all"])}, [],
                                      prior_released=prior_released, prior_suppressed=prior_suppressed)
    assert released(out) == {"all"}
    _release(dict(payload(out), release_id="audit-test-3"), ledger)


def test_a_cell_suppressed_earlier_stays_suppressed(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    _release(payload(schema.suppress_audit_cells({"a": cell(5)}, [])), ledger)
    prior_released, prior_suppressed = attest.prior_audit_cells(ledger)
    out = schema.suppress_audit_cells({"a": cell(60)}, [], prior_released=prior_released,
                                      prior_suppressed=prior_suppressed)
    assert released(out) == set() and "prior release" in out["a"]["reason"]


# ---------------------------------------------------------------------------
# stages on the synthetic fixture
# ---------------------------------------------------------------------------

def _build_site(name, n_background, seed):
    from src.data.extubation_cohort import build_extubation_cohort, load_extubation_config, table_availability
    from src.eval.extubation_labeler import build_extubation_labels
    from tests.fixtures_extubation import PARTITIONS, SPLIT_SEED, build_extubation_fixture

    config = load_extubation_config(COHORT_CONFIG_PATH)
    data_config = yaml.safe_load((ROOT / "configs/data.yaml").read_text())
    cohort_config = yaml.safe_load((ROOT / "configs/cohort.yaml").read_text())
    availability = table_availability(config, data_config)
    fixture = build_extubation_fixture(n_background=n_background, seed=seed, outcome_scenarios=True)
    built = build_extubation_cohort(
        fixture.tables, config, availability=availability,
        split={"partitions": PARTITIONS, "split_seed": SPLIT_SEED}, site=name,
        episodes=fixture.episodes, calendar_periods=fixture.calendar_periods,
    )
    labels = build_extubation_labels(built.cohort, fixture.tables, config, availability=availability,
                                     data_config=data_config, cohort_config=cohort_config)
    return built.cohort, built.device_rows, labels


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    """Synthetic site artifacts laid out as on a node: PHI under output/intermediate_phi."""
    root = tmp_path_factory.mktemp("audit")
    phi = root / "output/intermediate_phi"
    phi.mkdir(parents=True)
    for name, n, seed in (("mimic", 3000, 7), ("rush", 1200, 11), ("uchicago", 150, 3)):
        cohort, devices, labels = _build_site(name, n, seed)
        cohort.write_parquet(phi / f"{name}_cohort.parquet")
        devices.write_parquet(phi / f"{name}_cohort_device_rows.parquet")
        labels.write_parquet(phi / f"{name}_cohort_labels.parquet")
    return root


def audit(workdir, *args):
    from src.eval import extubation_audit

    cwd = os.getcwd()
    os.chdir(workdir)
    try:
        return extubation_audit.main([*args, "--registry", str(REGISTRY_PATH),
                                      "--cohort-config", str(COHORT_CONFIG_PATH),
                                      "--audit-config", str(ROOT / "configs/extubation_audit.yaml")])
    finally:
        os.chdir(cwd)


def cohort_arg(name):
    return ["--cohort", f"{name}=output/intermediate_phi/{name}_cohort.parquet"]


def ledger_releases(workdir):
    ledger = workdir / "output/intermediate_phi/extubation_audit_ledger.jsonl"
    return {e["confirm_release_id"] for e in attest.read_ledger(ledger) if "confirm_release_id" in e}


def test_blind_stage_reports_aggregates_with_no_outcome_by_arm_cell(workdir):
    report = audit(workdir, "blind", *cohort_arg("mimic"), "--release-id", "blind-mimic-1", *FAST)
    schema.validate_audit_export(report)
    assert report["stage"] == "blind"
    for value in report["cells"].values():
        assert not set(value["dimensions"]) & schema.AUDIT_OUTCOME_DIMENSIONS
    for value in report["results"].values():
        assert value["kind"] not in schema.AUDIT_OUTCOME_RESULT_KINDS
        assert not set(value) & schema.AUDIT_OUTCOME_RESULT_FIELDS
    # Arm sizes per partition and for the held-out union (KTD6).
    scopes = {v["dimensions"].get("scope") for v in report["cells"].values()}
    assert {"all", "held_out", "train", "validation", "calibration", "internal_test"} <= scopes
    kinds = {v["kind"] for v in report["results"].values()}
    assert {"feasibility", "covariate_coverage", "device_choice"} <= kinds
    feasibility = [v for v in report["results"].values() if v["kind"] == "feasibility"]
    assert {v["trial_id"] for v in feasibility} >= {"casey_2021_all_comers", "hernandez_2016_low_risk"}
    released_feasibility = [v for v in feasibility if v["status"] == schema.EVALUABLE]
    assert released_feasibility and all(
        v["baseline_risk_source"] == "trial_published_control_risk" for v in released_feasibility)
    # MIMIC declares a period source; the fixture's period table is evaluable.
    assert report["tables"]["mimic|period"]["status"] == "evaluable"
    assert report["tables"]["mimic|unit"]["status"] == "evaluable"
    out = workdir / "output/final_no_phi"
    assert (out / "extubation_audit_blind_mimic.json").exists()
    assert (out / "extubation_audit_blind_mimic.csv").exists()
    assert "blind-mimic-1" in ledger_releases(workdir)


def test_blind_stage_never_opens_or_imports_the_labels(workdir):
    script = (
        "import sys, builtins, polars as pl\n"
        "opened = []\n"
        "real_open, real_read, real_scan = builtins.open, pl.read_parquet, pl.scan_parquet\n"
        "builtins.open = lambda f, *a, **k: (opened.append(str(f)), real_open(f, *a, **k))[1]\n"
        "pl.read_parquet = lambda f, *a, **k: (opened.append(str(f)), real_read(f, *a, **k))[1]\n"
        "pl.scan_parquet = lambda f, *a, **k: (opened.append(str(f)), real_scan(f, *a, **k))[1]\n"
        "from src.eval import extubation_audit as a\n"
        f"a.main(['blind', '--cohort', 'mimic=output/intermediate_phi/mimic_cohort.parquet', "
        f"'--release-id', 'blind-guard', '--registry', {str(REGISTRY_PATH)!r}, "
        f"'--cohort-config', {str(COHORT_CONFIG_PATH)!r}, "
        f"'--audit-config', {str(ROOT / 'configs/extubation_audit.yaml')!r}, "
        "'--n-folds', '3', '--max-iter', '20', '--max-depth', '2'])\n"
        "assert 'src.eval.extubation_labeler' not in sys.modules, 'labeler imported'\n"
        "assert not [f for f in opened if 'labels' in f], [f for f in opened if 'labels' in f]\n"
        "assert any('mimic_cohort.parquet' in f for f in opened)\n"
    )
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    done = subprocess.run([sys.executable, "-c", script], cwd=workdir, env=env, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-2000:]


def test_period_table_without_a_declared_source_is_not_evaluable(workdir):
    # The rush cohort was built with the fixture's period table available, but the
    # contract declares no period source for rush, and event dates are never used.
    cohort = pl.read_parquet(workdir / "output/intermediate_phi/rush_cohort.parquet")
    assert cohort["calendar_period"].is_null().all() and cohort["time_zero_dttm"].is_not_null().all()
    report = audit(workdir, "blind", *cohort_arg("rush"), "--release-id", "blind-rush-1", *FAST)
    table = report["tables"]["rush|period"]
    assert table["status"] == "not_evaluable" and "no calendar period source" in table["reason"]
    assert not [k for k, v in report["cells"].items() if v["table"] == "first_device_by_period"]
    assert report["triggers"]["adoption_era_period"]["status"] == "not_evaluable"


def test_a_stop_rule_that_fires_is_written_into_the_report(workdir):
    report = audit(workdir, "blind", *cohort_arg("uchicago"), "--release-id", "blind-small-1", *FAST)
    rule = report["stop_rules"]["precision"]
    assert rule["status"] == "evaluated" and rule["fired"] is True
    on_disk = json.loads((workdir / "output/final_no_phi/extubation_audit_blind_uchicago.json").read_text())
    assert on_disk["stop_rules"]["precision"]["fired"] is True
    table = (workdir / "output/final_no_phi/extubation_audit_blind_uchicago.csv").read_text()
    assert "precision" in table and "FIRED" in table
    assert report["stop_rules"]["harmful_side"]["status"] == "not_evaluated"
    assert report["stop_rules"]["negative_controls"]["status"] == "not_evaluated"


def test_harmful_side_stop_rule_fires_on_an_interval_above_one():
    from src.eval import extubation_audit

    config = yaml.safe_load((ROOT / "configs/extubation_audit.yaml").read_text())
    rule = extubation_audit.harmful_side_rule({"lower": 1.3, "estimate": 1.8, "upper": 2.4}, config)
    assert rule["fired"] is True and rule["status"] == "evaluated"
    assert extubation_audit.harmful_side_rule({"lower": 0.8, "estimate": 1.2, "upper": 1.7}, config)["fired"] is False


def test_unblinded_refuses_every_comparison_but_casey_without_a_protocol_hash(workdir):
    from src.eval.causal.emulate import EmulationRefused

    with pytest.raises(EmulationRefused, match="protocol"):
        audit(workdir, "unblinded", *cohort_arg("mimic"), "--trial", "hernandez_2016_low_risk", *FAST)


def test_unblinded_refuses_rush_until_the_freeze_hashes_are_recorded(workdir, tmp_path):
    from src.eval.causal.emulate import EmulationRefused

    with pytest.raises(EmulationRefused):
        audit(workdir, "unblinded", *cohort_arg("rush"), *FAST)
    with pytest.raises(EmulationRefused, match="freeze"):
        audit(workdir, "unblinded", *cohort_arg("rush"), "--protocol-hash", PROTOCOL_HASH, *FAST)
    # Never opened: the refusal comes before any label is read.
    assert not list((workdir / "output/final_no_phi").glob("extubation_audit_unblinded_rush*"))


def test_blind_then_simulate_then_casey_end_to_end_through_the_cli(workdir):
    before = ledger_releases(workdir)
    audit(workdir, "blind", *cohort_arg("mimic"), "--release-id", "e2e-blind", *FAST)
    sim = audit(workdir, "simulate", *cohort_arg("mimic"), "--release-id", "e2e-sim", "--n-reps", "3",
                "--n-sim", "600", *FAST)
    schema.validate_audit_export(sim)
    runs = [v for v in sim["results"].values() if v["kind"] == "simulation"]
    ran = [v for v in runs if v["status"] == schema.EVALUABLE]
    assert ran and all(v["baseline_risk_source"] == "trial_published_control_risk" for v in ran)
    assert all(v["evaluable"] is False for v in runs if v["status"] != schema.EVALUABLE)
    for value in ran:
        assert set(value["pass_rates"]) == {"trial", "zero", "reversed"}
    labels = "output/intermediate_phi/mimic_cohort_labels.parquet"
    casey = audit(workdir, "unblinded", *cohort_arg("mimic"), "--labels", f"mimic={labels}",
                  "--release-id", "e2e-casey", "--n-boot", "20", *FAST)
    schema.validate_audit_export(casey)
    assert casey["authorization"]["basis"] == "pre_registration_audit"
    estimate = [v for v in casey["results"].values() if v["kind"] == "effect_estimate"]
    assert len(estimate) == 1 and estimate[0]["trial_id"] == "casey_2021_all_comers"
    assert casey["stop_rules"]["harmful_side"]["status"] in ("evaluated", "not_evaluable")
    assert {"e2e-blind", "e2e-sim", "e2e-casey"} <= ledger_releases(workdir) - before
    # Cells shared across stages carry the same id and count: the ledger saw no delta.
    blind = json.loads((workdir / "output/final_no_phi/extubation_audit_blind_mimic.json").read_text())
    shared = set(blind["cells"]) & set(casey["cells"])
    assert shared and all(blind["cells"][k].get("n") == casey["cells"][k].get("n") for k in shared)
