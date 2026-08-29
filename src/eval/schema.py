"""Allow-listed aggregate result schema for site-local evaluation export (U5).

Every artifact that leaves a site passes through `validate_export`. The contract is an
ALLOW-LIST, not a deny-list: a field that is not named below cannot be exported, so a
new debug value, a captured error string, or a fresh provenance key fails closed instead
of travelling. A recursive value scan and a named-substring deny-list run behind it as
independent second checks.

Validation runs at the WRITER, not only the reader. Development sites (MIMIC, Rush,
UChicago) export through `src/eval/clif_validate.py` rather than the shippable wheel, so
the sites holding real PHI must hit the same gate the aggregator does.

**The gate enforces suppression itself.** An earlier version computed suppression at the
producers (`subgroup_panel`, `evaluate_site`) and checked only field NAMES here, so a
payload assembled by any other route could export an n=1 cell through a boundary
documented as fail-closed. `validate_export` now re-derives `suppress_cell` for every
evaluable cell and rejects anything that would not have survived it. A boundary that
trusts its callers is not a boundary.

Three disclosure controls, not interchangeable:

  1. Denominator suppression  — a cell with n < MIN_CELL_SIZE is suppressed.
  2. Numerator suppression    — a cell whose positive OR negative count is below the
                                same threshold is suppressed even when n clears it,
                                because `n * prevalence` recovers the positive count.
  3. Resolution bounding      — a cell below CURVE_RELEASE_MIN reports scalar summaries
                                only. A released DCA curve is 50 equations in TP and FP;
                                with n and prevalence also released those invert to
                                per-patient counts at 50 cut-points.

Per-outcome label-validity reporting is REQUIRED. The allow-list closes the disclosure
channel; without this block it would close the validity channel too, leaving a site with
mis-mapped units or a differently-coded mCIDE concept returning a plausible AUROC that
nothing in the payload can contradict. Fields follow TRIPOD+AI's participants / outcome /
missing-data reporting (Collins et al., BMJ 2024;385:e078378).
"""

from __future__ import annotations

import functools
import math
import re
from pathlib import Path

import yaml

DEFAULT_ARTIFACT_POLICY = Path(__file__).resolve().parents[2] / "configs" / "artifact_policy.yaml"

METRIC_SCHEMA_VERSION = "u5.1"


class DisclosureError(ValueError):
    """Raised when an artifact would export something the policy forbids.

    Deliberately a hard failure: there is no "export it anyway with a warning" path,
    because a warning in a site-local batch process is a line in a log nobody reads.
    """


# ---------------------------------------------------------------- outcome status
EVALUABLE = "evaluable"
UNSUPPORTED_AT_SITE = "unsupported_at_site"
SINGLE_CLASS = "single_class"
INSUFFICIENT_N = "insufficient_n"
INSUFFICIENT_PARTITIONS = "insufficient_partitions"
SMALL_CELL_SUPPRESSED = "small_cell_suppressed"
ARTIFACT_MISMATCH = "artifact_mismatch"
RUNTIME_FAILURE = "runtime_failure"
# The model could not score enough of the ascertained cohort for the metric to mean
# anything — the frozen vocabulary did not transfer to this site (the PORTER failure
# mode). Distinct from INSUFFICIENT_N: a small site and a site where 99% of stays
# failed to tokenize must not report the same status, or a coverage failure hides as
# "too few patients" while the dropped majority vanishes from the artifact entirely.
COVERAGE_INSUFFICIENT = "coverage_insufficient"

OUTCOME_STATUSES = frozenset({
    EVALUABLE, UNSUPPORTED_AT_SITE, SINGLE_CLASS, INSUFFICIENT_N,
    INSUFFICIENT_PARTITIONS, SMALL_CELL_SUPPRESSED, ARTIFACT_MISMATCH, RUNTIME_FAILURE,
    COVERAGE_INSUFFICIENT,
})

NON_EVALUABLE_STATUSES = OUTCOME_STATUSES - {EVALUABLE}

SITE_ROLES = frozenset({"reference", "development", "external_confirmation"})
PARTITION_ROLES = frozenset({"train", "validation", "calibration", "test"})

# Closed set, and only one value may be written to disk. `pending_review` is a LOCAL
# DRAFT state: `write_export` refuses it, so a release boundary cannot be crossed without
# a recorded disclosure decision (review finding #16).
DRAFT_DISCLOSURE_STATUS = "pending_review"
RELEASABLE_DISCLOSURE_STATUSES = frozenset({"reviewed_approved"})
DISCLOSURE_STATUSES = RELEASABLE_DISCLOSURE_STATUSES | {DRAFT_DISCLOSURE_STATUS}


# ---------------------------------------------------------------- policy constants
# A deployed validator pins its policy from the BUNDLE, not the package: the default
# path above is repo-relative and dangles inside a wheel, and deep callers reach the
# policy through the cached `min_cell_size()` accessor with no parameter. This env var
# is the seam bundle loading uses; setting it must be followed by
# `min_cell_size.cache_clear()` so the pinned policy actually takes effect.
POLICY_OVERRIDE_ENV = "CLIF_ARTIFACT_POLICY_FILE"


def _resolved_policy_path() -> Path:
    import os
    override = os.environ.get(POLICY_OVERRIDE_ENV)
    return Path(override) if override else DEFAULT_ARTIFACT_POLICY


def load_min_cell_size(policy_path: str | Path | None = None) -> int:
    """Read the suppression threshold from the landed artifact policy.

    `policy_path=None` resolves the CLIF_ARTIFACT_POLICY_FILE override first, then the
    repo default. An override naming a missing file fails closed rather than silently
    falling back — a bundle that pinned a policy must never be served the package's.

    Single source of truth. `configs/artifact_policy.yaml` set `minimum_cell_size: 10`
    in U1 and `tests/test_artifact_policy.py` asserts it; re-deciding the number here is
    how a codebase ends up with two thresholds and a suppression rule that silently stops
    applying (which is exactly what `subgroup_panel`'s hard-coded 30 was).
    """
    if policy_path is None:
        policy_path = _resolved_policy_path()
    policy = yaml.safe_load(Path(policy_path).read_text())
    try:
        value = policy["classes"]["aggregate_no_phi"]["minimum_cell_size"]
    except (KeyError, TypeError) as exc:
        raise DisclosureError(
            f"artifact policy at {policy_path} does not declare "
            "classes.aggregate_no_phi.minimum_cell_size; refusing to guess a "
            "suppression threshold"
        ) from exc
    if not isinstance(value, int) or value < 1:
        raise DisclosureError(f"minimum_cell_size must be a positive integer, got {value!r}")
    return value


@functools.lru_cache(maxsize=1)
def min_cell_size() -> int:
    """Cached accessor. Deferred rather than loaded at import (review finding #32).

    Reading a YAML file as an import side effect meant importing any eval module failed
    when the policy file was absent or malformed, which is both surprising and
    inconsistent with every other config load in this repo.
    """
    return load_min_cell_size()


def load_max_dropped_fraction(policy_path: str | Path | None = None) -> float:
    """Read the coverage gate from the landed artifact policy (fail-closed).

    The maximum fraction of an outcome's ascertained cohort that may be dropped as
    untokenizable before the outcome is reported COVERAGE_INSUFFICIENT rather than
    scored on the surviving sliver. Like `minimum_cell_size`, it is a declared
    disclosure/validity decision, not a default this module may guess: a policy that
    does not declare it fails closed, so a governed bundle cannot silently ship without
    a coverage threshold. Resolves the CLIF_ARTIFACT_POLICY_FILE override first.
    """
    if policy_path is None:
        policy_path = _resolved_policy_path()
    policy = yaml.safe_load(Path(policy_path).read_text())
    try:
        value = policy["classes"]["aggregate_no_phi"]["max_dropped_fraction"]
    except (KeyError, TypeError) as exc:
        raise DisclosureError(
            f"artifact policy at {policy_path} does not declare "
            "classes.aggregate_no_phi.max_dropped_fraction; refusing to guess a "
            "coverage threshold. A validation run that cannot say how much of the "
            "cohort it dropped is not evaluable."
        ) from exc
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 < value <= 1:
        raise DisclosureError(
            f"max_dropped_fraction must be a number in (0, 1], got {value!r}"
        )
    return float(value)


@functools.lru_cache(maxsize=1)
def max_dropped_fraction() -> float:
    """Cached accessor for the coverage gate; cleared by the same bundle policy pin."""
    return load_max_dropped_fraction()


def band_dropped_count(n_dropped: int) -> int | str:
    """A dropped-stay count is releasable exactly only when it hides no small cell.

    0 (nobody dropped) and counts at or above the floor are safe to state precisely.
    A count in (0, floor) is an exact small-subgroup size — the stays whose data
    produced no token sequence, plausibly correlated with care setting or data pathway
    — and releasing it beside the released metrics is the same numerator disclosure
    `suppress_cell` refuses. Band it to `<floor`, matching `banded_status_counts`.
    """
    floor = min_cell_size()
    if n_dropped == 0 or n_dropped >= floor:
        return n_dropped
    return f"<{floor}"


def curve_release_min() -> int:
    """Threshold above which a cell may release its DCA / calibration curves.

    5x the cell minimum. A cell can clear the count threshold and still be inverted
    through its own curves, so curve release needs its own, higher bar: `net_benefit`
    emits 50 threshold points, and at n >= 50 that system is underdetermined for
    per-patient recovery. The multiplier is a judgment call, recorded here rather than
    left as a bare literal.
    """
    return 5 * min_cell_size()


def prevalence_step() -> float:
    """Quantisation step for exported prevalence. Coarsening, NOT a guarantee.

    **Read this before relying on it.** A previous version rounded to 2 decimals and
    claimed that left the positive count ambiguous. It did not: for every n from 20 to
    100, `round(pos/n, 2)` admits exactly one candidate integer. The count was fully
    recoverable and the reassuring comment was false — which is worse than no comment,
    because a reviewer reading it would stop looking.

    Quantising here does not fix that, and this docstring will not claim it does. For any
    n where 1/n <= step, distinct counts still map to distinct exported values. Making
    prevalence genuinely ambiguous requires exporting an interval rather than a point,
    which is a reporting-format decision this unit does not take.

    **The numerator guarantee is `suppress_cell`, not this function.** A released cell
    has at least MIN_CELL_SIZE positives and at least MIN_CELL_SIZE negatives, so
    recovering the exact positive count from `n * prevalence` identifies a group of at
    least MIN_CELL_SIZE patients — never an individual. That is the property the
    disclosure argument actually rests on. This step only reduces the precision of what
    is published; treat it as defence in depth.
    """
    return 1.0 / (2 * min_cell_size())


# ---------------------------------------------------------------- deny-list
# Second line of defence behind the allow-list. Substring match, case-insensitive.
# Free-text LOG redaction uses its own separate list in `src/eval/log_sanitizer.py`:
# these two matched with different semantics and must be free to diverge (finding #33).
PROHIBITED_SUBSTRINGS = (
    "patient_id", "patientid", "hospitalization_id", "hosp_id", "encounter_id",
    "subject_id", "stay_id", "mrn", "sequence", "token", "pos_min",
    "charttime", "storetime", "timestamp", "birth", "dob", "name", "address",
)

_PATH_MARKERS = ("/", "\\")

# A signature is 64 lowercase hex characters (HMAC-SHA256). Checked by FORMAT rather
# than exempted from checking: the previous `_OPAQUE_FIELDS` escape hatch existed to stop
# base64 tripping the path heuristic, but the value is hex and never needed it, and an
# allow-listed field exempt from all inspection is a hole (finding #28).
_SIGNATURE_RE = re.compile(r"\A[0-9a-f]{64}\Z")

# Subgroup keys must name a single attribute, never a crosstab. A joint cell
# (`sex_x_race`) is what makes cross-attribute differencing possible, so the joint is
# refused at the boundary rather than reasoned about (finding #26).
_CROSSTAB_MARKERS = ("_x_", "*", "×", " by ", "__")


# ---------------------------------------------------------------- allow-list
ENVELOPE_FIELDS = frozenset({
    "schema_version", "metric_version", "model_bundle_id", "model_version",
    "vocab_hash", "outcome_spec_hash", "target_map_hash", "clif_version",
    "site_id", "site_role", "partition_role", "generated_by", "outcomes",
    "disclosure_status", "release_id", "signature",
})

REQUIRED_ENVELOPE_FIELDS = frozenset({
    "schema_version", "metric_version", "model_bundle_id", "model_version",
    "vocab_hash", "outcome_spec_hash", "clif_version", "site_id", "site_role",
    "partition_role", "outcomes", "disclosure_status", "release_id",
})

OUTCOME_FIELDS = frozenset({
    "status", "reason", "metrics", "curves", "subgroups", "label_validity", "intervals",
})
REQUIRED_OUTCOME_FIELDS = frozenset({"status", "label_validity"})

METRIC_FIELDS = frozenset({
    "auroc", "auprc", "n", "prevalence", "ece", "brier", "calib_slope",
    "calib_intercept", "ici", "temperature", "n_dropped_nan",
    "cr_d_calibration", "aj_k_calibration", "iec",
})

CURVE_FIELDS = frozenset({
    "dca_thresholds", "dca_model", "dca_treat_all", "dca_treat_none",
    "calibration_bin_edges", "calibration_bin_observed", "calibration_bin_expected",
    "cr_d_calibration_bins",
})

LABEL_VALIDITY_FIELDS = frozenset({
    "outcome_definition_id", "outcome_definition_version", "status_counts",
    "evaluable_denominator_fraction", "measurement_density",
})
REQUIRED_LABEL_VALIDITY_FIELDS = frozenset({
    "outcome_definition_id", "outcome_definition_version",
    "status_counts", "evaluable_denominator_fraction",
})

U1_OUTCOME_STATES = (
    "positive", "negative", "censored", "competing_event",
    "prevalent", "not_ascertainable", "unsupported_at_site",
)

# A suppressed cell carries NO n (finding #3) — the count is the thing suppression
# exists to hide. `n_band` carries a coarse size signal instead when one is needed.
CELL_FIELDS = frozenset({"status", "reason", "n", "n_band", "prevalence",
                         "auroc", "auprc", "ece"})


# ---------------------------------------------------------------- suppression
def suppress_cell(n: int, n_positive: int) -> tuple[str, str | None]:
    """Decide a cell's disclosure status from its denominator AND numerator.

    Returns `(status, reason)`. A cell is releasable only when the total, the positive
    count, and the negative count all clear MIN_CELL_SIZE. Checking `n` alone is not
    suppression: a cell of n=12 at prevalence 0.0833 clears any size threshold while
    identifying exactly one outcome-positive patient.
    """
    floor = min_cell_size()
    if n < floor:
        return INSUFFICIENT_N, f"n < {floor}"
    n_negative = n - n_positive
    if n_positive == 0 or n_negative == 0:
        return SINGLE_CLASS, "outcome has a single class in this cell"
    if n_positive < floor or n_negative < floor:
        return SMALL_CELL_SUPPRESSED, (
            f"positive or negative count < {floor}; releasing n and prevalence would "
            "recover the exact event count"
        )
    return EVALUABLE, None


def n_band(n: int) -> str:
    """Coarse size signal for a suppressed cell, replacing its exact count."""
    floor = min_cell_size()
    return f"<{floor}" if n < floor else f">={floor}"


def round_prevalence(prevalence: float | None) -> float | None:
    """Quantise exported prevalence so at least two integer counts are consistent.

    Returns None — never NaN — for an undefined value, so the exported artifact stays
    valid JSON under a strict parser (`allow_nan=False`). See `prevalence_step` for why
    the step is what it is.
    """
    if prevalence is None:
        return None
    value = float(prevalence)
    if math.isnan(value) or math.isinf(value):
        return None
    step = prevalence_step()
    return round(round(value / step) * step, 6)


def curves_releasable(n: int) -> bool:
    """Whether a cell of this size may release its DCA / calibration curves."""
    return n >= curve_release_min()


def apply_complementary_suppression(cells: dict[str, dict]) -> dict[str, dict]:
    """Suppress additional cells when one suppressed cell is recoverable by differencing.

    Suppressing a single cell inside an attribute is not enough: with the sibling cells
    and the attribute total both released, the hidden value is a subtraction away. When
    exactly one cell in an attribute is suppressed, the next-smallest releasable cell is
    suppressed alongside it so at least two unknowns remain.

    Cross-ATTRIBUTE differencing is handled structurally rather than arithmetically: the
    schema refuses joint/crosstab subgroup keys (`_CROSSTAB_MARKERS`), so marginals over
    the same rows never combine into a joint cell. Two attributes' marginals alone do not
    determine any joint count.

    Operates on already-status-tagged cells and returns a new mapping. Suppressed cells
    lose their exact `n` and keep only a band.
    """
    out = {k: dict(v) for k, v in cells.items()}
    suppressed = [k for k, v in out.items() if v.get("status") in NON_EVALUABLE_STATUSES]
    releasable = [k for k, v in out.items() if v.get("status") == EVALUABLE]
    if len(suppressed) == 1 and releasable:
        victim = min(releasable, key=lambda k: (out[k].get("n", 0), k))
        out[victim] = {
            "status": SMALL_CELL_SUPPRESSED,
            "reason": "complementary suppression: a single suppressed sibling would be "
                      "recoverable by differencing against the attribute total",
            "n_band": n_band(out[victim].get("n", 0)),
        }
    # Strip exact counts from every suppressed cell (finding #3).
    for key, cell in out.items():
        if cell.get("status") in NON_EVALUABLE_STATUSES and "n" in cell:
            cell["n_band"] = n_band(cell.pop("n"))
    return out


# ---------------------------------------------------------------- validation
def _looks_like_path(value: object) -> bool:
    if not isinstance(value, str):
        return False
    if value.startswith("~"):
        return True
    return any(marker in value for marker in _PATH_MARKERS)


def _check_key(key: object, allowed: frozenset[str], where: str) -> None:
    if not isinstance(key, str):
        raise DisclosureError(f"{where}: key {key!r} is not a string")
    if key not in allowed:
        raise DisclosureError(
            f"{where}: field {key!r} is not in the allow-list. Add it to "
            f"src/eval/schema.py deliberately, or do not export it."
        )
    lowered = key.lower()
    for banned in PROHIBITED_SUBSTRINGS:
        if banned in lowered:
            raise DisclosureError(f"{where}: field {key!r} matches prohibited pattern {banned!r}")


def _check_dynamic_key(key: object, where: str) -> None:
    """Validate an allow-list-exempt key: an outcome, attribute, or category name."""
    if not isinstance(key, str) or not key:
        raise DisclosureError(f"{where}: key {key!r} must be a non-empty string")
    lowered = key.lower()
    for banned in PROHIBITED_SUBSTRINGS:
        if banned in lowered:
            raise DisclosureError(f"{where}: key {key!r} matches prohibited pattern {banned!r}")
    if _looks_like_path(key):
        raise DisclosureError(f"{where}: key {key!r} looks like a path")


def _check_value(key: str, value: object, where: str, _depth: int = 0) -> None:
    """Recursively validate an exported value.

    Scalars, and every scalar reachable inside a list or mapping. The previous version
    inspected only top-level strings, so identifiers, paths, or row-level records placed
    inside `curves`, `intervals`, `measurement_density`, or `status_counts` travelled
    untouched (review finding #20).
    """
    if _depth > 6:
        raise DisclosureError(f"{where}: value of {key!r} nests deeper than the export contract allows")

    if isinstance(value, dict):
        for sub_key, sub_value in value.items():
            _check_dynamic_key(sub_key, f"{where}.{key}")
            _check_value(str(sub_key), sub_value, f"{where}.{key}", _depth + 1)
        return
    if isinstance(value, (list, tuple)):
        if len(value) > 512:
            raise DisclosureError(
                f"{where}: {key!r} has {len(value)} elements; an array that long is a "
                "row-level export, not an aggregate"
            )
        for item in value:
            _check_value(key, item, where, _depth + 1)
        return
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            raise DisclosureError(
                f"{where}: {key!r} is {value!r}. Non-finite floats serialize as bare "
                "NaN/Infinity, which is not valid JSON and cannot be verified by a "
                "non-Python consumer. Emit null instead."
            )
        return
    if not isinstance(value, str):
        raise DisclosureError(f"{where}: {key!r} has unsupported type {type(value).__name__}")

    if _looks_like_path(value):
        raise DisclosureError(
            f"{where}: value of {key!r} looks like a local filesystem path ({value!r}); "
            "site directory layout must not leave the node"
        )
    lowered = value.lower()
    for banned in PROHIBITED_SUBSTRINGS:
        if banned in lowered:
            raise DisclosureError(
                f"{where}: value of {key!r} contains prohibited content {banned!r}"
            )


def _validate_label_validity(block: object, outcome: str) -> None:
    where = f"outcomes.{outcome}.label_validity"
    if not isinstance(block, dict):
        raise DisclosureError(
            f"{where}: required and must be a mapping. A report without per-outcome "
            "label-validity diagnostics is non-evaluable: nothing in the payload could "
            "contradict a plausible AUROC computed from mis-mapped labels."
        )
    for key, value in block.items():
        _check_key(key, LABEL_VALIDITY_FIELDS, where)
        _check_value(key, value, where)
    missing = REQUIRED_LABEL_VALIDITY_FIELDS - set(block)
    if missing:
        raise DisclosureError(f"{where}: missing required fields {sorted(missing)}")

    counts = block["status_counts"]
    if not isinstance(counts, dict):
        raise DisclosureError(f"{where}.status_counts: must be a mapping of state -> count")
    unknown = sorted(set(counts) - set(U1_OUTCOME_STATES))
    if unknown:
        raise DisclosureError(f"{where}.status_counts: unknown outcome states {unknown}")
    missing_states = sorted(set(U1_OUTCOME_STATES) - set(counts))
    if missing_states:
        raise DisclosureError(
            f"{where}.status_counts: must account for all seven U1 outcome states; "
            f"missing {missing_states}"
        )
    # Exact small counts are as disclosive as an exact small cell (finding #15).
    floor = min_cell_size()
    for state, count in counts.items():
        if isinstance(count, int) and 0 < count < floor:
            raise DisclosureError(
                f"{where}.status_counts.{state}: exact count {count} is below the "
                f"suppression floor ({floor}). Export a band such as '<{floor}', not the "
                "count itself."
            )


def _validate_cell(cell: object, where: str) -> None:
    if not isinstance(cell, dict):
        raise DisclosureError(f"{where}: must be a mapping")
    for key, value in cell.items():
        _check_key(key, CELL_FIELDS, where)
        _check_value(key, value, where)
    status = cell.get("status")
    if status not in OUTCOME_STATUSES:
        raise DisclosureError(f"{where}.status: {status!r} is not a recognised status")
    if status != EVALUABLE:
        if "n" in cell:
            raise DisclosureError(
                f"{where}: suppressed cell carries an exact n. The count is what "
                "suppression exists to hide; export n_band instead."
            )
        return
    _enforce_suppression(cell, where)


def _enforce_suppression(block: dict, where: str) -> None:
    """Re-derive suppression at the gate and reject anything that would not survive it.

    This is the boundary's own check, not a restatement of the producer's. Computing
    suppression in `subgroup_panel` and `evaluate_site` and then trusting the result here
    meant any other assembly route could export an unsuppressed cell through a gate
    documented as fail-closed (review finding #5).
    """
    n = block.get("n")
    if not isinstance(n, int):
        raise DisclosureError(
            f"{where}: an evaluable cell must carry an integer n so the gate can verify "
            f"suppression; got {n!r}"
        )
    prevalence = block.get("prevalence")
    if prevalence is None:
        raise DisclosureError(f"{where}: an evaluable cell must carry prevalence")
    n_positive = round(n * float(prevalence))
    status, reason = suppress_cell(n, n_positive)
    if status != EVALUABLE:
        raise DisclosureError(
            f"{where}: cell is marked evaluable but would be suppressed ({status}: "
            f"{reason}). n={n}, implied positive count={n_positive}."
        )


def _validate_outcome(name: str, block: object) -> None:
    where = f"outcomes.{name}"
    _check_dynamic_key(name, "outcomes")
    if not isinstance(block, dict):
        raise DisclosureError(f"{where}: must be a mapping")
    for key in block:
        _check_key(key, OUTCOME_FIELDS, where)
    missing = REQUIRED_OUTCOME_FIELDS - set(block)
    if missing:
        raise DisclosureError(f"{where}: missing required fields {sorted(missing)}")

    status = block["status"]
    if status not in OUTCOME_STATUSES:
        raise DisclosureError(
            f"{where}.status: {status!r} is not one of {sorted(OUTCOME_STATUSES)}"
        )
    if "reason" in block:
        _check_value("reason", block["reason"], where)

    _validate_label_validity(block["label_validity"], name)

    if status != EVALUABLE:
        if "metrics" in block or "curves" in block:
            raise DisclosureError(
                f"{where}: status is {status!r} but a metrics or curves block is present. "
                "A non-evaluable outcome must not carry numbers that read as scores."
            )
        return

    metrics = block.get("metrics")
    if not isinstance(metrics, dict):
        raise DisclosureError(f"{where}.metrics: required when status is {EVALUABLE!r}")
    for key, value in metrics.items():
        _check_key(key, METRIC_FIELDS, f"{where}.metrics")
        _check_value(key, value, f"{where}.metrics")

    _enforce_suppression(metrics, f"{where}.metrics")

    if "curves" in block:
        # Fail CLOSED: an absent or non-integer n must block curve release, not skip the
        # check. The guard used to sit on the wrong side of the condition (finding #6).
        n = metrics.get("n")
        if not isinstance(n, int) or not curves_releasable(n):
            raise DisclosureError(
                f"{where}.curves: cell of n={n!r} may not release curves (minimum "
                f"{curve_release_min()}). A 50-point DCA curve is 50 equations in TP and "
                "FP; with n and prevalence also released they invert to per-patient counts."
            )
        if not isinstance(block["curves"], dict):
            raise DisclosureError(f"{where}.curves: must be a mapping")
        for key, value in block["curves"].items():
            _check_key(key, CURVE_FIELDS, f"{where}.curves")
            _check_value(key, value, f"{where}.curves")

    if "intervals" in block:
        _check_value("intervals", block["intervals"], where)

    for attr, cells in (block.get("subgroups") or {}).items():
        _check_dynamic_key(attr, f"{where}.subgroups")
        if any(marker in str(attr).lower() for marker in _CROSSTAB_MARKERS):
            raise DisclosureError(
                f"{where}.subgroups.{attr}: joint/crosstab subgroups may not be exported. "
                "Marginals over the same rows do not determine a joint count; a released "
                "joint cell does."
            )
        if not isinstance(cells, dict):
            raise DisclosureError(f"{where}.subgroups.{attr}: must be a mapping")
        for cat, cell in cells.items():
            _check_dynamic_key(cat, f"{where}.subgroups.{attr}")
            _validate_cell(cell, f"{where}.subgroups.{attr}.{cat}")


def validate_export(payload: dict) -> dict:
    """Validate an aggregate artifact before it is written. Returns it unchanged.

    The writer-side gate. `clif_forest_plot` runs the same allow-list at read time, but a
    site that never ships the wheel still exports through here, so the check has to exist
    on both sides of the boundary.
    """
    if not isinstance(payload, dict):
        raise DisclosureError("export payload must be a mapping")

    for key, value in payload.items():
        _check_key(key, ENVELOPE_FIELDS, "envelope")
        if key == "outcomes":
            continue
        if key == "signature":
            if not isinstance(value, str) or not _SIGNATURE_RE.match(value):
                raise DisclosureError(
                    "envelope.signature: must be 64 lowercase hex characters "
                    "(HMAC-SHA256). Checked by format rather than exempted from checking."
                )
            continue
        _check_value(key, value, "envelope")

    missing = REQUIRED_ENVELOPE_FIELDS - set(payload)
    if missing:
        raise DisclosureError(f"envelope: missing required fields {sorted(missing)}")

    if payload["site_role"] not in SITE_ROLES:
        raise DisclosureError(
            f"envelope.site_role: {payload['site_role']!r} is not one of {sorted(SITE_ROLES)}"
        )
    if payload["partition_role"] not in PARTITION_ROLES:
        raise DisclosureError(
            f"envelope.partition_role: {payload['partition_role']!r} is not one of "
            f"{sorted(PARTITION_ROLES)}"
        )
    if payload["disclosure_status"] not in DISCLOSURE_STATUSES:
        raise DisclosureError(
            f"envelope.disclosure_status: {payload['disclosure_status']!r} is not one of "
            f"{sorted(DISCLOSURE_STATUSES)}"
        )

    outcomes = payload["outcomes"]
    if not isinstance(outcomes, dict):
        raise DisclosureError("envelope.outcomes: must be a mapping of outcome name -> block")
    for name, block in outcomes.items():
        _validate_outcome(name, block)

    return payload


def non_evaluable(status: str, reason: str, label_validity: dict) -> dict:
    """Build a per-outcome block for an outcome that could not be scored.

    Explicit constructor so the failure path is as easy to emit as the success path — an
    evaluator that has to hand-assemble a status block is one that will return a
    fabricated metric instead.
    """
    if status not in NON_EVALUABLE_STATUSES:
        raise DisclosureError(f"{status!r} is not a non-evaluable status")
    return {"status": status, "reason": reason, "label_validity": label_validity}


def banded_status_counts(counts: dict[str, int]) -> dict[str, object]:
    """Replace exact small state counts with a band before export (finding #15)."""
    floor = min_cell_size()
    return {
        state: (f"<{floor}" if isinstance(c, int) and 0 < c < floor else c)
        for state, c in counts.items()
    }


# Backwards-compatible re-exports: log sanitization moved to its own module (finding #34).
from src.eval.log_sanitizer import (  # noqa: E402,F401
    SanitizingFilter,
    install_log_sanitizer,
    redact,
)
