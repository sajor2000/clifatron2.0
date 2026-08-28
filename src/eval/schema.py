"""Allow-listed aggregate result schema for site-local evaluation export (U5).

Every artifact that leaves a site passes through here. The contract is an ALLOW-LIST,
not a deny-list: a field that is not named below cannot be exported, so a new debug
value, a captured error string, or a fresh provenance key fails closed instead of
travelling. The deny-list (`PROHIBITED_SUBSTRINGS`) is kept as a second, independent
check for the specific field names we know are dangerous.

Validation runs at the WRITER, not only the reader. Development sites (MIMIC, Rush,
UChicago) export through `src/eval/clif_validate.py` rather than the shippable wheel,
so the sites holding real PHI must hit the same allow-list the aggregator does.

Three disclosure controls live here, and they are not interchangeable:

  1. Denominator suppression  — a cell with n < MIN_CELL_SIZE is suppressed.
  2. Numerator suppression    — a cell whose positive OR negative count is below the
                                same threshold is suppressed even when n clears it.
                                `n * prevalence` recovers the exact positive count, so
                                guarding the denominator alone is not suppression.
  3. Resolution bounding      — a cell below CURVE_RELEASE_MIN reports scalar summaries
                                only. A released DCA curve is 50 equations in TP and FP;
                                with n and prevalence also released those invert to
                                per-patient counts at 50 cut-points.

Per-outcome label-validity reporting is REQUIRED, not optional. The allow-list closes
the disclosure channel; without this block it would also close the validity channel,
leaving a site with mis-mapped units or a differently-coded mCIDE concept returning a
plausible AUROC that nothing in the payload can contradict. Fields follow TRIPOD+AI's
participants / outcome / missing-data reporting applied to the federated case
(Collins et al., BMJ 2024;385:e078378, doi:10.1136/bmj-2023-078378).
"""

from __future__ import annotations

import math
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
# The seven states a per-outcome result can be in. `EVALUABLE` is the only one that
# carries metrics; every other state carries a reason and no numbers, so a consumer
# can never mistake "we could not compute this" for "this scored badly".
EVALUABLE = "evaluable"
UNSUPPORTED_AT_SITE = "unsupported_at_site"
SINGLE_CLASS = "single_class"
INSUFFICIENT_N = "insufficient_n"
SMALL_CELL_SUPPRESSED = "small_cell_suppressed"
ARTIFACT_MISMATCH = "artifact_mismatch"
RUNTIME_FAILURE = "runtime_failure"

OUTCOME_STATUSES = frozenset({
    EVALUABLE, UNSUPPORTED_AT_SITE, SINGLE_CLASS, INSUFFICIENT_N,
    SMALL_CELL_SUPPRESSED, ARTIFACT_MISMATCH, RUNTIME_FAILURE,
})

NON_EVALUABLE_STATUSES = OUTCOME_STATUSES - {EVALUABLE}

# Site and partition roles (R15). A site carries exactly one role; a within-site
# partition carries exactly one role. Keeping them as closed sets is what stops
# "development" quietly becoming "external confirmation" in a later report.
SITE_ROLES = frozenset({"reference", "development", "external_confirmation"})
PARTITION_ROLES = frozenset({"train", "validation", "calibration", "test"})


# ---------------------------------------------------------------- policy constants
def load_min_cell_size(policy_path: str | Path = DEFAULT_ARTIFACT_POLICY) -> int:
    """Read the suppression threshold from the landed artifact policy.

    Single source of truth. `configs/artifact_policy.yaml` set `minimum_cell_size: 10`
    in U1 and `tests/test_artifact_policy.py` asserts it; re-deciding the number here
    is how a codebase ends up with two thresholds and a suppression rule that silently
    stops applying (which is exactly what `subgroup_panel`'s hard-coded 30 was).
    """
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


MIN_CELL_SIZE = load_min_cell_size()

# Materially larger than MIN_CELL_SIZE, per the U5 approach. A cell can clear the count
# threshold and still be inverted through its own curves, so curve release needs its own,
# higher bar. 5x the cell minimum: with n >= 50 a 50-point DCA curve is underdetermined
# for per-patient recovery.
CURVE_RELEASE_MIN = 5 * MIN_CELL_SIZE

# Exported prevalence is rounded so `n * prevalence` cannot be inverted to an exact
# positive count. Two decimal places against a MIN_CELL_SIZE-bounded n leaves the
# integer ambiguous.
PREVALENCE_DECIMALS = 2


# ---------------------------------------------------------------- deny-list
# Second line of defence behind the allow-list. Substring match, case-insensitive, so
# `patient_id`, `PatientID`, and `subject_patient_id` are all caught.
PROHIBITED_SUBSTRINGS = (
    "patient_id", "patientid", "hospitalization_id", "hosp_id", "encounter_id",
    "subject_id", "stay_id", "mrn", "sequence", "token", "pos_min",
    "charttime", "storetime", "timestamp", "birth", "dob", "name", "address",
)

# Values that look like a local filesystem path must never appear, regardless of the
# key they sit under. `results["site"] = str(data_path)` is how the site's directory
# layout used to travel.
_PATH_MARKERS = ("/", "\\")


# ---------------------------------------------------------------- allow-list
# Top-level envelope fields. Anything else fails closed.
ENVELOPE_FIELDS = frozenset({
    "schema_version",        # this module's METRIC_SCHEMA_VERSION
    "metric_version",        # version of the metric implementations
    "model_bundle_id",       # opaque identifier for the frozen bundle
    "model_version",         # bundle version, used to key the disclosure ledger
    "vocab_hash",
    "outcome_spec_hash",
    "target_map_hash",
    "clif_version",
    "site_id",               # OPAQUE site identifier - never a path, never a name
    "site_role",
    "partition_role",
    "generated_by",          # tool identifier, not a user or host name
    "outcomes",              # mapping: outcome name -> outcome block
    "disclosure_status",
    "signature",             # site->aggregator report authentication
})

REQUIRED_ENVELOPE_FIELDS = frozenset({
    "schema_version", "metric_version", "model_bundle_id", "model_version",
    "vocab_hash", "outcome_spec_hash", "clif_version", "site_id", "site_role",
    "partition_role", "outcomes", "disclosure_status",
})

# Per-outcome block.
OUTCOME_FIELDS = frozenset({
    "status",                # one of OUTCOME_STATUSES
    "reason",               # free-form-ish explanation for a non-evaluable status
    "metrics",              # scalar metrics; absent unless status == EVALUABLE
    "curves",               # DCA / calibration bins; absent below CURVE_RELEASE_MIN
    "subgroups",            # mapping attribute -> category -> cell block
    "label_validity",       # REQUIRED - see LABEL_VALIDITY_FIELDS
    "intervals",            # aggregate confidence intervals
})

REQUIRED_OUTCOME_FIELDS = frozenset({"status", "label_validity"})

# Scalar metrics permitted inside an outcome block.
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

# Per-outcome label-validity block (TRIPOD+AI participants/outcome/missing-data).
LABEL_VALIDITY_FIELDS = frozenset({
    "outcome_definition_id",
    "outcome_definition_version",
    "status_counts",              # counts across the seven U1 outcome states
    "evaluable_denominator_fraction",
    "measurement_density",        # post-anchor measurement density summary
})

REQUIRED_LABEL_VALIDITY_FIELDS = frozenset({
    "outcome_definition_id", "outcome_definition_version",
    "status_counts", "evaluable_denominator_fraction",
})

# The seven U1 outcome states that `status_counts` must account for.
U1_OUTCOME_STATES = (
    "positive", "negative", "censored", "competing_event",
    "prevalent", "not_ascertainable", "unsupported_at_site",
)

CELL_FIELDS = frozenset({"status", "reason", "n", "prevalence", "auroc", "auprc", "ece"})


# ---------------------------------------------------------------- suppression
def suppress_cell(n: int, n_positive: int) -> tuple[str, str | None]:
    """Decide a cell's disclosure status from its denominator AND numerator.

    Returns `(status, reason)`. A cell is releasable only when the total, the positive
    count, and the negative count all clear MIN_CELL_SIZE. Checking `n` alone is not
    suppression: a cell of n=12 at prevalence 0.0833 clears any size threshold while
    identifying exactly one outcome-positive patient.
    """
    if n < MIN_CELL_SIZE:
        return INSUFFICIENT_N, f"n < {MIN_CELL_SIZE}"
    n_negative = n - n_positive
    if n_positive == 0 or n_negative == 0:
        return SINGLE_CLASS, "outcome has a single class in this cell"
    if n_positive < MIN_CELL_SIZE or n_negative < MIN_CELL_SIZE:
        return SMALL_CELL_SUPPRESSED, (
            f"positive or negative count < {MIN_CELL_SIZE}; releasing n and prevalence "
            "would recover the exact event count"
        )
    return EVALUABLE, None


def round_prevalence(prevalence: float) -> float:
    """Round exported prevalence so it cannot be inverted to an exact event count."""
    if prevalence is None or (isinstance(prevalence, float) and math.isnan(prevalence)):
        return float("nan")
    return round(float(prevalence), PREVALENCE_DECIMALS)


def curves_releasable(n: int) -> bool:
    """Whether a cell of this size may release its DCA / calibration curves."""
    return n >= CURVE_RELEASE_MIN


def apply_complementary_suppression(cells: dict[str, dict]) -> dict[str, dict]:
    """Suppress additional cells when one suppressed cell is recoverable by differencing.

    Suppressing a single cell inside an attribute is not enough: with the sibling cells
    and the attribute total both released, the hidden value is a subtraction away. When
    exactly one cell in an attribute is suppressed, the next-smallest releasable cell is
    suppressed alongside it so at least two unknowns remain.

    Operates on already-status-tagged cells and returns a new mapping.
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
            "n": out[victim].get("n"),
        }
    return out


# ---------------------------------------------------------------- validation
def _looks_like_path(value: object) -> bool:
    if not isinstance(value, str):
        return False
    if value.startswith("~"):
        return True
    return any(marker in value for marker in _PATH_MARKERS)


def _check_key(key: str, allowed: frozenset[str], where: str) -> None:
    if key not in allowed:
        raise DisclosureError(
            f"{where}: field {key!r} is not in the allow-list. Add it to "
            f"src/eval/schema.py deliberately, or do not export it."
        )
    lowered = key.lower()
    for banned in PROHIBITED_SUBSTRINGS:
        if banned in lowered:
            raise DisclosureError(f"{where}: field {key!r} matches prohibited pattern {banned!r}")


# Opaque fields whose bytes are not human-meaningful and may legitimately contain
# characters the path heuristic would flag (base64 padding, slashes).
_OPAQUE_FIELDS = frozenset({"signature"})


def _check_value(key: str, value: object, where: str) -> None:
    if key in _OPAQUE_FIELDS:
        return
    if _looks_like_path(value):
        raise DisclosureError(
            f"{where}: value of {key!r} looks like a local filesystem path ({value!r}); "
            "site directory layout must not leave the node"
        )
    if isinstance(value, str):
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
    for key in block:
        _check_key(key, LABEL_VALIDITY_FIELDS, where)
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


def _validate_outcome(name: str, block: object) -> None:
    where = f"outcomes.{name}"
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

    _validate_label_validity(block["label_validity"], name)

    if status != EVALUABLE:
        if "metrics" in block:
            raise DisclosureError(
                f"{where}: status is {status!r} but a metrics block is present. A "
                "non-evaluable outcome must not carry numbers that read as scores."
            )
        return

    metrics = block.get("metrics")
    if not isinstance(metrics, dict):
        raise DisclosureError(f"{where}.metrics: required when status is {EVALUABLE!r}")
    for key, value in metrics.items():
        _check_key(key, METRIC_FIELDS, f"{where}.metrics")
        _check_value(key, value, f"{where}.metrics")

    n = metrics.get("n")
    if isinstance(n, int) and "curves" in block and not curves_releasable(n):
        raise DisclosureError(
            f"{where}.curves: cell of n={n} is below the curve-release minimum "
            f"({CURVE_RELEASE_MIN}). A released DCA curve is 50 equations in TP and FP; "
            "with n and prevalence also released they invert to per-patient counts."
        )
    if "curves" in block:
        if not isinstance(block["curves"], dict):
            raise DisclosureError(f"{where}.curves: must be a mapping")
        for key in block["curves"]:
            _check_key(key, CURVE_FIELDS, f"{where}.curves")

    for attr, cells in (block.get("subgroups") or {}).items():
        if not isinstance(cells, dict):
            raise DisclosureError(f"{where}.subgroups.{attr}: must be a mapping")
        for cat, cell in cells.items():
            cell_where = f"{where}.subgroups.{attr}.{cat}"
            if not isinstance(cell, dict):
                raise DisclosureError(f"{cell_where}: must be a mapping")
            for key, value in cell.items():
                _check_key(key, CELL_FIELDS, cell_where)
                _check_value(key, value, cell_where)


def validate_export(payload: dict) -> dict:
    """Validate an aggregate artifact before it is written. Returns it unchanged.

    This is the writer-side gate. `clif_forest_plot` runs the same allow-list at read
    time, but a site that never ships the wheel still exports through here, so the
    check has to exist on both sides of the boundary.
    """
    if not isinstance(payload, dict):
        raise DisclosureError("export payload must be a mapping")

    for key, value in payload.items():
        _check_key(key, ENVELOPE_FIELDS, "envelope")
        if key != "outcomes":
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

    outcomes = payload["outcomes"]
    if not isinstance(outcomes, dict):
        raise DisclosureError("envelope.outcomes: must be a mapping of outcome name -> block")
    for name, block in outcomes.items():
        _validate_outcome(name, block)

    return payload


def non_evaluable(status: str, reason: str, label_validity: dict) -> dict:
    """Build a per-outcome block for an outcome that could not be scored.

    Explicit constructor so the failure path is as easy to emit as the success path —
    an evaluator that has to hand-assemble a status block is an evaluator that will
    return a fabricated metric instead.
    """
    if status not in NON_EVALUABLE_STATUSES:
        raise DisclosureError(f"{status!r} is not a non-evaluable status")
    return {"status": status, "reason": reason, "label_validity": label_validity}
