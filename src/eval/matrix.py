"""Reporting harness — the task x site money matrix.

Superseded by two focused modules; kept as the stable import surface:
  - metrics.py   : the TRIPOD+AI metric panel (auroc/auprc/ece/brier/calib/ICI/DCA/LPE/subgroup)
  - method3.py   : the driver — anchor states from a CLIFATRON checkpoint → probe/xgboost heads →
                   3x3 transportability matrix (MIMIC/Rush/UChicago)
  - schema.py    : the allow-listed export contract every artifact passes through

Run the matrix via:  python -m src.eval.method3 --checkpoint ... --site MIMIC=... --site Rush=...
External CLIF-federation validation reuses metrics.full_panel inside the shippable clif-validate/
package (each site runs it locally and returns only aggregate metrics).
"""
from __future__ import annotations

from src.eval.metrics import (  # noqa: F401  (re-exported stable surface)
    expected_calibration_error,
    fit_temperature,
    full_panel,
    local_patient_equivalence,
    net_benefit,
    net_benefit_releasable,
    score,
    subgroup_panel,
    temperature_scale,
)
from src.eval.method3 import transportability_matrix  # noqa: F401
from src.eval.schema import DisclosureError, validate_export  # noqa: F401


def ensemble_mean(*probs, allow_cross_site_ensemble: bool = False):
    """Elemento inference-time ensemble: mean of site models' probabilities.

    **Gated off by default (U5 D11).** This was a second, independent cross-site
    ensembling entry point alongside `method3`'s `matrix["ensemble"]` row. Closing one
    and leaving the other callable is not a fix, so both carry the same gate.

    Averaging predictions from models trained at different sites presupposes a
    derived-model exchange across the institutional boundary. `AGENTS.md` prohibits
    cross-site pooling of model updates, and the plan's Scope Boundaries list cross-site
    ensembles as requiring a separate, currently unobtained approval. Pass
    `allow_cross_site_ensemble=True` only with that approval recorded.

    Site-local ensembling of several models trained at ONE site does not cross the
    boundary and does not need the gate — call `numpy.mean` directly for that.
    """
    if not allow_cross_site_ensemble:
        raise DisclosureError(
            "cross-site ensembling is gated: averaging predictions from models trained "
            "at different sites presupposes a derived-model exchange that has not been "
            "approved. Pass allow_cross_site_ensemble=True only with a recorded "
            "derived-model transfer approval."
        )
    import numpy as np
    return np.mean(np.stack(probs, axis=0), axis=0)


if __name__ == "__main__":
    raise SystemExit("Run the matrix via:  python -m src.eval.method3 --help")
