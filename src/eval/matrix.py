"""Reporting harness — the task x site money matrix.

Superseded by two focused modules; kept as the stable import surface:
  - metrics.py   : the TRIPOD+AI metric panel (auroc/auprc/ece/brier/calib/ICI/DCA/LPE/subgroup)
  - method3.py   : the driver — anchor states from a CLIFATRON checkpoint → probe/xgboost heads →
                   3x3 transportability matrix (MIMIC/Rush/UChicago) + Elemento ensemble

Run the matrix via:  python -m src.eval.method3 --checkpoint ... --site MIMIC=... --site Rush=...
External CLIF-federation validation reuses metrics.full_panel inside the shippable clif-validate/
package (each site runs it locally and returns only aggregate metrics).
"""
from __future__ import annotations

from src.eval.metrics import (  # noqa: F401  (re-exported stable surface)
    expected_calibration_error,
    full_panel,
    local_patient_equivalence,
    net_benefit,
    score,
    subgroup_panel,
    temperature_scale,
)
from src.eval.method3 import transportability_matrix  # noqa: F401


def ensemble_mean(*probs):
    """Elemento inference-time ensemble: mean of site models' probabilities."""
    import numpy as np
    return np.mean(np.stack(probs, axis=0), axis=0)


if __name__ == "__main__":
    raise SystemExit("Run the matrix via:  python -m src.eval.method3 --help")
