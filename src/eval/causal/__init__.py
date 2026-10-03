"""Causal estimators, design diagnostics and the trial-benchmark layer for the extubation emulation.

Modules (nothing is imported here, so importing the package pulls in no dependency):

- `estimators`, `diagnostics` (U11): standard library, numpy and scikit-learn only.
- `benchmark` (U12): benchmark registry, feasibility screen, agreement rule and the
  paired comparison between estimators. Standard library and numpy; PyYAML only inside
  `load_benchmark_registry`.
- `simulation` (U12): planted-effect simulation of the agreement rule. Standard library,
  numpy and scikit-learn.
- `emulate` (U12): per-trial emulation on a site cohort and the outcome-by-arm gate
  (`authorize_outcome_by_arm`). Also uses polars for the cohort, label and device-row
  frames, and imports `src.eval.extubation_labeler` for outcome arrays.

All of them are written to be vendored into the `clif-validate` site package, which
declares numpy, scikit-learn, pyyaml and polars (not LightGBM or scipy).
"""
