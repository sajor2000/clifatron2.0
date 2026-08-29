"""Bundle-driven zero-shot inference — stable re-export of the vendored wiring."""

from clif_validate._vendor.eval.bundle_inference import (
    bundle_predict_fn,
    resolve_outcome_queries,
    sequences_by_stay,
)
from clif_validate._vendor.eval.clif_validate import (
    evaluate_site,
    load_checkpoint,
    zero_shot_predictions,
)

__all__ = [
    "bundle_predict_fn",
    "evaluate_site",
    "load_checkpoint",
    "resolve_outcome_queries",
    "sequences_by_stay",
    "zero_shot_predictions",
]
