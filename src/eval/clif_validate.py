"""CLIF-Validate: turnkey federation validation package.

Each external CLIF site receives a frozen model checkpoint + this script.
The site runs it on its LOCAL CLIF tables; the auto-labeler derives outcomes
from standard CLIF fields (no manual annotation); the zero-shot threshold/
competing-risk heads produce predictions without local training; and only
aggregate metrics are returned — no raw data, labels, or gradients leave
the node.

This IS the thesis: one small model → many outcomes → many hospitals → one node.

Usage (at each external CLIF site):
    python -m src.eval.clif_validate \
        --checkpoint /path/to/frozen_checkpoint \
        --data /path/to/local_clif_parquet/ \
        --episode-artifact /path/to/episodes.parquet \
        --site-id SITE-07 \
        --release-id 2026-08-28-site07-v0 \
        --signing-key-file /secure/site07.key \
        --out output/final_no_phi/site_07.json

`--site-id` is an OPAQUE identifier, not a hospital name and not a path: the site
identifier travels in the exported artifact, so it must not carry anything that
identifies the institution's filesystem or, by itself, the institution.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from src.eval import attestation as _attest
from src.eval import metrics as M
from src.eval import schema as _schema

# Site-side logging is sanitized at the source. Stripping the path from the exported
# JSON is not enough on its own: this module used to print the data path and per-outcome
# counts to stdout, so a returned console log carried what the artifact no longer did.
_log = logging.getLogger("clif_validate")
_schema.install_log_sanitizer(_log)


class ArtifactMismatch(RuntimeError):
    """Raised when the bundle a site was given cannot be used to produce a real result.

    Deliberately fatal. The failure mode this exists to stop is a validator that
    degrades into a success-shaped report: partial weights, a placeholder prediction, or
    a vocabulary that does not match the model, all of which produce numbers that look
    exactly like evidence.
    """


def load_checkpoint(path: str):
    """Load frozen CLIFATRON checkpoint with our heads attached. Fails closed (U5 D2).

    The checkpoint directory must contain:
      - pytorch_model.bin (or safetensors)  — backbone weights
      - config.json                          — HF model config
      - head_weights.pt                      — our trained head parameters
      - vocab.json                           — frozen CLIF mCIDE vocabulary

    Head weights are now REQUIRED and are loaded with `strict=True`. Previously the file
    was optional and loaded with `strict=False`, so a checkpoint with missing or partial
    heads produced an untrained-head model that still emitted a full metric panel. A
    site cannot tell that report from a real one.

    There is deliberately no `allow_partial` escape hatch (U5 review #31). One existed,
    with a docstring promising a partial load "never reaches an exported artifact" and
    nothing whatsoever enforcing that -- no caller, no CLI flag, no coupling to outcome
    status. An unenforced exemption inside a fail-closed control is worse than none: it
    reads as a considered safety valve while behaving as an open door.
    """
    import torch
    from src.model.head_adapter import CLIFATRONHeads, load_backbone

    ckpt = Path(path)
    head_path = ckpt / "head_weights.pt"
    if not head_path.exists():
        raise ArtifactMismatch(
            "head_weights.pt is absent from the bundle. Without trained heads this "
            "validator would score an untrained model and emit a benchmark-shaped "
            "report; refusing rather than degrading."
        )

    backbone = load_backbone(path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = backbone.to(device).eval()

    n_targets = 10
    model = CLIFATRONHeads(backbone, n_targets, freeze_backbone=True)

    state = torch.load(head_path, map_location=device, weights_only=True)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ArtifactMismatch(
            f"head weights do not match the model definition: {exc}. A partial load "
            "would leave some heads untrained while the report still looked complete."
        ) from exc

    return model.to(device)


def verify_bundle_compatibility(checkpoint_path: str, *, vocab_hash: str | None = None,
                                outcome_spec_hash: str | None = None,
                                clif_version: str | None = None,
                                supported_clif_versions: tuple[str, ...] = ("2.1",)) -> dict:
    """Check the bundle's identity hashes before any inference runs (U5 D2).

    Mirrors the fail-closed shape `src/data/value_stats.py` established for value
    statistics: an artifact is bound to the exact vocabulary it was built against, and a
    null or mismatched hash is a hard failure rather than a warning. Returns the
    provenance block the export schema requires.
    """
    manifest_path = Path(checkpoint_path) / "bundle_manifest.json"
    if not manifest_path.exists():
        raise ArtifactMismatch(
            f"bundle_manifest.json is absent from {checkpoint_path}. Provenance cannot "
            "be established, so no result from this bundle is attributable."
        )
    manifest = json.loads(manifest_path.read_text())

    for field, expected in (("vocab_hash", vocab_hash),
                            ("outcome_spec_hash", outcome_spec_hash)):
        actual = manifest.get(field)
        if actual in (None, ""):
            raise ArtifactMismatch(f"bundle manifest declares no {field}")
        if expected is not None and actual != expected:
            raise ArtifactMismatch(
                f"{field} mismatch: bundle declares {actual!r}, site expects {expected!r}. "
                "Running anyway would score one vocabulary's model against another's tokens."
            )

    version = clif_version or manifest.get("clif_version")
    if version not in supported_clif_versions:
        raise ArtifactMismatch(
            f"CLIF version {version!r} is not supported by this bundle "
            f"(supported: {list(supported_clif_versions)})"
        )

    return {
        "model_bundle_id": manifest["model_bundle_id"],
        "model_version": manifest["model_version"],
        "vocab_hash": manifest["vocab_hash"],
        "outcome_spec_hash": manifest["outcome_spec_hash"],
        "clif_version": version,
    }


def zero_shot_predictions(model, batches, target_indices, tau_bins,
                          directions, batch_size=8):
    """Run zero-shot threshold-hazard inference.

    Returns per-outcome per-stay cumulative failure probabilities at 48h.
    No trained task heads needed — the threshold head is zero-shot.
    """
    import torch

    device = next(model.parameters()).device
    n_outcomes = len(target_indices)
    n_stays = len(batches)
    probs = np.zeros((n_stays, n_outcomes))

    for start in range(0, n_stays, batch_size):
        end = min(start + batch_size, n_stays)
        batch_ids = [b["input_ids"] for b in batches[start:end]]
        batch_masks = [b["attention_mask"] for b in batches[start:end]]

        max_len = max(len(ids) for ids in batch_ids)
        ids = torch.zeros((len(batch_ids), max_len), dtype=torch.long, device=device)
        masks = torch.zeros((len(batch_ids), max_len), dtype=torch.long, device=device)
        for i, (id_seq, mask_seq) in enumerate(zip(batch_ids, batch_masks)):
            ids[i, :len(id_seq)] = torch.tensor(id_seq)
            masks[i, :len(mask_seq)] = torch.tensor(mask_seq)

        for k in range(n_outcomes):
            f = model.threshold_prob(
                ids, masks,
                torch.tensor([target_indices[k]] * len(batch_ids), device=device),
                torch.tensor([tau_bins[k]] * len(batch_ids), device=device),
                torch.tensor([directions[k]] * len(batch_ids), device=device),
            )
            probs[start:end, k] = f[:, -1].cpu().numpy()

    return probs


def _label_validity_block(labels_df, name: str, spec_version: str) -> dict:
    """Per-outcome TRIPOD+AI participants/outcome/missing-data reporting.

    Required by the export schema. Without it a site whose labels are wrong — a
    mis-mapped unit, a differently-coded mCIDE concept, an outcome ascertained on a
    systematically different subset — returns a plausible AUROC that nothing in the
    payload can contradict.
    """
    # `auto_label` emits a companion `<name>_status` column carrying the U1 outcome
    # state verbatim. Prefer it: inferring state from the binary column is exactly the
    # null-to-negative collapse U1 exists to prevent.
    status_col = f"{name}_status"
    counts = {state: 0 for state in _schema.U1_OUTCOME_STATES}
    if status_col in labels_df.columns:
        col = labels_df[status_col].to_list()
        for value in col:
            key = str(value)
            counts[key if key in counts else "not_ascertainable"] += 1
    else:
        col = labels_df[name].to_list()
        for value in col:
            if value in (0, 1, True, False):
                counts["positive" if value in (1, True) else "negative"] += 1
            else:
                counts["not_ascertainable"] += 1
    total = max(len(col), 1)
    evaluable = counts["positive"] + counts["negative"]
    return {
        "outcome_definition_id": name,
        "outcome_definition_version": spec_version,
        # Banded, not exact (U5 review #15). A suppressed outcome used to ship exact
        # positive/negative/censored counts through the very block added to close the
        # validity channel, so counts below the floor left the hospital anyway.
        "status_counts": _schema.banded_status_counts(counts),
        "evaluable_denominator_fraction": round(evaluable / total, 3),
    }


def _binary_labels(labels_df, name: str):
    """Extract (mask, y) for the rows where this outcome is genuinely ascertained."""
    status_col = f"{name}_status"
    raw = labels_df[name].to_list()
    statuses = (labels_df[status_col].to_list() if status_col in labels_df.columns
                else [None] * len(raw))
    mask, y = [], []
    for value, status in zip(raw, statuses):
        # A row counts only when the outcome was genuinely ascertained. Censored,
        # prevalent, not-ascertainable and unsupported rows are excluded from the
        # denominator rather than being read as negatives.
        ascertained = (status in ("positive", "negative") if status is not None
                       else value in (0, 1, True, False))
        mask.append(bool(ascertained))
        if ascertained:
            y.append(1 if (status == "positive" or value in (1, True)) else 0)
    return np.asarray(mask, dtype=bool), np.asarray(y, dtype=int)


def evaluate_site(checkpoint_path: str, data_path: str, episode_artifact: str,
                  outcome_cfgs: list[dict], *, spec_version: str = "1.0.0",
                  predict_fn=None) -> dict:
    """Run full evaluation at one site. Returns a schema-valid, aggregate-only artifact.

    **No prediction path in this function can produce a random number (U5 D1).** Two
    `np.random.random(...)` call sites previously supplied the predictions that
    `full_panel` then turned into AUROC/ECE/Brier, so the validator emitted
    benchmark-shaped metrics from noise. There is now no fallback at all: either real
    inference runs or the run fails.

    `episode_artifact` is REQUIRED (U5 D10). `auto_label`'s landed signature is
    `auto_label(data_dir, episode_artifact, outcomes=None, ...)`, and this function used
    to call `auto_label(data_path, outcome_names)` — passing the outcome-name list into
    the `episode_artifact` slot, so the validator raised before it ever reached
    inference. No test exercised this path, so the suite never caught it.

    `predict_fn` is the seam for real inference: a callable taking the labeled frame and
    returning `(n_stays, n_outcomes)` probabilities. Production wires
    `tokenize_site -> ModelDataset -> zero_shot_predictions` through it; tests pass a
    deterministic stub. It has no default, so there is nothing to fall back to.
    """
    from src.eval.clif_auto_labeler import auto_label

    if predict_fn is None:
        raise ArtifactMismatch(
            "evaluate_site requires a prediction function. There is deliberately no "
            "default: the previous placeholder produced np.random predictions that "
            "full_panel turned into a benchmark-shaped report."
        )

    outcome_names = [o["name"] for o in outcome_cfgs]
    labels_df = auto_label(data_path, episode_artifact, outcome_names)
    _log.info("labeled %d stays across %d outcomes", len(labels_df), len(outcome_names))

    probs = np.asarray(predict_fn(labels_df), dtype=float)
    if probs.shape != (len(labels_df), len(outcome_cfgs)):
        raise ArtifactMismatch(
            f"prediction function returned {probs.shape}, expected "
            f"{(len(labels_df), len(outcome_cfgs))}"
        )
    if not np.isfinite(probs).any():
        raise ArtifactMismatch("prediction function returned no finite predictions")

    outcomes: dict[str, dict] = {}
    for k, cfg in enumerate(outcome_cfgs):
        name = cfg["name"]
        validity = _label_validity_block(labels_df, name, spec_version)
        mask, y = _binary_labels(labels_df, name)

        if mask.sum() == 0:
            outcomes[name] = _schema.non_evaluable(
                _schema.UNSUPPORTED_AT_SITE, "outcome is not ascertainable at this site",
                validity)
            continue

        p = probs[mask, k]
        status, reason = _schema.suppress_cell(int(mask.sum()), int(y.sum()))
        if status != _schema.EVALUABLE:
            outcomes[name] = _schema.non_evaluable(status, reason, validity)
            continue

        panel = M.full_panel(p, y)
        block = {
            "status": _schema.EVALUABLE,
            "label_validity": validity,
            "metrics": {
                "auroc": panel["auroc"], "auprc": panel["auprc"], "n": panel["n"],
                "prevalence": _schema.round_prevalence(panel["prevalence"]),
                "ece": panel.get("ece"), "brier": panel.get("brier"),
                "calib_slope": panel.get("calib_slope"),
                "calib_intercept": panel.get("calib_intercept"),
                "ici": panel.get("ici"), "temperature": panel.get("temperature"),
                "n_dropped_nan": panel.get("n_dropped_nan", 0),
            },
        }
        block["metrics"] = {k2: v for k2, v in block["metrics"].items() if v is not None}
        curve = M.net_benefit_releasable(p, y)
        if curve is not None:
            block["curves"] = {
                "dca_thresholds": [float(t) for t in curve["thresholds"]],
                "dca_model": [float(v) for v in curve["model"]],
                "dca_treat_all": [float(v) for v in curve["treat_all"]],
            }
        outcomes[name] = block
        # Aggregate shape only; never the site path, never a per-stay value.
        _log.info("outcome %s: n=%d status=%s", name, panel["n"], _schema.EVALUABLE)

    return {"outcomes": outcomes}


def build_export(outcomes: dict, provenance: dict, *, site_id: str, site_role: str,
                 partition_role: str, release_id: str,
                 disclosure_status: str = _schema.DRAFT_DISCLOSURE_STATUS,
                 signing_key: bytes | None = None) -> dict:
    """Assemble, validate, and optionally sign the artifact that leaves the site.

    Validation runs HERE, at the writer. `clif_forest_plot` runs the same allow-list at
    read time, but development sites export through this module rather than the
    shippable wheel, so the sites holding real PHI would otherwise pass through only a
    named-field deny-list. An unrecognized key fails closed on both sides of the boundary.
    """
    payload = {
        "schema_version": _schema.METRIC_SCHEMA_VERSION,
        "metric_version": _schema.METRIC_SCHEMA_VERSION,
        "site_id": site_id,
        "site_role": site_role,
        "partition_role": partition_role,
        # Defaults to the DRAFT status. `write_export` refuses to release a draft, so
        # crossing the boundary requires someone to make and record a disclosure
        # decision rather than inheriting a decorative default (U5 review #16).
        "disclosure_status": disclosure_status,
        # Unique per release, so a replayed report cannot be counted as another site
        # (U5 review #13).
        "release_id": release_id,
        "generated_by": "clif_validate",
        "outcomes": outcomes,
        **provenance,
    }
    _schema.validate_export(payload)
    if signing_key:
        payload = _attest.sign_report(payload, signing_key)
    return payload


def write_export(payload: dict, out_path: str | Path, ledger_path: str | Path,
                 published_release_ids: set | None = None) -> Path:
    """Check, write, then record. Refuses to release a draft.

    **A draft is not a release (U5 review #16).** `build_export` produces
    `pending_review`, and this function refuses it, so the release boundary cannot be
    crossed without a recorded disclosure decision. The status was previously written to
    disk verbatim, which made it decorative.

    **Ordering (U5 review #17, then Greptile PR #4).** Three steps, and the order is the
    whole point:

        1. write the payload to a `.partial` path -- durable, but NOT the artifact
        2. append to the ledger
        3. atomically rename `.partial` into place -- this is what publishes it

    The original order appended to the ledger first, so a failed write left a phantom
    release blocking the next genuine one. Simply inverting that traded the phantom for
    its mirror: rename-then-append can publish a report the ledger never records, and a
    later differencing check would then miss a real prior release -- the exact failure
    the ledger exists to prevent.

    Writing to a side path first gets both. If the ledger append fails, the artifact was
    never published and the temp file is removed. If the rename fails, the ledger holds
    an entry for something unpublished, which over-records: it can only block a future
    release, never permit a disclosure. Both failure directions are conservative.
    """
    # Reconciliation is ENFORCED, not documented (Greptile PR #4, follow-up on #6).
    # A crash between publication and confirmation leaves a visible artifact whose ledger
    # record is unconfirmed. Saying in a docstring that an operator "should" reconcile
    # before the next export is not a control: nothing stopped the next export from
    # running against an incomplete history. This gate does.
    residue = _attest.unconfirmed_releases(ledger_path) - {payload.get("release_id")}
    if residue:
        if published_release_ids is None:
            raise _schema.DisclosureError(
                f"the ledger holds unconfirmed releases {sorted(residue)} from an earlier "
                "attempt. Each either published and needs confirming, or never published "
                "and is inert -- and this process cannot tell which. Pass "
                "published_release_ids={...} listing what is actually visible on the "
                "export volume so the residue can be classified."
            )
        unresolved = _attest.reconcile_ledger(ledger_path, published_release_ids)
        if unresolved:
            raise _schema.DisclosureError(
                f"releases {unresolved} are published but unconfirmed. Confirm them with "
                "attestation.confirm_publication before exporting again -- until then the "
                "next release would be checked against a history missing artifacts that "
                "are already visible."
            )

    status = payload.get("disclosure_status")
    if status not in _schema.RELEASABLE_DISCLOSURE_STATUSES:
        raise _schema.DisclosureError(
            f"refusing to release an artifact with disclosure_status {status!r}. Only "
            f"{sorted(_schema.RELEASABLE_DISCLOSURE_STATUSES)} may leave the site. A "
            "draft is reviewed and re-stamped by whoever owns the disclosure decision."
        )

    _attest.check_cross_release_differencing(payload, ledger_path)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".partial")

    # 1. Durable, but not yet the artifact. Nothing reads a .partial path.
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))

    # 2. Record the INTENT to release, durably and unconfirmed. An unconfirmed entry
    #    gates nothing, so a failure here or below leaves the release id retryable.
    try:
        _attest.append_to_ledger(payload, ledger_path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise

    # 3. Publish. Atomic rename, so no reader ever sees a partial artifact. On failure the
    #    temp file is removed and the unconfirmed ledger entry is left behind inert --
    #    it blocks nothing, and the same release id can be retried.
    try:
        tmp.replace(out)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise

    # 4. Confirm. Only now does this count as a prior release for differencing and replay.
    #    A crash between 3 and 4 is the one remaining window, and it is recoverable rather
    #    than silent: `attestation.reconcile_ledger` names any published-but-unconfirmed
    #    release so an operator can confirm it before the next export.
    _attest.confirm_publication(payload, ledger_path)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--episode-artifact", required=True,
                    help="U1 episode/split artifact (parquet); required by auto_label")
    ap.add_argument("--site-id", required=True,
                    help="OPAQUE site identifier. Not a hostname, not a path.")
    ap.add_argument("--site-role", default="development", choices=sorted(_schema.SITE_ROLES))
    ap.add_argument("--partition-role", default="test",
                    choices=sorted(_schema.PARTITION_ROLES))
    # Defaults inside the directories configs/artifact_policy.yaml actually declares
    # (U5 review #18). `results/` was unclassified, so a default run wrote
    # disclosure-relevant files somewhere with no retention or export rules.
    ap.add_argument("--out", default="output/final_no_phi/validation.json")
    ap.add_argument("--ledger", default="output/intermediate_phi/disclosure_ledger.jsonl")
    ap.add_argument("--access-log", default="output/intermediate_phi/access_log.jsonl")
    ap.add_argument("--release-id", required=True,
                    help="unique identifier for THIS release. Replaying one is rejected, "
                         "so a duplicated report cannot be counted as another site.")
    ap.add_argument("--signing-key-file", default=None,
                    help="file holding this site's hex shared secret. Without it the "
                         "report is unsigned and the aggregator will refuse it.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _schema.install_log_sanitizer(logging.getLogger())

    outcome_cfgs = [
        {"name": "in_hospital_mortality", "direction": "above"},
        {"name": "new_imv_24h", "direction": "above"},
        {"name": "new_vasopressor_24h", "direction": "above"},
        {"name": "aki_kdigo_48h", "direction": "above"},
        {"name": "resp_failure_48h", "direction": "above"},
    ]

    provenance = verify_bundle_compatibility(args.checkpoint)
    model = load_checkpoint(args.checkpoint)

    def predict_fn(labels_df):
        """Real inference seam.

        DELIBERATELY UNIMPLEMENTED, and failing closed rather than approximating.

        The pieces exist -- `src/data/tokenize.py::tokenize_site` produces the event
        shards, `src/data/dataset.py::ModelDataset` / `make_dataloader` batch them, and
        `zero_shot_predictions` above consumes those batches -- but wiring them needs
        four things this CLI does not yet receive: the resolved data config, the
        bundle-pinned vocabulary and numeric edges, a policy-checked output directory
        for the intermediate shards, and the episode frame rather than its path
        (`tokenize_site(cfg, site, base, out, vocab, edges, ..., episodes=...)`).

        Raising here is the point. The whole of D1 was a placeholder that produced
        something plausible instead of stopping, and an approximate call against a
        signature this module cannot satisfy would be the same defect wearing different
        clothes -- it would fail at a site, at runtime, with a confusing error, after
        the operator had already been told the run was under way.

        U5 is data-free by contract: the seam, the fail-closed behaviour, and the export
        path are in scope and tested; running real inference on governed site data is
        not. Supply `predict_fn` from a caller that has the bundle artifacts, or wire
        this once the bundle-manifest vocabulary lands with U9.
        """
        raise ArtifactMismatch(
            "real inference is not wired into this CLI yet: it needs the resolved data "
            "config, the bundle-pinned vocabulary and numeric edges, and a "
            "policy-checked shard directory. Pass an explicit predict_fn from a caller "
            "that holds the bundle artifacts. Refusing rather than approximating -- an "
            "approximate prediction path is the defect this unit exists to remove."
        )

    result = evaluate_site(args.checkpoint, args.data, args.episode_artifact,
                           outcome_cfgs, predict_fn=predict_fn)

    signing_key = (bytes.fromhex(Path(args.signing_key_file).read_text().strip())
                   if args.signing_key_file else None)
    payload = build_export(result["outcomes"], provenance, site_id=args.site_id,
                           site_role=args.site_role, partition_role=args.partition_role,
                           release_id=args.release_id, signing_key=signing_key)
    out = write_export(payload, args.out, args.ledger)
    _attest.record_access(args.access_log, model_version=provenance["model_version"],
                          actor_role="site_operator", artifact_id=out.name, action="export")

    # Reports what actually happened, not an unconditional reassurance. The previous
    # version printed "No raw data ... have left the node." on every run, including runs
    # that failed or wrote a path-bearing artifact.
    evaluable = sum(1 for b in result["outcomes"].values()
                    if b["status"] == _schema.EVALUABLE)
    _log.info("export validated against the allow-list and written: %d/%d outcomes evaluable",
              evaluable, len(result["outcomes"]))


if __name__ == "__main__":
    main()