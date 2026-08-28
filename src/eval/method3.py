"""Method 3 — calibrated survival/probe heads on a CLIFATRON backbone.

The wedge (notes/INTEGRATION.md): reuse CLIFATRON's tokenized narratives + trained checkpoint,
extract the hour-24 anchor hidden state, and show our objective + calibration beats their
Method 1 (XGBoost-on-embeddings) on AUPRC/calibration and Method 2 (MC rollout) on cost.

Runs the DEVELOPMENT 3x3 transportability matrix (MIMIC/Rush/UChicago) + Elemento ensemble,
with the full TRIPOD+AI panel (metrics.py). Two head modes:
  - probe      : our TaskHead (binary) on the frozen anchor state — runs on ANY released
                 checkpoint TODAY; the fair head-to-head vs their XGBoost.
  - zero_shot  : CompetingRiskHead / ThresholdHazardHead — needs a checkpoint pretrained WITH
                 our heads (phase 2, head_adapter joint training); gives label-free predictions,
                 the mechanism that makes external CLIF-federation validation possible.

Assumes each site's narratives parquet has: sequence (list[int], truncated to first 24h),
label (int/float), and optional subgroup columns (sex, race, age_band). Column names are
configurable; VERIFY against CLIFATRON's build_benchmark.py output.

Data never crosses sites: point --site NAME=PATH at each node's local parquet; the matrix is
assembled from per-site metric JSONs, not pooled rows.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.eval import metrics as M
from src.eval import schema as _schema


class PartitionError(ValueError):
    """Raised when a fit/evaluate workflow is handed rows without usable partition roles."""


INSUFFICIENT_PARTITIONS = "insufficient_partitions"


# --------------------------------------------------------------------------- data
def load_site(path: str, seq_col: str, label_col: str, group_cols: list[str],
              partition_col: str = "partition"):
    """Load one site's narratives into (sequences, labels, groups, partitions). No pooling.

    `partition_col` is REQUIRED to exist (U5 D6). Every fit/evaluate workflow downstream
    needs to know which rows may fit a predictor, which may fit a calibrator, and which
    may be scored -- and inferring that from array shape is exactly how the diagonal
    ended up fitting and scoring on identical rows. The column comes from the U1 split
    artifact (`src/data/splits.py`; `clif_auto_labeler.auto_label` already returns it).
    """
    import polars as pl
    df = pl.read_parquet(path)
    if partition_col not in df.columns:
        raise PartitionError(
            f"{path} has no {partition_col!r} column. Site arrays without partition "
            "roles cannot enter a fit/evaluate workflow: there is no way to keep the "
            "rows that fit a predictor disjoint from the rows that score it. Join the "
            "U1 split artifact onto this table first."
        )
    seqs = df[seq_col].to_list()
    labels = np.asarray(df[label_col].to_list())
    groups = {g: np.asarray(df[g].to_list()) for g in group_cols if g in df.columns}
    partitions = np.asarray(df[partition_col].to_list())
    unknown = sorted(set(partitions.tolist()) - set(_schema.PARTITION_ROLES))
    if unknown:
        raise PartitionError(
            f"{path}: unknown partition roles {unknown}; expected a subset of "
            f"{sorted(_schema.PARTITION_ROLES)}"
        )
    return seqs, labels, groups, partitions


def collate(seqs: list[list[int]], pad_id: int = 0):
    """Right-pad token sequences; build attention mask (1=real). Matches CLIFATRON's collate."""
    import torch
    max_len = max(len(s) for s in seqs)
    ids = torch.zeros((len(seqs), max_len), dtype=torch.long) + pad_id
    mask = torch.zeros((len(seqs), max_len), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        mask[i, : len(s)] = 1
    return ids, mask


# ----------------------------------------------------------------- anchor embeddings
def extract_anchor_states(backbone, seqs, device, batch_size=16, pad_id=0):
    """Frozen forward pass → hour-24 anchor hidden state per stay (last real token)."""
    import torch
    from src.model.head_adapter import CLIFATRONHeads
    backbone.eval().to(device)
    states = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            ids, mask = collate(seqs[i : i + batch_size], pad_id)
            ids, mask = ids.to(device), mask.to(device)
            out = backbone(input_ids=ids, attention_mask=mask, output_hidden_states=True)
            H = out.hidden_states[-1]                       # [B, T, d]
            anchor = mask.long().sum(1) - 1
            h = H[torch.arange(H.size(0)), anchor]          # [B, d]
            states.append(h.float().cpu().numpy())
    return np.concatenate(states, axis=0)                   # [N, d]


# ---------------------------------------------------------------------- head modes
def fit_probe(X_tr, y_tr, epochs=200, lr=1e-3, wd=1e-4, device="cpu"):
    """Our TaskHead (single binary head) on frozen states. Returns a fn X -> logits."""
    import torch
    from src.model.heads import TaskHead
    Xt = torch.tensor(X_tr, dtype=torch.float32, device=device)
    yt = torch.tensor(y_tr, dtype=torch.float32, device=device)[:, None]
    head = TaskHead(Xt.shape[1], 1).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    mask = torch.ones_like(yt)
    for _ in range(epochs):
        opt.zero_grad()
        loss = head.loss(Xt, yt, mask)
        loss.backward()
        opt.step()

    def predict(X):
        with torch.no_grad():
            z = head(torch.tensor(X, dtype=torch.float32, device=device)).cpu().numpy()[:, 0]
        return z  # logits

    return predict


def fit_xgboost(X_tr, y_tr):
    """CLIFATRON Method 1 baseline: XGBoost on the same frozen embeddings. Returns X -> prob."""
    import xgboost as xgb
    clf = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss")
    clf.fit(X_tr, y_tr.astype(int))
    return lambda X: clf.predict_proba(X)[:, 1]


# ------------------------------------------------------------------- matrix assembly
def _role_mask(partitions: np.ndarray, role: str) -> np.ndarray:
    return np.asarray(partitions) == role


def transportability_matrix(states: dict, labels: dict, groups: dict, partitions: dict,
                            method: str = "probe",
                            allow_cross_site_ensemble: bool = False):
    """states/labels/groups/partitions keyed by site. Returns matrix[train_site][test_site].

    `method` in {'probe','xgboost'} -- our head vs CLIFATRON Method 1, same embeddings.

    **Partition isolation (U5 D6, D7).** Each site's predictor is fitted on that site's
    `train` rows, its calibrator on `calibration` rows, and every reported number comes
    from `test` rows. The previous implementation fitted on `states[s], labels[s]` --
    the whole site -- and then scored `matrix[tr][te]` including the `tr == te` diagonal,
    so the diagonal fitted and scored on identical rows and `full_panel(recalibrate=True)`
    re-fitted temperature on `labels[te]` on top of that. LPE had the same defect: it was
    fitted on the very test labels it was scoring against.

    The diagonal is still reported, and is now legitimate: train and test are disjoint
    partitions of the same site, which is internal validation rather than fit-on-self.

    A site missing a required partition yields an `insufficient_partitions` cell rather
    than a number -- fail closed, not fall back to the whole array.

    **`allow_cross_site_ensemble` defaults to False (U5 D8).** The Elemento-style
    inference-time ensemble averages site-local model predictions across sites, which
    presupposes a cross-site derived-model exchange that `AGENTS.md` and this plan's
    Scope Boundaries both list as unapproved. It is not a metric option; it is a
    governance boundary, so it is off unless a caller names the approval.
    """
    sites = list(states.keys())
    fit = fit_probe if method == "probe" else fit_xgboost

    missing = [s for s in sites if s not in partitions]
    if missing:
        raise PartitionError(
            f"sites {missing} were passed without partition roles. Unsplit site arrays "
            "cannot enter a fit/evaluate workflow."
        )

    # One predictor per site, fitted on that site's TRAIN rows only (local).
    predictors, calibrations = {}, {}
    for s in sites:
        tr_mask = _role_mask(partitions[s], "train")
        if tr_mask.sum() == 0 or len(np.unique(labels[s][tr_mask])) < 2:
            predictors[s] = None
            continue
        predictors[s] = fit(states[s][tr_mask], labels[s][tr_mask])

        # Calibrator fitted on the CALIBRATION partition, never on anything scored.
        cal_mask = _role_mask(partitions[s], "calibration")
        if method == "probe" and cal_mask.sum() > 0 and len(np.unique(labels[s][cal_mask])) > 1:
            cal_logits = predictors[s](states[s][cal_mask])
            calibrations[s] = M.fit_temperature(cal_logits, labels[s][cal_mask])

    matrix, probs_on = {}, {t: {} for t in sites}
    for tr in sites:
        matrix[tr] = {}
        for te in sites:
            te_mask = _role_mask(partitions[te], "test")
            if predictors[tr] is None or te_mask.sum() == 0:
                matrix[tr][te] = {
                    "status": INSUFFICIENT_PARTITIONS,
                    "reason": ("train partition unusable at the fitting site"
                               if predictors[tr] is None
                               else "no test partition at the evaluating site"),
                }
                continue

            X_te, y_te = states[te][te_mask], labels[te][te_mask]
            raw = predictors[tr](X_te)
            if method == "probe":
                logits = raw
                p = 1.0 / (1.0 + np.exp(-raw))
                cell = M.full_panel(p, y_te, logits=logits,
                                    temperature=calibrations.get(tr))
            else:
                p = raw
                cell = M.full_panel(p, y_te)
            cell["status"] = _schema.EVALUABLE

            if te in groups:
                cell["subgroups"] = M.subgroup_panel(
                    p, y_te, {k: v[te_mask] for k, v in groups[te].items()})

            if tr != te:
                # LPE fits a local model; it needs its OWN fit partition (D7). Fitting it
                # on the rows being scored made it a comparison against itself.
                lpe_mask = _role_mask(partitions[te], "train")
                if lpe_mask.sum() > 0 and len(np.unique(labels[te][lpe_mask])) > 1:
                    cell["lpe"] = M.local_patient_equivalence(
                        cell.get("auroc", 0.0),
                        states[te][lpe_mask], labels[te][lpe_mask],
                        eval_X=X_te, eval_y=y_te)
                else:
                    cell["lpe"] = None

            matrix[tr][te] = cell
            probs_on[te][tr] = p

    if allow_cross_site_ensemble:
        matrix["ensemble"] = {}
        for te in sites:
            te_mask = _role_mask(partitions[te], "test")
            available = [probs_on[te][tr] for tr in sites if tr in probs_on[te]]
            if not available or te_mask.sum() == 0:
                matrix["ensemble"][te] = {"status": INSUFFICIENT_PARTITIONS,
                                          "reason": "no usable per-site predictions"}
                continue
            p_ens = np.mean(available, axis=0)
            y_te = labels[te][te_mask]
            cell = M.full_panel(p_ens, y_te)
            cell["status"] = _schema.EVALUABLE
            if te in groups:
                cell["subgroups"] = M.subgroup_panel(
                    p_ens, y_te, {k: v[te_mask] for k, v in groups[te].items()})
            matrix["ensemble"][te] = cell
    return matrix


# ------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="trained CLIFATRON HF checkpoint dir")
    ap.add_argument("--site", action="append", required=True,
                    help="NAME=/path/to/site_narratives.parquet (repeat per site)")
    ap.add_argument("--seq-col", default="sequence")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--group-cols", default="sex,race,age_band")
    ap.add_argument("--partition-col", default="partition",
                    help="column carrying the U1 split role (train/validation/calibration/test)")
    ap.add_argument("--allow-cross-site-ensemble", action="store_true",
                    help="requires a recorded derived-model transfer approval; off by default")
    ap.add_argument("--method", choices=["probe", "xgboost", "both"], default="both")
    ap.add_argument("--out", default="benchmark/results/method3.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from src.model.head_adapter import load_backbone
    group_cols = [g for g in args.group_cols.split(",") if g]
    backbone = load_backbone(args.checkpoint)

    states, labels, groups, partitions = {}, {}, {}, {}
    for entry in args.site:
        name, path = entry.split("=", 1)
        seqs, y, g, part = load_site(path, args.seq_col, args.label_col, group_cols,
                                     partition_col=args.partition_col)
        states[name] = extract_anchor_states(backbone, seqs, args.device)
        labels[name], groups[name], partitions[name] = y, g, part
        # Aggregate shape only: counts per partition role, never the path or the rows.
        roles = {r: int((part == r).sum()) for r in sorted(set(part.tolist()))}
        print(f"[{name}] {len(y)} stays, partitions={roles}")

    methods = ["probe", "xgboost"] if args.method == "both" else [args.method]
    result = {m: transportability_matrix(
        states, labels, groups, partitions, method=m,
        allow_cross_site_ensemble=args.allow_cross_site_ensemble) for m in methods}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=float))
    print(f"wrote {out}")
    # headline: our probe vs their xgboost on external-transport AUPRC + calibration
    if "probe" in result and "xgboost" in result:
        sites = [s for s in states]
        for te in sites:
            ext = [tr for tr in sites if tr != te]
            pr = np.nanmean([result["probe"][tr][te].get("auprc", np.nan) for tr in ext])
            xg = np.nanmean([result["xgboost"][tr][te].get("auprc", np.nan) for tr in ext])
            print(f"  external->{te}: AUPRC probe={pr:.3f} vs xgboost={xg:.3f}")


if __name__ == "__main__":
    main()
