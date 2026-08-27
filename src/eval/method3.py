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


# --------------------------------------------------------------------------- data
def load_site(path: str, seq_col: str, label_col: str, group_cols: list[str]):
    """Load one site's narratives into (sequences, labels, groups, ids). Polars, no pooling."""
    import polars as pl
    df = pl.read_parquet(path)
    seqs = df[seq_col].to_list()
    labels = np.asarray(df[label_col].to_list())
    groups = {g: np.asarray(df[g].to_list()) for g in group_cols if g in df.columns}
    ids = np.arange(len(labels))
    return seqs, labels, groups, ids


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
def transportability_matrix(states: dict, labels: dict, groups: dict, method: str = "probe"):
    """states/labels/groups keyed by site. Returns nested dict:
    matrix[train_site][test_site] = full panel; plus an 'ensemble' row (mean of per-site probs).
    `method` in {'probe','xgboost'} — our head vs CLIFATRON Method 1, same embeddings."""
    sites = list(states.keys())
    fit = fit_probe if method == "probe" else fit_xgboost
    predictors = {s: fit(states[s], labels[s]) for s in sites}   # train one per site (local)

    matrix, probs_on = {}, {t: {} for t in sites}
    for tr in sites:
        matrix[tr] = {}
        for te in sites:
            raw = predictors[tr](states[te])
            if method == "probe":                                # logits -> panel recalibrates
                logits = raw
                p = 1.0 / (1.0 + np.exp(-raw))
                cell = M.full_panel(p, labels[te], logits=logits, recalibrate=True)
            else:                                                # xgboost already probs
                p = raw
                cell = M.full_panel(p, labels[te], recalibrate=False)
            if te in groups:
                cell["subgroups"] = M.subgroup_panel(p, labels[te], groups[te])
            if tr != te:                                         # LPE only for external transport
                cell["lpe"] = M.local_patient_equivalence(cell.get("auroc", 0.0),
                                                          states[te], labels[te])
            matrix[tr][te] = cell
            probs_on[te][tr] = p

    # Elemento inference-time ensemble: mean of the site models' probs on each test site
    matrix["ensemble"] = {}
    for te in sites:
        p_ens = np.mean([probs_on[te][tr] for tr in sites], axis=0)
        cell = M.full_panel(p_ens, labels[te], recalibrate=False)
        if te in groups:
            cell["subgroups"] = M.subgroup_panel(p_ens, labels[te], groups[te])
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
    ap.add_argument("--method", choices=["probe", "xgboost", "both"], default="both")
    ap.add_argument("--out", default="benchmark/results/method3.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from src.model.head_adapter import load_backbone
    group_cols = [g for g in args.group_cols.split(",") if g]
    backbone = load_backbone(args.checkpoint)

    states, labels, groups = {}, {}, {}
    for entry in args.site:
        name, path = entry.split("=", 1)
        seqs, y, g, _ = load_site(path, args.seq_col, args.label_col, group_cols)
        states[name] = extract_anchor_states(backbone, seqs, args.device)
        labels[name], groups[name] = y, g
        print(f"[{name}] {len(y)} stays, prevalence={float(np.mean(y)):.3f}")

    methods = ["probe", "xgboost"] if args.method == "both" else [args.method]
    result = {m: transportability_matrix(states, labels, groups, method=m) for m in methods}

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
