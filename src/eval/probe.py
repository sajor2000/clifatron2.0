"""Frozen-state linear probe (plan U7; R38).

The published CLIF GEM is scored by representation-based inference: truncate each
sequence at hour 24, take the final hidden state, and fit a regularised linear model on
it. This module is that procedure for any trunk in this repo:

    trunk_states   frozen forward pass -> one hour-24 state per stay
    frozen_probe   linear head (`method3.fit_probe`) on those states, on fixed roles

The trunk is frozen here, not by the caller: its parameters lose `requires_grad` and it
is put in eval mode before any forward pass, and every forward pass runs under
`no_grad`, so nothing a probe does can reach the trunk's weights. The head is fitted on
the fit partition, its weight decay chosen on the selection partition, its temperature
on the calibration partition, and it is scored with `metrics.full_panel` on the
evaluation partition (`clif_tasks.partitioned_cell`) — the same rows as the baselines.
"""

from __future__ import annotations

import numpy as np

from src.eval.clif_tasks import Fitter, partitioned_cell, select_candidate
from src.eval.method3 import collate, extract_anchor_states, fit_probe

DEFAULT_PROBE = {"epochs": 200, "lr": 1.0e-2, "default": {"wd": 1.0e-4},
                 "grid": {"wd": [1.0e-4, 1.0e-2]}}


def freeze_trunk(trunk):
    """Turn gradients off on every trunk parameter and put the trunk in eval mode."""
    for parameter in trunk.parameters():
        parameter.requires_grad_(False)
    return trunk.eval()


def trunk_states(trunk, sequences: list[list[int]],
                 positions: list[list[int]] | None = None, *,
                 device: str = "cpu", batch_size: int = 16) -> np.ndarray:
    """Final hidden state at each sequence's last token, `[stays, d]`, trunk frozen.

    `sequences` must already end at the anchor (`clif_tasks.observation_sequences`).
    With `positions` (minutes, one per token) the trunk is called as this repo's
    encoder, `trunk(token, pos_min)`; without, as a transformers causal LM
    (`method3.extract_anchor_states`). Both are causal, so right-padding a batch does
    not change a shorter stay's state.
    """
    import torch

    if any(len(sequence) == 0 for sequence in sequences):
        raise ValueError("every stay needs at least one token to have a state")
    freeze_trunk(trunk)
    if positions is None:
        return extract_anchor_states(trunk, sequences, device, batch_size=batch_size)
    if len(positions) != len(sequences) or any(
            len(p) != len(s) for p, s in zip(positions, sequences)):
        raise ValueError("positions must align with sequences token for token")
    trunk.to(device)
    states = []
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            ids, mask = collate(sequences[start:start + batch_size])
            pos, _ = collate(positions[start:start + batch_size])
            hidden = trunk(ids.to(device), pos.to(device))            # [B, T, d]
            last = mask.sum(1).to(device) - 1
            states.append(hidden[torch.arange(hidden.size(0), device=device), last]
                          .float().cpu().numpy())
    return np.concatenate(states, axis=0)


def linear_probe_fitter(cfg: dict | None = None, *, seed: int = 0,
                        device: str = "cpu") -> Fitter:
    """A `partitioned_cell` fitter: `method3.fit_probe` on standardised states.

    States are standardised with the FIT rows' mean and scale (a frozen trunk's state
    has no particular scale, and the head is trained by a fixed number of AdamW steps).
    Weight decay is chosen on the selection rows.
    """
    import torch

    cfg = cfg or DEFAULT_PROBE

    def fit(X_fit, y_fit, X_sel, y_sel):
        X_fit = np.asarray(X_fit, dtype=np.float64)
        mean, scale = X_fit.mean(axis=0), X_fit.std(axis=0)
        scale[scale < 1e-8] = 1.0

        def standardise(X):
            return (np.asarray(X, dtype=np.float64) - mean) / scale

        def fit_one(params):
            torch.manual_seed(seed)                      # same head init for every candidate
            predict = fit_probe(standardise(X_fit), y_fit, epochs=int(cfg["epochs"]),
                                lr=float(cfg["lr"]), wd=float(params["wd"]), device=device)
            return (lambda X: predict(standardise(X))), {}

        return select_candidate(cfg, fit_one, X_sel, y_sel)

    return fit


def frozen_probe(y, partitions, roles: dict[str, str], *, site: str,
                 states: np.ndarray | None = None, trunk=None,
                 sequences: list[list[int]] | None = None,
                 positions: list[list[int]] | None = None,
                 cfg: dict | None = None, seed: int = 0, device: str = "cpu",
                 batch_size: int = 16) -> dict:
    """One task's probe cell from a trunk (with its sequences) or precomputed states.

    `y` is float with NaN for stays outside the task. For several tasks on one trunk,
    call `trunk_states` once and pass `states` here per task.
    """
    if (states is None) == (trunk is None):
        raise ValueError("pass either a trunk (with its sequences) or precomputed states")
    if states is None:
        if sequences is None:
            raise ValueError("a trunk needs the token sequences to encode")
        states = trunk_states(trunk, sequences, positions, device=device,
                              batch_size=batch_size)
    return partitioned_cell(linear_probe_fitter(cfg, seed=seed, device=device),
                            np.asarray(states), y, partitions, roles, site=site)
