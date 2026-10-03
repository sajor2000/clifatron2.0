"""Prediction heads.

(A) NextEventHead        — tied/untied next-event projection (SurvivEHR / HealthFormer)
(B) CompetingRiskHead    — per-type discrete-time CIF over horizon (SurvivEHR)
(C) ThresholdHazardHead  — ICareFM: P(concept k crosses threshold τ, in `direction`,
                           within horizon h). Learned threshold + direction embeddings,
                           discrete hazard over 48h. THIS is the zero-shot multi-
                           prediction engine — new events need no retraining.
(D) TaskHead             — K downstream binary heads on the frozen trunk.

The threshold head is what makes one model answer many outcomes: at inference,
composite events combine univariate failure probabilities under conditional
independence, e.g. circulatory failure = F_MAP(h,<65) * F_Lact(h,>2). See METHODS.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def time_bin(minutes: torch.Tensor, n_time_bins: int, horizon_hours: float) -> torch.Tensor:
    """Minutes since the anchor -> bin on ONE head's own grid (KTD4): `n_time_bins` equal
    left-closed bins over `horizon_hours`, ``floor(minutes / bin width)``, not clamped.

    The same number reads two ways. For an event it is the bin the event fell in (the
    caller clamps to the last bin: an event exactly at the horizon belongs to it). For a
    censored or event-free interval it is the count of FULLY observed bins (the caller
    clamps to `n_time_bins`); the bin the interval ended in is only partly observed and
    earns no survival credit. Integer arithmetic, so a time exactly on a bin edge can
    never round into the wrong bin."""
    horizon_minutes = round(float(horizon_hours) * 60)
    if n_time_bins <= 0 or horizon_minutes <= 0:
        raise ValueError("time grid needs a positive bin count and horizon")
    return torch.div(minutes.long() * int(n_time_bins), horizon_minutes, rounding_mode="floor")


class NextEventHead(nn.Module):
    """Next-token projection, untied by default with an explicit tied ablation."""

    def __init__(self, d: int, vocab_size: int, *, tie_weights: bool = False,
                 input_embedding: nn.Embedding | None = None):
        super().__init__()
        self.projection = nn.Linear(d, vocab_size, bias=False)
        if tie_weights:
            if input_embedding is None:
                raise ValueError("input_embedding is required when tie_weights=True")
            self.projection.weight = input_embedding.weight

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.projection(h)


def next_event_loss(
    logits: torch.Tensor,
    target_tok: torch.Tensor,
    target_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Next-token CE.

    Without `target_mask`, token[t+1] is predicted from state[t] for legacy callers.
    With `target_mask`, `target_tok[t]` is the precomputed eligible next-token target
    at state[t], and only masked positions contribute. This keeps treatments as
    context while excluding them from prediction targets.
    """
    if target_mask is not None:
        per = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_tok.reshape(-1),
            reduction="none",
            ignore_index=0,
        ).view_as(target_tok)
        m = target_mask.float()
        return (per * m).sum() / m.sum().clamp_min(1)
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        target_tok[:, 1:].reshape(-1),
        ignore_index=0,
    )


class ValueRegressionHead(nn.Module):
    """The ORA 'mark' (arXiv:2602.00541): predict the continuous value of the NEXT
    event as a Gaussian (mean/log-var), conditioned on state and the next token id.
    Masked to events that carry a numeric value. Lifts physiology tasks +33-38% vs NTP."""

    def __init__(self, d: int, vocab_size: int, emb_dim: int = 64):
        super().__init__()
        self.next_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.mlp = nn.Sequential(nn.Linear(d + emb_dim, d), nn.GELU(), nn.Linear(d, 2))

    def loss(self, h: torch.Tensor, next_tok: torch.Tensor,
             next_val: torch.Tensor, val_mask: torch.Tensor) -> torch.Tensor:
        # h[t] predicts value at t+1; align by dropping last position.
        q = torch.cat([h[:, :-1], self.next_emb(next_tok[:, 1:])], dim=-1)
        mu, logvar = self.mlp(q).unbind(-1)                 # [B, T-1]
        mu = mu.float()
        logvar = logvar.float().clamp(-10.0, 10.0)
        y = next_val[:, 1:].float()
        m = val_mask[:, 1:].float()
        nll = 0.5 * (logvar + (y - mu) ** 2 / logvar.exp())  # Gaussian NLL (drop const)
        return (nll * m).sum() / m.sum().clamp_min(1)

    def loss_aligned(self, h: torch.Tensor, target_tok: torch.Tensor,
                     target_val: torch.Tensor, val_mask: torch.Tensor) -> torch.Tensor:
        """Gaussian mark loss for precomputed target tensors aligned at state[t]."""
        q = torch.cat([h, self.next_emb(target_tok)], dim=-1)
        mu, logvar = self.mlp(q).unbind(-1)
        mu = mu.float()
        logvar = logvar.float().clamp(-10.0, 10.0)
        target_val = target_val.float()
        m = val_mask.float()
        nll = 0.5 * (logvar + (target_val - mu) ** 2 / logvar.exp())
        return (nll * m).sum() / m.sum().clamp_min(1)


class CompetingRiskHead(nn.Module):
    """Discrete-time competing-risks CIF with a conditional per-time-bin distribution
    over K modeled causes plus no event (SurvivEHR / Dynamic-DeepHit likelihood).

    Each time bin computes (K+1) logits, normalized via softmax.  Event mass for
    cause k at bin t is: S(t-1) * q[k, t] where S(t-1) is the event-free probability
    through bin t-1 and q[k,t] is the conditional cause probability for that bin.
    Censoring contributes S(c-1) where c is the number of fully observed bins: the bin
    the censoring time fell in is only partly observed and earns no credit (KTD4).

    Invariants: S(t) + sum_k CIF_k(t) = 1 for all t; all CIFs are nonnegative and
    monotone; logits must have at least 2 channels (1 cause + no event minimum)."""

    def __init__(self, d: int, n_types: int, n_time_bins: int):
        super().__init__()
        self.n_types, self.n_bins = n_types, n_time_bins
        self.fc = nn.Linear(d, (n_types + 1) * n_time_bins)

    def _logits(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc(h).view(*h.shape[:-1], self.n_types + 1, self.n_bins)

    def _distribution(self, h: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self._logits(h), dim=-2)   # [..., K+1, B]

    def cif(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (cause CIF [...,K,B], event_free [...,B])."""
        q = self._distribution(h)                        # [..., K+1, B]
        q_no_event = q[..., -1, :]                       # [..., B]
        q_cause = q[..., :-1, :]                         # [..., K, B]
        event_free = torch.cumprod(q_no_event, dim=-1)   # [..., B]
        surv_shifted = torch.cat([torch.ones_like(event_free[..., :1]),
                                   event_free[..., :-1]], dim=-1)
        cif = surv_shifted.unsqueeze(-2) * q_cause       # [..., K, B]
        cif = torch.cumsum(cif, dim=-1)                  # monotone CIF
        return cif, event_free

    def loss(self, h: torch.Tensor, event_type: torch.Tensor,
             dt_bin: torch.Tensor, censored: torch.Tensor | None = None) -> torch.Tensor:
        """NLL under cause-plus-no-event parameterization.

        event_type: 0..K-1, ignored when censored==True.
        dt_bin: `time_bin` of the observed time on this head's grid.  For events it is
                the bin in which the event occurred (clamped to the last bin).  For
                censored samples it is the number of FULLY observed bins, 0..n_bins:
                survival is credited through bin dt_bin-1 and no further.
        censored: optional bool mask; when not supplied, event_type < 0 signals
                  censoring (legacy).
        """
        if self.n_types < 1:
            raise ValueError("n_types must be >= 1 for a valid CR head")
        if censored is None:
            censored = event_type < 0
        N = h.size(0)
        idx = torch.arange(N, device=h.device)
        # Log space: with K causes over B bins the event-free probability through the
        # horizon starts near (K+1)**-B, far below any clamp a product would need, and a
        # clamped likelihood has no gradient.
        log_q = torch.log_softmax(self._logits(h).float(), dim=-2)       # [N, K+1, B]
        log_q_event = log_q[idx, event_type.clamp(min=0)]                 # [N, B]
        # log S(c): event-free through the first c bins (S(0) = 1), c = 0..n_bins.
        log_survival = torch.cat(
            [torch.zeros_like(log_q[:, -1, :1]), torch.cumsum(log_q[:, -1, :], dim=-1)],
            dim=-1,
        )
        b = dt_bin.clamp(min=0, max=self.n_bins - 1)
        ll_event = log_survival[idx, b] + log_q_event[idx, b]
        ll_censor = log_survival[idx, dt_bin.clamp(min=0, max=self.n_bins)]
        log_lik = torch.where(censored, ll_censor, ll_event)
        return -log_lik.mean()


class ThresholdHazardHead(nn.Module):
    """ICareFM head. Input: patient state H_t, a queried threshold τ (as its value-bin
    index) and a direction (0=below,1=above) for target concept k. Output: discrete
    hazard over `n_time_bins` hours -> cumulative failure prob F_k(h|H_t,τ).

    `n_value_bins` comes from the frozen vocabulary (`segments.n_value_bins`: the largest
    segment count over every binned concept, plus one), never a default: a concept with
    more bins than the embedding would otherwise index past it."""

    def __init__(self, d: int, n_targets: int, n_time_bins: int,
                 n_value_bins: int, thr_dim: int = 32):
        super().__init__()
        self.n_targets, self.n_bins = n_targets, n_time_bins
        self.n_value_bins = int(n_value_bins)
        self.thr_emb = nn.Embedding(n_value_bins + 1, thr_dim)   # learned threshold embedding
        self.dir_emb = nn.Embedding(2, thr_dim)                  # learned direction embedding
        self.target_emb = nn.Embedding(n_targets, thr_dim)
        self.mlp = nn.Sequential(
            nn.Linear(d + 3 * thr_dim, d), nn.GELU(), nn.Linear(d, n_time_bins)
        )

    def _logits(self, h_last, target_idx, tau_bin, direction) -> torch.Tensor:
        if tau_bin.numel() and tau_bin.is_cuda:
            # No host sync on the GPU: the check runs on the device and fails the next
            # kernel launch (a device-side assert) instead of stalling every forward.
            lo, hi = torch.aminmax(tau_bin)
            torch._assert_async(
                (lo >= 0) & (hi < self.thr_emb.num_embeddings),
                "threshold value bin out of range: n_value_bins must be derived from the "
                "vocabulary the query's tau_bin was computed against")
        elif tau_bin.numel():
            lo, hi = torch.stack(torch.aminmax(tau_bin)).tolist()   # one host sync
            if int(lo) < 0 or int(hi) >= self.thr_emb.num_embeddings:
                raise ValueError(
                    f"threshold value bin out of range [0, {self.thr_emb.num_embeddings}): "
                    f"got {int(lo)}..{int(hi)}; n_value_bins must be derived from the "
                    "vocabulary the query's tau_bin was computed against"
                )
        q = torch.cat(
            [h_last, self.target_emb(target_idx), self.thr_emb(tau_bin), self.dir_emb(direction)], dim=-1
        )
        return self.mlp(q)

    def hazard(self, h_last, target_idx, tau_bin, direction) -> torch.Tensor:
        return torch.sigmoid(
            self._logits(h_last, target_idx, tau_bin, direction).float()
        )  # [N, n_time_bins] discrete hazard

    def cumulative_failure(self, h_last, target_idx, tau_bin, direction) -> torch.Tensor:
        """F(h|H_t,τ) = 1 - prod(1 - hazard) over time bins. Used zero-shot."""
        haz = self.hazard(h_last, target_idx, tau_bin, direction).clamp(1e-6, 1 - 1e-6)
        return 1 - torch.cumprod(1 - haz, dim=-1)

    def loss(self, h_last, target_idx, tau_bin, direction, crossed_bin, observed_bin=None) -> torch.Tensor:
        """crossed_bin: first hour the target crossed τ within horizon, or -1 if never.
        observed_bin: for a query with no crossing, the number of FULLY observed bins
        (`time_bin` on this head's grid; default: the whole horizon) — survival is
        credited for those bins only (KTD4).
        Discrete-time hazard NLL with random τ sampled by the trainer."""
        logits = self._logits(h_last, target_idx, tau_bin, direction).float()
        N, Bn = logits.shape
        t = torch.arange(Bn, device=logits.device)[None].expand(N, Bn)
        crossed = crossed_bin[:, None]
        if observed_bin is None:
            observed = torch.full_like(crossed, Bn)
        else:
            observed = observed_bin[:, None].clamp(min=0, max=Bn)
        # crossed < 0 means no crossing observed through `observed`; contribute
        # survival only through the observed at-risk interval, not necessarily the
        # full horizon when follow-up is censored early.
        observed_until = torch.where(crossed >= 0, crossed.clamp(min=0), observed)
        surv_ll = torch.where(
            t < observed_until,
            F.logsigmoid(-logits),
            torch.zeros_like(logits),
        )
        event_ll = torch.where(
            (t == crossed) & (crossed >= 0),
            F.logsigmoid(logits),
            torch.zeros_like(logits),
        )
        return -(surv_ll + event_ll).sum(-1).mean()

    def predict_with_confidence(
        self, h_last, target_idx, tau_bin, direction
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Selective prediction with uncertainty (arxiv:2603.02719).

        Returns (failure_prob, confidence) at the full horizon.
        confidence is derived from the variance of the cumulative hazard
        across time bins. High variance → uncertain prediction → candidate
        for deferral to human review.

        The deferral threshold is set at calibration time and should be
        tuned per-outcome on a held-out validation set.
        """
        cf = self.cumulative_failure(h_last, target_idx, tau_bin, direction)
        f_horizon = cf[:, -1]
        haz = self.hazard(h_last, target_idx, tau_bin, direction)
        haz_var = haz * (1 - haz)
        cum_uncertainty = torch.sqrt(haz_var.sum(dim=-1))
        confidence = 1.0 - torch.tanh(cum_uncertainty)
        return f_horizon, confidence


def composite_or(*failure_probs: torch.Tensor) -> torch.Tensor:
    """Disjunction under conditional independence: P(any) = 1 - prod(1 - F_i)."""
    keep = torch.ones_like(failure_probs[0])
    for f in failure_probs:
        keep = keep * (1 - f)
    return 1 - keep


def composite_and(*failure_probs: torch.Tensor) -> torch.Tensor:
    """Conjunction under conditional independence: P(all) = prod(F_i)."""
    out = torch.ones_like(failure_probs[0])
    for f in failure_probs:
        out = out * f
    return out


class TaskHead(nn.Module):
    """K downstream binary heads on the frozen trunk; masked BCE for missing labels."""

    def __init__(self, d: int, n_tasks: int):
        super().__init__()
        self.fc = nn.Linear(d, n_tasks)

    def forward(self, h_last: torch.Tensor) -> torch.Tensor:
        return self.fc(h_last)

    def loss(self, h_last, labels, mask) -> torch.Tensor:
        logits = self.fc(h_last)
        per = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction="none")
        return (per * mask).sum() / mask.sum().clamp_min(1)
