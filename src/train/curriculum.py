"""NTP→TTE curriculum scheduler (RESEARCH.md §3; model.yaml curriculum: ntp_then_tte).

Phase 1 (warmup): next-event token only. Freezes TTE heads.
Phase 2 (transition): linear blend from pure NTP to full ORA marked-TTE.
Phase 3 (mixed): full ORA objective with NTP as low-weight auxiliary.
"""

from __future__ import annotations

from typing import NamedTuple


class Mix(NamedTuple):
    w_ntp: float
    w_cr: float
    w_th: float
    w_val: float
    train_heads: bool


def curriculum_weights(step: int, total_steps: int,
                       warmup_frac: float = 0.15,
                       transition_frac: float = 0.05) -> Mix:
    """Return (w_ntp, w_cr, w_th, w_val, train_heads) for the current step.

    warmup_frac: fraction of total steps spent on pure NTP
    transition_frac: fraction for the linear blend

    Phase boundaries:
      0 .. warmup_end                  NTP only, TTE heads frozen
      warmup_end .. transition_end     linear blend, TTE heads unfrozen
      transition_end .. total_steps    full ORA + auxiliary NTP
    """
    warmup_end = int(total_steps * warmup_frac)
    transition_end = warmup_end + int(total_steps * transition_frac)

    if step < warmup_end:
        return Mix(1.0, 0.0, 0.0, 0.0, False)

    if step < transition_end:
        progress = (step - warmup_end) / max(transition_end - warmup_end, 1)
        return Mix(0.2, progress, progress, 0.5 * progress, True)

    return Mix(0.2, 1.0, 1.0, 0.5, True)