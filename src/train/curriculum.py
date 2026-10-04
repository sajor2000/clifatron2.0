"""NTP→TTE curriculum and the objective arms (KTD5, R31; model.yaml curriculum: ntp_then_tte).

Phase 1 (warmup): next-event token only; the TTE heads carry zero weight.
Phase 2 (transition): linear blend from pure NTP to the configured objective weights.
Phase 3 (mixed): the configured weights (the full objective keeps NTP as a low-weight
auxiliary).

`step` is the count of optimizer updates already applied — the training engine's update
counter, restored from the checkpoint on resume — never a count of forward calls.

An objective arm (`configs/objective_arms.yaml`) is one choice of those configured weights
plus whether the curriculum runs. Loss balancing is fixed weights; nothing else exists.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Mapping, NamedTuple, Sequence

import yaml

HEADS = ("next_event", "competing_risk", "threshold_hazard", "value_regression")
LOSS_BALANCING = ("fixed",)
CURRICULA = ("ntp_then_tte", "none")
OBJECTIVE_ARMS_CONFIG = "configs/objective_arms.yaml"
ARM_KEYS = ("description", "heads", "curriculum", "tags")


class Mix(NamedTuple):
    w_ntp: float
    w_cr: float
    w_th: float
    w_val: float
    train_heads: bool


def curriculum_weights(step: int, total_steps: int,
                       warmup_frac: float = 0.15,
                       transition_frac: float = 0.05,
                       *, target: Sequence[float] = (0.2, 1.0, 1.0, 0.5)) -> Mix:
    """Return (w_ntp, w_cr, w_th, w_val, train_heads) for the current step.

    warmup_frac: fraction of total steps spent on pure NTP
    transition_frac: fraction for the linear blend
    target: the configured (w_ntp, w_cr, w_th, w_val) the schedule ends on

    Phase boundaries:
      0 .. warmup_end                  NTP only, TTE heads at zero weight
      warmup_end .. transition_end     linear blend toward `target`
      transition_end .. total_steps    `target`
    """
    if len(target) != 4:
        raise ValueError("target must be (w_ntp, w_cr, w_th, w_val)")
    if min(target) < 0:
        raise ValueError("objective weights cannot be negative")
    w_ntp, w_cr, w_th, w_val = (float(w) for w in target)
    warmup_end = int(total_steps * warmup_frac)
    transition_end = warmup_end + int(total_steps * transition_frac)

    if step < warmup_end:
        return Mix(1.0, 0.0, 0.0, 0.0, False)

    if step < transition_end:
        progress = (step - warmup_end) / max(transition_end - warmup_end, 1)
        return Mix(1.0 + progress * (w_ntp - 1.0), progress * w_cr, progress * w_th,
                   progress * w_val, True)

    return Mix(w_ntp, w_cr, w_th, w_val, True)


def curriculum_enabled(mcfg: Mapping) -> bool:
    """Check a model config's `loss_balancing` and `curriculum`; True when the
    curriculum runs. Both default to the plain objective (fixed weights, no curriculum).

    Unknown values are refused. `loss_balancing: uncertainty` was accepted for months
    and never implemented, so every such run trained with fixed weights unannounced."""
    balancing = mcfg.get("loss_balancing", "fixed")
    if balancing not in LOSS_BALANCING:
        raise ValueError(
            f"loss_balancing {balancing!r} is not implemented: the objective is a sum with "
            "the fixed weights heads.*.weight. Set loss_balancing: fixed")
    curriculum = mcfg.get("curriculum", "none")
    if curriculum not in CURRICULA:
        raise ValueError(f"curriculum {curriculum!r} is unknown; expected one of "
                         f"{', '.join(CURRICULA)} (ntp_then_tte | none)")
    return curriculum == "ntp_then_tte"


class ObjectiveArm(NamedTuple):
    name: str
    weights: dict[str, float]      # configured weight per head (HEADS)
    curriculum: bool               # next-token warm-up, then the blend to `weights`


def load_objective_arms(path: str | Path = OBJECTIVE_ARMS_CONFIG) -> dict[str, ObjectiveArm]:
    """Every arm of `configs/objective_arms.yaml`, validated.

    An arm sets the four head weights and the curriculum, nothing else: steps, batch
    size, accumulation and token budget are the run's train config, so the arms cannot
    differ in compute (R31)."""
    arms = {}
    for name, spec in (yaml.safe_load(Path(path).read_text()).get("arms") or {}).items():
        extra = sorted(set(spec) - set(ARM_KEYS))
        if extra:
            raise ValueError(
                f"objective arm {name}: {extra} is not an objective setting. Arms share "
                "the train config's steps, batch size and token budget (equal compute)")
        heads = spec.get("heads") or {}
        if set(heads) != set(HEADS):
            raise ValueError(f"objective arm {name}: heads must set exactly {list(HEADS)}")
        weights = {head: float(heads[head]) for head in HEADS}
        if min(weights.values()) < 0:
            raise ValueError(f"objective arm {name}: weights cannot be negative")
        arms[name] = ObjectiveArm(
            name, weights, curriculum_enabled({"curriculum": spec.get("curriculum")}))
    return arms


def resolve_objective_arm(name: str,
                          path: str | Path = OBJECTIVE_ARMS_CONFIG) -> ObjectiveArm:
    """Arm name -> (weights, curriculum flag); an unknown arm is refused."""
    arms = load_objective_arms(path)
    if name not in arms:
        raise ValueError(f"unknown objective arm {name!r}; choices: {', '.join(arms)}")
    return arms[name]


def apply_objective_arm(mcfg: Mapping, arm: ObjectiveArm) -> dict:
    """A copy of the model config with the arm's weights and curriculum. Every head stays
    constructed, so all arms have the same parameters and checkpoints stay compatible."""
    mcfg = copy.deepcopy(dict(mcfg))
    for head, weight in arm.weights.items():
        spec = mcfg["heads"].setdefault(head, {})
        if weight > 0 and not spec.get("enabled", True):
            raise ValueError(f"objective arm {arm.name} weights heads.{head}, which the "
                             "model config disables")
        spec["weight"] = weight
    mcfg["curriculum"] = "ntp_then_tte" if arm.curriculum else "none"
    mcfg["objective_arm"] = arm.name
    return mcfg


def describe_schedule(total_steps: int, weights: Mapping[str, float], curriculum: bool, *,
                      arm: str | None = None, warmup_frac: float = 0.15,
                      transition_frac: float = 0.05) -> str:
    """One line naming the weight schedule a run was configured for (logged at start)."""
    target = " ".join(f"{head}={float(weights[head]):g}" for head in HEADS)
    head = f"objective arm={arm or 'model config'} loss_balancing=fixed"
    if not curriculum:
        return f"{head} curriculum=none: updates 0-{total_steps - 1} at {target}"
    warmup_end = int(total_steps * warmup_frac)
    transition_end = warmup_end + int(total_steps * transition_frac)
    phases = [(0, warmup_end, "next_event only"),
              (warmup_end, transition_end, "linear blend"),
              (transition_end, total_steps, f"at {target}")]
    return f"{head} curriculum=ntp_then_tte: " + "; ".join(
        f"updates {lo}-{hi - 1} {what}" for lo, hi, what in phases if hi > lo)


def compute_budget(tcfg: Mapping) -> dict[str, int]:
    """The compute an objective arm trains with — the train config's, for every arm."""
    return {
        "total_steps": int(tcfg["schedule"]["total_steps"]),
        "per_gpu_batch": int(tcfg["batch"]["per_gpu"]),
        "grad_accum": int(tcfg["batch"].get("grad_accum", 1)),
        "token_budget": int(tcfg["runtime"].get("token_budget", 0) or 0),
    }
