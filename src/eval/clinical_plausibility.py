"""Local, heuristic checks for human inspection of CLIF token rollouts.

These checks are deliberately conservative. They identify structural problems
and distributional oddities that make a rollout worth reviewing; they are not
clinical validation, a safety score, or a substitute for clinician review.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

SPECIAL_TOKENS = frozenset({"<pad>", "<bos>", "<eos>", "<unk>"})


def split_sequence(sequence: str, *, vocab: set[str] | None = None) -> list[str]:
    """Split a space-joined sequence while preserving vocab tokens containing spaces."""
    words = sequence.split()
    if not vocab:
        return words
    phrases = sorted(
        {
            tuple(token.split())
            for token in vocab
            if " " in token and token.strip()
        },
        key=lambda phrase: (-len(phrase), phrase),
    )
    tokens: list[str] = []
    index = 0
    while index < len(words):
        match = next(
            (
                phrase
                for phrase in phrases
                if words[index : index + len(phrase)] == list(phrase)
            ),
            None,
        )
        if match is None:
            tokens.append(words[index])
            index += 1
        else:
            tokens.append(" ".join(match))
            index += len(match)
    return tokens


def split_fused_token(token: str) -> tuple[str, int | None]:
    """Return the concept and numeric bin for a CLIF ``concept=bin`` token."""
    if "=" not in token:
        return token, None
    concept, suffix = token.rsplit("=", 1)
    if suffix.isdigit() and concept:
        return concept, int(suffix)
    return token, None


def _warning(code: str, severity: str, message: str, **extra: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "code": code,
        "severity": severity,
        "message": message,
    }
    result.update(extra)
    return result


def assess_sequence(
    tokens: Sequence[str],
    *,
    vocab: set[str] | None = None,
    edges: Mapping[str, Sequence[float]] | None = None,
    prompt_tokens: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Assess structural and lightweight clinical-sequence heuristics.

    The result is intentionally explainable: every score contribution is tied
    to a warning. ``prompt_tokens=None`` means prompt provenance was not
    supplied, while an empty prompt is a valid, explicit prompt.
    """
    token_list = [str(token) for token in tokens]
    warnings: list[dict[str, Any]] = []
    unknown = [token for token in token_list if vocab is not None and token not in vocab]

    if unknown:
        warnings.append(_warning(
            "unknown-token",
            "error",
            f"{len(unknown)} token(s) are outside the frozen vocabulary.",
            count=len(unknown),
            sample=unknown[:10],
        ))

    invalid_bins: list[str] = []
    for token in token_list:
        concept, bin_index = split_fused_token(token)
        if bin_index is None or edges is None or concept not in edges:
            continue
        n_bins = len(edges[concept]) + 1
        if bin_index < 0 or bin_index >= n_bins:
            invalid_bins.append(token)
    if invalid_bins:
        warnings.append(_warning(
            "invalid-bin",
            "error",
            f"{len(invalid_bins)} numeric bin token(s) are outside the frozen edges.",
            count=len(invalid_bins),
            sample=invalid_bins[:10],
        ))

    special_positions = [
        (index, token)
        for index, token in enumerate(token_list)
        if token in SPECIAL_TOKENS
    ]
    structural_specials = [
        (index, token)
        for index, token in special_positions
        if token in {"<pad>", "<bos>", "<unk>"}
    ]
    if structural_specials:
        warnings.append(_warning(
            "structural-special-token",
            "error",
            "Continuation contains padding, BOS, or UNK special token(s).",
            count=len(structural_specials),
            sample=[token for _, token in structural_specials[:10]],
        ))
    eos_positions = [index for index, token in special_positions if token == "<eos>"]
    if eos_positions and eos_positions[-1] != len(token_list) - 1:
        warnings.append(_warning(
            "tokens-after-eos",
            "warning",
            "Tokens occur after an EOS marker.",
            first_index=eos_positions[0],
        ))

    max_token_run = 0
    max_token_run_token: str | None = None
    run_start = 0
    for index, token in enumerate(token_list):
        if index == 0 or token != token_list[index - 1]:
            run_start = index
        run_length = index - run_start + 1
        if run_length > max_token_run:
            max_token_run = run_length
            max_token_run_token = token
    if max_token_run >= 4:
        warnings.append(_warning(
            "repeated-token-run",
            "warning",
            f"Token {max_token_run_token!r} repeats {max_token_run} times consecutively.",
            token=max_token_run_token,
            run_length=max_token_run,
        ))

    concepts = [split_fused_token(token)[0] for token in token_list]
    max_concept_window = 0
    max_concept_window_concept: str | None = None
    for end in range(len(concepts)):
        start = max(0, end - 7)
        window = concepts[start : end + 1]
        if not window:
            continue
        counts = {concept: window.count(concept) for concept in set(window)}
        concept, count = max(counts.items(), key=lambda item: item[1])
        if count > max_concept_window:
            max_concept_window = count
            max_concept_window_concept = concept
    if max_concept_window >= 6:
        warnings.append(_warning(
            "concept-loop",
            "warning",
            f"Concept {max_concept_window_concept!r} dominates a short event window.",
            concept=max_concept_window_concept,
            count=max_concept_window,
            window=8,
        ))

    combined = list(prompt_tokens or []) + token_list
    has_icu_context = any(split_fused_token(token)[0] == "icu" for token in combined)
    if combined and not has_icu_context:
        warnings.append(_warning(
            "missing-icu-context",
            "info",
            "No ICU-context marker was found in the supplied prompt plus sequence.",
        ))
    if prompt_tokens is None and token_list:
        warnings.append(_warning(
            "missing-prompt-provenance",
            "info",
            "No prompt provenance was supplied, so prefix continuity cannot be checked.",
        ))

    n = len(token_list)
    error_count = sum(warning["severity"] == "error" for warning in warnings)
    warning_count = sum(warning["severity"] == "warning" for warning in warnings)
    info_count = sum(warning["severity"] == "info" for warning in warnings)
    score = 1.0
    score -= min(0.55, (len(unknown) / max(n, 1)) * 1.2)
    score -= min(0.35, (len(invalid_bins) / max(n, 1)) * 1.5)
    score -= min(0.25, len(structural_specials) / max(n, 1))
    score -= 0.08 if max_token_run >= 4 else 0.0
    score -= 0.05 if max_concept_window >= 6 else 0.0
    score -= 0.02 if not has_icu_context and combined else 0.0
    score = round(max(0.0, min(1.0, score)), 4)
    status = "poor" if error_count else "review" if warning_count else "good"

    return {
        "score": score,
        "status": status,
        "interpretation": (
            "Heuristic structural review only; not clinical validation."
        ),
        "warning_count": len(warnings),
        "error_count": error_count,
        "info_count": info_count,
        "warnings": warnings,
        "prompt": {
            "present": prompt_tokens is not None,
            "n_tokens": len(prompt_tokens) if prompt_tokens is not None else None,
        },
        "stats": {
            "n_tokens": n,
            "n_unique": len(set(token_list)),
            "unknown_count": len(unknown),
            "invalid_bin_count": len(invalid_bins),
            "max_token_run": max_token_run,
            "max_token_run_token": max_token_run_token,
            "max_concept_window": max_concept_window,
            "max_concept_window_concept": max_concept_window_concept,
        },
    }
