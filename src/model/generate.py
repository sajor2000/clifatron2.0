"""Autoregressive generation for the from-scratch CLIFEncoder trunk — the GEM sampler.

The trunk had training but no inference path: generation previously lived only in the
vendored HF-checkpoint pipeline (external/clifatron/benchmark/generate_simulations.py,
Qwen2/GPT2 checkpoints). This module gives the from-scratch decoder a KV-cached
sampler so the same trunk we pretrain can also GENERATE CLIF token trajectories, with
outputs written in the vendored simulations parquet schema (directly consumable by
src/viewer/sequence_viewer.py).

Evidence anchors (docs/plans/2026-09-25-001-feat-icu-gem-generative-model-plan.md):
  - Pure-NTP base then RL post-training (Xiao et al. 2026, arXiv:2609.12277) — this is
    the sampling half of that loop; the GRPO trainer (G5) consumes `generate`.
  - Sampling parameters mirror the vendored pipeline (temperature, top-p, stop tokens)
    so pre/post-RL comparisons are apples-to-apples.
  - Positions: the trunk's RoPE contract is admission-relative MINUTES, so generated
    tokens advance `pos_step_min` per step (default 60 — an hourly event clock, à la
    EHR2Path's hourly state aggregation) until a learned time model lands (G4).

CLI (smoke/demo path — uniform prompt positions):
    uv run --frozen --group dev python -m src.model.generate \
        --checkpoint output/intermediate_phi/ckpt.pt \
        --vocab output/intermediate_phi/mimic/vocab.json \
        --prompts prefixes.txt --n-simulations 4 --max-new-tokens 256 \
        --output output/intermediate_phi/sims.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F

from src.data.segments import n_value_bins, vocab_segments
from src.model.encoder import CLIFEncoder, apply_rope, build_rope_cache
from src.eval.clinical_plausibility import assess_sequence, split_sequence


# --------------------------------------------------------------------- KV cache


class KVCache:
    """Per-layer key/value cache. Prefill must be the first call (chunk length > 1
    requires an empty cache); afterwards exactly one token per step."""

    def __init__(self, n_layers: int):
        self.k: list[torch.Tensor | None] = [None] * n_layers
        self.v: list[torch.Tensor | None] = [None] * n_layers

    def is_empty(self) -> bool:
        return self.k[0] is None


def _block_forward_cached(blk, x, cos, sin, cache, layer_idx, prefill):
    """Cache-aware mirror of Block.forward, reusing the block's own weights (so a
    saved checkpoint trains and generates with identical parameters)."""
    B, T, D = x.shape
    h = blk.ln1(x)
    q, k, v = blk.qkv(h).split(D, dim=2)
    q = q.view(B, T, blk.n_heads, blk.hd).transpose(1, 2)
    k = k.view(B, T, blk.n_heads, blk.hd).transpose(1, 2)
    v = v.view(B, T, blk.n_heads, blk.hd).transpose(1, 2)
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    if cache.k[layer_idx] is not None:
        k = torch.cat([cache.k[layer_idx], k], dim=2)
        v = torch.cat([cache.v[layer_idx], v], dim=2)
    cache.k[layer_idx] = k
    cache.v[layer_idx] = v
    # Prefill chunk attends causally within itself; single-token steps attend to the
    # whole cache (the query IS the last position, so no mask is needed).
    o = F.scaled_dot_product_attention(q, k, v, is_causal=(prefill and T > 1))
    o = o.transpose(1, 2).reshape(B, T, D)
    x = x + blk.drop(blk.proj(o))
    g = blk.ln2(x)
    x = x + blk.w_down(F.silu(blk.w_gate(g)) * blk.w_up(g))
    return x


def _cached_forward(enc: CLIFEncoder, token_ids: torch.Tensor, pos_min: torch.Tensor,
                    cache: KVCache) -> torch.Tensor:
    """Forward through the trunk with cache maintenance. token_ids [B, T], pos_min
    [B, T] (admission-relative minutes). T > 1 is only legal on an empty cache."""
    T = token_ids.size(1)
    if not cache.is_empty() and T != 1:
        raise ValueError(
            "chunked decoding after prefill supports exactly one new token per step"
        )
    x = enc.tok_emb(token_ids)
    cos, sin = build_rope_cache(pos_min, enc.head_dim, enc.rope_base)
    for layer_idx, blk in enumerate(enc.blocks):
        prefill = cache.k[layer_idx] is None
        x = _block_forward_cached(blk, x, cos, sin, cache, layer_idx, prefill)
    return enc.ln_f(x)


# ---------------------------------------------------------------------- sampler


def sample_logits(logits: torch.Tensor, *, temperature: float = 1.0, top_k: int | None = None,
                  top_p: float | None = 1.0, min_token_ids: tuple[int, ...] = (),
                  allowed_token_ids: tuple[int, ...] | None = None,
                  allowed_token_ids_per_row: Sequence[tuple[int, ...]] | None = None,
                  recent_token_ids_per_row: Sequence[Sequence[int]] | None = None,
                  repetition_penalty: float = 1.0,
                  generator: torch.Generator | None = None) -> torch.Tensor:
    """Sample [B] token ids from [B, V] logits.

    temperature <= 0 → greedy argmax (deterministic). top_k keeps the k most likely
    tokens; top_p keeps the smallest set whose cumulative mass >= top_p (nucleus).
    `min_token_ids` are forbidden (used for stop-token suppression before
    min_new_tokens). `allowed_token_ids` closes the world to the frozen CLIF vocab
    (hard rule 2: vocab is frozen mCIDE) — the trunk embeds `target_vocab` slots but
    the real vocab is far smaller, and untrained slot rows would otherwise leak as
    `<unk:N>` decodes. Mirrors the vendored pipeline's parameters (temperature=1.0,
    top_p=0.95) for apples-to-apples pre/post-RL comparison.
    """
    # Forbidden ids are masked in BOTH branches — greedy must respect
    # min_new_tokens stop-suppression, not just the sampling branch.
    if min_token_ids:
        for tid in min_token_ids:
            logits = logits.index_fill(-1, torch.tensor([tid], device=logits.device),
                                       float("-inf"))
    if allowed_token_ids is not None:
        keep = torch.zeros(logits.size(-1), dtype=torch.bool, device=logits.device)
        keep.index_fill_(
            0,
            torch.tensor(list(allowed_token_ids), dtype=torch.long, device=logits.device),
            True,
        )
        logits = logits.masked_fill(~keep, float("-inf"))
    if allowed_token_ids_per_row is not None:
        if len(allowed_token_ids_per_row) != logits.size(0):
            raise ValueError("allowed_token_ids_per_row must have one entry per batch row")
        base_logits = logits
        masked_rows = []
        for row, allowed in enumerate(allowed_token_ids_per_row):
            if allowed:
                row_keep = torch.zeros(logits.size(-1), dtype=torch.bool, device=logits.device)
                row_keep.index_fill_(
                    0,
                    torch.tensor(list(allowed), dtype=torch.long, device=logits.device),
                    True,
                )
                row_logits = base_logits[row].masked_fill(~row_keep, float("-inf"))
            else:
                row_logits = base_logits[row]
            if not bool(torch.isfinite(row_logits).any()):
                if allowed_token_ids is None:
                    raise ValueError("transition mask removed every candidate token")
                fallback = torch.zeros(logits.size(-1), dtype=torch.bool, device=logits.device)
                fallback.index_fill_(
                    0,
                    torch.tensor(list(allowed_token_ids), dtype=torch.long, device=logits.device),
                    True,
                )
                row_logits = base_logits[row].masked_fill(~fallback, float("-inf"))
            masked_rows.append(row_logits)
        logits = torch.stack(masked_rows, dim=0)
    if recent_token_ids_per_row is not None and repetition_penalty > 1.0:
        if len(recent_token_ids_per_row) != logits.size(0):
            raise ValueError("recent_token_ids_per_row must have one entry per batch row")
        penalized_rows = []
        for row, recent in enumerate(recent_token_ids_per_row):
            row_logits = logits[row]
            recent_ids = tuple(set(int(token_id) for token_id in recent))
            if recent_ids:
                recent_mask = torch.zeros(
                    logits.size(-1), dtype=torch.bool, device=logits.device
                )
                recent_mask.index_fill_(
                    0,
                    torch.tensor(recent_ids, dtype=torch.long, device=logits.device),
                    True,
                )
                penalized = torch.where(
                    row_logits < 0,
                    row_logits * repetition_penalty,
                    row_logits / repetition_penalty,
                )
                row_logits = torch.where(recent_mask, penalized, row_logits)
            penalized_rows.append(row_logits)
        logits = torch.stack(penalized_rows, dim=0)
    global_allowed = set(allowed_token_ids) if allowed_token_ids is not None else None

    def candidate_ids(row: int) -> tuple[int, ...] | None:
        if allowed_token_ids_per_row is None and global_allowed is None:
            return None
        if allowed_token_ids_per_row is None:
            candidates = tuple(sorted(global_allowed or ()))
        else:
            candidates = tuple(allowed_token_ids_per_row[row])
            if global_allowed is not None:
                candidates = tuple(token_id for token_id in candidates if token_id in global_allowed)
        if not candidates:
            raise ValueError("allowed-token intersection is empty")
        return candidates

    if temperature <= 0:
        greedy = []
        for row in range(logits.size(0)):
            candidates = candidate_ids(row)
            if candidates is None:
                greedy.append(logits[row].argmax())
            else:
                candidate_tensor = torch.tensor(
                    candidates, dtype=torch.long, device=logits.device
                )
                values = logits[row].index_select(0, candidate_tensor)
                greedy.append(candidate_tensor[values.argmax()])
        return torch.stack(greedy)
    logits = logits / float(temperature)
    if top_k is not None and top_k >= 1:
        kth = logits.topk(min(int(top_k), logits.size(-1)), dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = sorted_probs.cumsum(dim=-1)
        # Drop tokens whose PRECEDING mass already covers top_p; always keep the top token.
        remove_sorted = cumulative - sorted_probs >= top_p
        remove_sorted[..., 0] = False
        remove = torch.zeros_like(logits, dtype=torch.bool).scatter(-1, sorted_idx, remove_sorted)
        logits = logits.masked_fill(remove, float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    sampled = []
    for row in range(logits.size(0)):
        candidates = candidate_ids(row)
        if candidates is None:
            sampled.append(torch.multinomial(probs[row], num_samples=1, generator=generator).squeeze(0))
            continue
        candidate_tensor = torch.tensor(
            candidates, dtype=torch.long, device=logits.device
        )
        candidate_probs = probs[row].index_select(0, candidate_tensor)
        candidate_probs = torch.nan_to_num(candidate_probs, nan=0.0, posinf=0.0, neginf=0.0)
        total = candidate_probs.sum()
        if not bool(torch.isfinite(total)) or float(total) <= 0:
            candidate_probs = torch.ones_like(candidate_probs) / len(candidates)
        else:
            candidate_probs = candidate_probs / total
        sampled.append(
            candidate_tensor[
                torch.multinomial(candidate_probs, num_samples=1, generator=generator).squeeze(0)
            ]
        )
    return torch.stack(sampled)


# --------------------------------------------------------------------- generate


@torch.no_grad()
def generate(model: CLIFEncoder, input_ids: torch.Tensor, pos_min: torch.Tensor, *,
             max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None,
             top_p: float | None = 1.0, stop_token_ids: tuple[int, ...] = (),
             min_new_tokens: int = 0, pos_step_min: int = 60, pad_token_id: int = 0,
             allowed_token_ids: tuple[int, ...] | None = None,
             transition_allowed_token_ids: Mapping[int, tuple[int, ...]] | None = None,
             repetition_penalty: float = 1.0,
             generator: torch.Generator | None = None) -> dict[str, torch.Tensor]:
    """Batched autoregressive continuation of a from-scratch trunk, with KV caching.

    input_ids [B, T] hard token ids; pos_min [B, T] their admission-relative minutes
    (the trunk's RoPE contract — NOT sequence indices). Generated tokens advance
    `pos_step_min` minutes per step from the last prefix position (hourly event clock
    by default) until a learned time model lands (G4).

    Returns {"tokens": [B, max_new_tokens] (pad-filled past each row's length),
    "lengths": [B], "stopped": [B]}. A stop token IS recorded (the vendored pipeline
    also emits it) and everything after it is padding.
    """
    model.eval()
    B, T = input_ids.shape
    device = input_ids.device
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be >= 0")
    if pos_min.shape != input_ids.shape:
        raise ValueError("pos_min must match input_ids shape")
    cache = KVCache(len(model.blocks))
    H = _cached_forward(model, input_ids, pos_min, cache)
    logits = model.lm_logits(H[:, -1])
    stop_t = (torch.tensor(list(stop_token_ids), dtype=torch.long, device=device)
              if stop_token_ids else None)

    out = torch.full((B, max_new_tokens), int(pad_token_id), dtype=torch.long, device=device)
    lengths = torch.zeros(B, dtype=torch.long, device=device)
    stopped = torch.zeros(B, dtype=torch.bool, device=device)
    last_pos = pos_min[:, -1]
    previous_ids = input_ids[:, -1]
    history = [
        input_ids[row, max(0, T - 8):].tolist()
        for row in range(B)
    ]

    for step in range(max_new_tokens):
        forbid = tuple(stop_token_ids) if step < min_new_tokens else ()
        transition_allowed = None
        if transition_allowed_token_ids is not None:
            fallback_allowed = (
                allowed_token_ids
                if allowed_token_ids is not None
                else tuple(range(logits.size(-1)))
            )
            transition_allowed = []
            for token_id in previous_ids.tolist():
                candidate = transition_allowed_token_ids.get(
                    int(token_id), fallback_allowed
                )
                if step < min_new_tokens and candidate:
                    candidate = tuple(
                        value for value in candidate if value not in stop_token_ids
                    ) or fallback_allowed
                transition_allowed.append(candidate)
        next_id = sample_logits(logits, temperature=temperature, top_k=top_k, top_p=top_p,
                                min_token_ids=forbid, allowed_token_ids=allowed_token_ids,
                                allowed_token_ids_per_row=transition_allowed,
                                recent_token_ids_per_row=history,
                                repetition_penalty=repetition_penalty,
                                generator=generator)
        active = ~stopped
        out[active, step] = next_id[active]
        lengths[active] += 1
        for row in range(B):
            if bool(active[row]):
                history[row].append(int(next_id[row]))
                history[row] = history[row][-8:]
        previous_ids = next_id
        if stop_t is not None:
            hit = torch.isin(next_id, stop_t)
            stopped = stopped | (hit & active)
        if bool(stopped.all()) or step + 1 == max_new_tokens:
            break
        next_pos = last_pos + pos_step_min
        last_pos = next_pos
        # Stopped rows keep flowing through with a pad token (kept in-batch for shape
        # simplicity; their outputs are ignored). Early-exit compaction = L40 TODO.
        feed = next_id.masked_fill(~active, int(pad_token_id))
        H = _cached_forward(model, feed.unsqueeze(1), next_pos.unsqueeze(1), cache)
        logits = model.lm_logits(H[:, -1])

    return {"tokens": out, "lengths": lengths, "stopped": stopped}


# ------------------------------------------------------------- vocab + parquet


def load_vocab(path: str | Path) -> dict[str, int]:
    """token -> id map from a tokenizer `vocab.json` (accepts the `{"vocab": {...}}`
    wrapper or a flat {token: id} map)."""
    blob = json.loads(Path(path).read_text())
    vocab = blob.get("vocab", blob)
    if not isinstance(vocab, dict):
        raise ValueError(f"vocab.json has no token->id mapping: {path}")
    return vocab


def load_vocab_artifact(path: str | Path) -> dict:
    """Load the full frozen tokenizer-v2 vocab artifact (vocab + segments + manifest).
    A pre-v2 artifact (edge lists, no tokenizer version) is refused: re-tokenize."""
    blob = json.loads(Path(path).read_text())
    vocab_segments(blob)
    return blob


def load_transition_index(
    events_path: str | Path,
    *,
    max_successors: int | None = None,
) -> dict[int, tuple[int, ...]]:
    """Build an observed next-token index from local tokenizer shards.

    This is an optional generation guard, not a learned clinical grammar. A
    missing predecessor backs off to the frozen vocabulary in ``generate``.
    """
    import polars as pl

    successors: dict[int, set[int]] = {}
    frame = pl.read_parquet(events_path, columns=["token"])
    for sequence in frame["token"].to_list():
        for current, following in zip(sequence, sequence[1:]):
            successors.setdefault(int(current), set()).add(int(following))
    if max_successors is None:
        return {key: tuple(sorted(values)) for key, values in successors.items()}
    return {
        key: tuple(sorted(values)[:max_successors])
        for key, values in successors.items()
    }


def make_decoder(vocab: dict[str, int]) -> "callable":
    """ids -> token strings; unknown ids become '<unk:N>' (never silent)."""
    id_to_token = {i: t for t, i in vocab.items()}

    def decode(ids: list[int]) -> list[str]:
        return [id_to_token.get(int(i), f"<unk:{int(i)}>") for i in ids]

    return decode


class SimulationWriter:
    """Collect rollout rows and write the vendored generate_simulations parquet
    schema — simulation_id, hospitalization_id, simulation_number, generated_sequence,
    dataset (+ any per-row label columns) — so src/viewer/sequence_viewer.py consumes
    the file unchanged. Rows buffer in memory; incremental flush is an L40 TODO
    (the vendored writer appends every N hospitalizations)."""

    def __init__(self, dataset: str):
        self.dataset = dataset
        self._rows: list[dict] = []

    def add(self, hospitalization_id, simulation_number: int, tokens, **labels) -> None:
        seq = tokens if isinstance(tokens, str) else " ".join(tokens)
        self._rows.append({
            "hospitalization_id": hospitalization_id,
            "simulation_number": int(simulation_number),
            "generated_sequence": seq,
            "generated_tokens": (
                list(tokens) if not isinstance(tokens, str) else tokens.split()
            ),
            **labels,
        })

    def write(self, path: str | Path) -> int:
        import polars as pl

        keys: list[str] = []
        for row in self._rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        records = [
            {"simulation_id": i, **{k: row.get(k) for k in keys}, "dataset": self.dataset}
            for i, row in enumerate(self._rows)
        ]
        df = pl.DataFrame(records) if records else pl.DataFrame(
            schema={"simulation_id": pl.Int64, "hospitalization_id": pl.Utf8,
                    "simulation_number": pl.Int64, "generated_sequence": pl.Utf8,
                    "generated_tokens": pl.List(pl.Utf8),
                    "dataset": pl.Utf8})
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out)
        return len(records)


# -------------------------------------------------------------------------- CLI


def load_generation_model(checkpoint_path: str | Path, model_config: str | Path,
                          data_config: str | Path, vocab_artifact: dict) -> CLIFEncoder:
    """Load a pretrain-style checkpoint (schema_version 2; state under 'model') for
    generation. Falls back to a bare `enc.`-prefixed (or bare-encoder) state dict.

    The checkpoint must be bound to `vocab_artifact` (KTD7): a checkpoint trained on a
    different vocabulary or different segments — or on the previous tokenizer, which
    recorded no binding — is refused, since its token ids and bins would be decoded
    against the wrong artifact. The threshold head's value-bin count is derived from
    the artifact's segments."""
    import yaml

    from src.train.checkpoint import load_checkpoint, verify_checkpoint_binding
    from src.train.pretrain import Model

    mcfg = yaml.safe_load(Path(model_config).read_text())
    dcfg = yaml.safe_load(Path(data_config).read_text())
    n_targets = len(dcfg["target_concepts"])
    vocab_size = mcfg["trunk"].get("target_vocab", 10000)
    blob = load_checkpoint(checkpoint_path)
    verify_checkpoint_binding(blob, vocab_artifact)
    state = blob.get("model", blob)
    try:
        model = Model(vocab_size, n_targets, mcfg,
                      n_value_bins=n_value_bins(vocab_artifact))
        model.load_state_dict(state)
        return model.enc
    except (RuntimeError, KeyError):
        enc_state = {k[len("enc."):]: v for k, v in state.items() if k.startswith("enc.")}
        enc = CLIFEncoder(vocab_size, mcfg)
        enc.load_state_dict(enc_state if enc_state else state)
        return enc


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Generate CLIF token trajectories from a from-scratch checkpoint (GEM sampler)")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model-config", default="configs/model.yaml")
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument("--vocab", required=True,
                    help="the checkpoint's tokenizer-v2 vocab.json (binding check, decode "
                         "prompts + outputs)")
    ap.add_argument("--prompts", default=None,
                    help="text file, one space-joined token sequence per line "
                         "(default: a single <bos>-only prompt)")
    ap.add_argument("--n-simulations", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--min-new-tokens", type=int, default=1)
    ap.add_argument("--repetition-penalty", type=float, default=1.05,
                    help="penalize recently emitted tokens; 1 disables the guard")
    ap.add_argument("--reference-events", default=None,
                    help="optional local events.parquet for an observed-transition guard")
    ap.add_argument("--pos-step-min", type=int, default=60,
                    help="minutes assigned per generated token (hourly clock)")
    ap.add_argument("--stop-token-ids", type=int, nargs="*", default=None,
                    help="default: <eos> from --vocab, else id 2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", required=True, help="output simulations parquet")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    device = torch.device(args.device)
    vocab_artifact = load_vocab_artifact(args.vocab)
    vocab = vocab_artifact["vocab"]
    tok2id = vocab
    bos = tok2id.get("<bos>", 1)

    if args.prompts:
        if vocab is None:
            ap.error("--prompts requires --vocab (prompt lines are token strings)")
        unk = tok2id.get("<unk>", 3)
        lines = [ln for ln in Path(args.prompts).read_text().splitlines() if ln.strip()]
        prompt_tokens = [
            split_sequence(line, vocab=set(tok2id))
            for line in lines
        ]
        prompt_ids = [
            [tok2id.get(tok, unk) for tok in tokens] or [bos]
            for tokens in prompt_tokens
        ]
        hosp_ids = [f"prompt-{i + 1}" for i in range(len(lines))]
    else:
        prompt_ids, hosp_ids = [[bos]], ["prompt-1"]

    stop_ids = (tuple(args.stop_token_ids) if args.stop_token_ids is not None
               else ((vocab or {}).get("<eos>", 2),))

    enc = load_generation_model(args.checkpoint, args.model_config, args.data_config,
                                vocab_artifact).to(device)
    decoder = make_decoder(vocab) if vocab else (lambda ids: [str(i) for i in ids])
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    transitions = (
        load_transition_index(args.reference_events)
        if args.reference_events
        else None
    )

    writer = SimulationWriter(dataset="gem")
    allowed = tuple(sorted(vocab.values())) if vocab else None
    for hosp, pids in zip(hosp_ids, prompt_ids):
        n = args.n_simulations
        prompt_tokens = decoder(list(pids))
        prompt_text = " ".join(prompt_tokens)
        ids = torch.tensor([pids] * n, dtype=torch.long, device=device)
        pos = (torch.arange(len(pids), device=device, dtype=torch.long).unsqueeze(0)
               .expand(n, -1) * args.pos_step_min)
        result = generate(
            enc, ids, pos,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
            stop_token_ids=stop_ids, min_new_tokens=args.min_new_tokens,
            pos_step_min=args.pos_step_min,
            pad_token_id=tok2id.get("<pad>", 0),
            allowed_token_ids=allowed,
            transition_allowed_token_ids=transitions,
            repetition_penalty=args.repetition_penalty,
            generator=generator,
        )
        for sim in range(n):
            length = int(result["lengths"][sim])
            tokens = decoder(result["tokens"][sim, :length].tolist())
            # `prompt` column: rollout provenance (the exact prefix tokens) so
            # downstream evals pair rollouts with their real continuations
            # without positional conventions (used by src/eval/generative.py).
            review = assess_sequence(
                tokens,
                vocab=set(tok2id) if tok2id else None,
                segments=vocab_artifact["segments"],
                prompt_tokens=prompt_text.split(),
            )
            writer.add(
                hosp,
                sim + 1,
                tokens,
                prompt=prompt_text,
                prompt_tokens=prompt_tokens,
                plausibility_score=review["score"],
                plausibility_status=review["status"],
                plausibility_warning_count=review["warning_count"],
            )
    n_rows = writer.write(args.output)
    print(f"Wrote {n_rows} simulations to {args.output} (viewer-compatible schema)")


if __name__ == "__main__":
    main()
