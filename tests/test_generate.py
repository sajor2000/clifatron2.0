"""Tests for the from-scratch generation stack (src/model/generate.py).

Covers: sampler semantics (greedy/top-k/nucleus/forbidden ids/determinism), KV-cache
mathematical equivalence with the training forward (prefill + incremental teacher
forcing), generate() stop/min/pad semantics, vocab decode, and the simulations
parquet writer — including compatibility with the sequence viewer's ParquetSource.

The CLI checkpoint-loading path (load_generation_model) is exercised at G2 with a
real pretrain checkpoint; here the model is always a tiny randomly-initialized
CLIFEncoder with dropout=0 so cache equivalence is exact up to float32 reassociation.
"""

from __future__ import annotations

import json

import polars as pl
import pytest
import torch

from src.model.encoder import CLIFEncoder
from src.model.generate import (
    KVCache,
    SimulationWriter,
    TerminalVocabularyError,
    _cached_forward,
    classify_rollouts,
    gem_anchor_prompt,
    generate,
    load_transition_index,
    load_vocab,
    make_decoder,
    rollout_to_disposition,
    sample_logits,
    stop_token_ids_for,
    terminal_token_ids,
)
from src.viewer.sequence_viewer import ParquetSource

VOCAB = 64


def tiny_encoder(seed: int = 0) -> CLIFEncoder:
    torch.manual_seed(seed)
    cfg = {"trunk": {"d_model": 32, "n_heads": 4, "n_layers": 2, "ffn_mult": 2,
                     "dropout": 0.0, "tied_embeddings": False}}
    return CLIFEncoder(VOCAB, cfg)


def random_tokens(batch: int, length: int, seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(4, VOCAB, (batch, length), generator=g)


def minute_positions(batch: int, length: int, step: int = 30) -> torch.Tensor:
    return torch.arange(length).unsqueeze(0).expand(batch, -1) * step


# --------------------------------------------------------------------- sampler


class TestSampler:
    def test_greedy_is_argmax(self):
        logits = torch.randn(4, VOCAB)
        assert torch.equal(sample_logits(logits, temperature=0.0),
                           logits.argmax(dim=-1))

    def test_top_k_one_equals_argmax(self):
        logits = torch.randn(3, VOCAB)
        g = torch.Generator().manual_seed(0)
        assert torch.equal(sample_logits(logits, temperature=1.0, top_k=1, generator=g),
                           logits.argmax(dim=-1))

    def test_tiny_nucleus_keeps_only_top_token(self):
        # top_p smaller than any single-token mass leaves exactly one survivor.
        logits = torch.randn(3, VOCAB) * 3
        g = torch.Generator().manual_seed(0)
        for _ in range(5):
            assert torch.equal(sample_logits(logits, temperature=1.0, top_p=0.001, generator=g),
                               logits.argmax(dim=-1))

    def test_forbidden_ids_never_sampled(self):
        logits = torch.full((4, VOCAB), -10.0)
        logits[:, 5] = 50.0  # overwhelming spike on a forbidden id
        for _ in range(10):
            sampled = sample_logits(logits, temperature=1.0, min_token_ids=(5,))
            assert not bool(torch.isin(sampled, torch.tensor([5])).any())
        # greedy must also respect the ban
        assert not bool(torch.isin(sample_logits(logits, temperature=0.0,
                                                 min_token_ids=(5,)),
                                   torch.tensor([5])).any())

    def test_allowed_ids_close_the_world(self):
        # The trunk embeds target_vocab slots; the frozen CLIF vocab is far smaller.
        # Untrained slot rows (or spikes on out-of-world ids) must never leak.
        logits = torch.full((4, VOCAB), -1.0)
        logits[:, 10] = 60.0           # in-world spike
        logits[:, VOCAB - 1] = 500.0   # overwhelming out-of-world spike
        allowed = tuple(range(32))
        for _ in range(10):
            sampled = sample_logits(logits, temperature=1.0, allowed_token_ids=allowed)
            assert bool((sampled < 32).all())
        greedy = sample_logits(logits, temperature=0.0, allowed_token_ids=allowed)
        assert torch.equal(greedy, torch.full((4,), 10, dtype=torch.long))

    def test_allowed_ids_can_vary_by_batch_row(self):
        logits = torch.full((2, VOCAB), -1.0)
        logits[0, 7] = 50.0
        logits[1, 8] = 50.0
        sampled = sample_logits(
            logits,
            temperature=0.0,
            allowed_token_ids_per_row=((7,), (8,)),
        )
        assert sampled.tolist() == [7, 8]
        logits[:, VOCAB - 1] = 500.0
        for _ in range(10):
            sampled = sample_logits(
                logits,
                temperature=1.0,
                allowed_token_ids=tuple(range(32)),
                allowed_token_ids_per_row=((7,), (8,)),
            )
            assert sampled.tolist() == [7, 8]

    def test_deterministic_with_generator(self):
        logits = torch.randn(4, VOCAB)
        g1, g2 = torch.Generator().manual_seed(7), torch.Generator().manual_seed(7)
        a = torch.stack([sample_logits(logits, temperature=1.2, generator=g1) for _ in range(6)])
        b = torch.stack([sample_logits(logits, temperature=1.2, generator=g2) for _ in range(6)])
        assert torch.equal(a, b)


# --------------------------------------------------------------------- KV cache


class TestKVCacheEquivalence:
    def test_prefill_matches_full_forward(self):
        enc = tiny_encoder()
        ids, pos = random_tokens(2, 7), minute_positions(2, 7)
        full = enc(ids, pos)
        cached = _cached_forward(enc, ids, pos, KVCache(len(enc.blocks)))
        assert torch.allclose(cached, full, atol=1e-5)

    def test_incremental_matches_full_forward(self):
        enc = tiny_encoder()
        ids, pos = random_tokens(1, 9), minute_positions(1, 9)
        full = enc.lm_logits(enc(ids, pos))  # [1, 9, V]
        cache = KVCache(len(enc.blocks))
        _cached_forward(enc, ids[:, :3], pos[:, :3], cache)
        steps = []
        for t in range(3, 9):
            H = _cached_forward(enc, ids[:, t:t + 1], pos[:, t:t + 1], cache)
            steps.append(enc.lm_logits(H[:, -1:, :]))  # keep [1,1,D] → logits [1,1,V]
        assert torch.allclose(torch.cat(steps, dim=1), full[:, 3:], atol=1e-4)

    def test_multitoken_after_prefill_rejected(self):
        enc = tiny_encoder()
        ids, pos = random_tokens(1, 4), minute_positions(1, 4)
        cache = KVCache(len(enc.blocks))
        _cached_forward(enc, ids, pos, cache)
        with pytest.raises(ValueError, match="exactly one new token"):
            _cached_forward(enc, ids, pos, cache)


# --------------------------------------------------------------------- generate


class TestGenerate:
    def test_stops_on_forced_token(self):
        enc = tiny_encoder()
        ids, pos = random_tokens(1, 4), minute_positions(1, 4)
        forced = int(enc.lm_logits(enc(ids, pos)[:, -1]).argmax(dim=-1))
        res = generate(enc, ids, pos, max_new_tokens=5, temperature=0.0,
                        stop_token_ids=(forced,), min_new_tokens=0)
        assert res["lengths"].tolist() == [1]
        assert bool(res["stopped"].all())
        assert res["tokens"][0, 0].item() == forced

    def test_min_new_tokens_suppresses_stop(self):
        enc = tiny_encoder()
        ids, pos = random_tokens(1, 4), minute_positions(1, 4)
        forced = int(enc.lm_logits(enc(ids, pos)[:, -1]).argmax(dim=-1))
        res = generate(enc, ids, pos, max_new_tokens=4, temperature=0.0,
                       stop_token_ids=(forced,), min_new_tokens=2)
        assert res["tokens"][0, 0].item() != forced
        assert int(res["lengths"][0]) >= 2

    def test_batch_pads_after_length(self):
        enc = tiny_encoder()
        ids, pos = random_tokens(2, 5), minute_positions(2, 5)
        first = enc.lm_logits(enc(ids, pos)[:, -1]).argmax(dim=-1)  # [2]
        res = generate(enc, ids, pos, max_new_tokens=6, temperature=0.0,
                       stop_token_ids=tuple(first.tolist()), min_new_tokens=0)
        assert res["lengths"].tolist() == [1, 1]
        assert res["tokens"][:, 0].tolist() == first.tolist()
        assert bool(res["stopped"].all())
        assert (res["tokens"][:, 1:] == 0).all()  # pad_token_id=0 past the stop

    def test_reproducible_with_generator(self):
        enc = tiny_encoder()
        ids, pos = random_tokens(3, 4), minute_positions(3, 4)

        def run():
            g = torch.Generator().manual_seed(7)
            return generate(enc, ids, pos, max_new_tokens=8, temperature=1.3,
                            top_p=0.9, generator=g)

        a, b = run(), run()
        assert torch.equal(a["tokens"], b["tokens"])
        assert a["lengths"].tolist() == [8, 8, 8]  # no stop ids → full length

    def test_max_new_tokens_zero(self):
        enc = tiny_encoder()
        ids, pos = random_tokens(2, 4), minute_positions(2, 4)
        res = generate(enc, ids, pos, max_new_tokens=0)
        assert res["tokens"].shape == (2, 0)
        assert res["lengths"].tolist() == [0, 0]

    def test_rejects_mismatched_positions(self):
        enc = tiny_encoder()
        ids = random_tokens(2, 4)
        with pytest.raises(ValueError, match="pos_min"):
            generate(enc, ids, minute_positions(2, 5), max_new_tokens=4)


# ------------------------------------------------------- vocab + parquet writer


class TestVocabAndWriter:
    def test_load_vocab_flat_and_wrapped(self, tmp_path):
        flat = {"<bos>": 1, "<eos>": 2}
        (tmp_path / "flat.json").write_text(json.dumps(flat))
        assert load_vocab(tmp_path / "flat.json") == flat
        (tmp_path / "wrapped.json").write_text(json.dumps({"vocab": flat}))
        assert load_vocab(tmp_path / "wrapped.json") == flat

    def test_make_decoder_roundtrip_and_unk(self):
        vocab = {"<bos>": 1, "<eos>": 2, "hr=60_70": 4}
        decode = make_decoder(vocab)
        assert decode([1, 4, 2]) == ["<bos>", "hr=60_70", "<eos>"]
        assert decode([99]) == ["<unk:99>"]

    def test_writer_schema_and_viewer_compat(self, tmp_path):
        writer = SimulationWriter(dataset="gem")
        writer.add("prompt-1", 1, ["<bos>", "hr=60_70", "<eos>"], arm="ntp")
        writer.add("prompt-1", 2, "<bos> hr=60_70 <eos>")  # raw-string passthrough
        out = tmp_path / "sims.parquet"
        assert writer.write(out) == 2

        df = pl.read_parquet(out)
        for col in ("simulation_id", "hospitalization_id", "simulation_number",
                    "generated_sequence", "generated_tokens", "dataset"):
            assert col in df.columns
        assert df["simulation_id"].to_list() == [0, 1]
        assert df["dataset"].unique().to_list() == ["gem"]

        # The viewer must consume the file unchanged.
        src = ParquetSource(path=out, name="gem")
        assert src.seq_col == "generated_tokens"
        rows = src.list_rows(offset=0, limit=10, search=None)
        assert len(rows) == 2
        assert all("hr=60_70" in r["preview"] for r in rows)

    def test_writer_label_columns_survive(self, tmp_path):
        writer = SimulationWriter(dataset="gem")
        writer.add("h1", 1, ["a"], temperature=1.0)
        writer.add("h1", 2, ["b"], temperature=0.8, arm="grpo")
        out = tmp_path / "labels.parquet"
        writer.write(out)
        df = pl.read_parquet(out)
        assert df.sort("simulation_number")["arm"].to_list() == [None, "grpo"]
        assert df.sort("simulation_number")["temperature"].to_list() == [1.0, 0.8]

    def test_transition_index_reads_local_event_shards(self, tmp_path):
        events = tmp_path / "events.parquet"
        pl.DataFrame({"token": [[1, 4, 5], [1, 4, 6], [2]]}).write_parquet(events)
        assert load_transition_index(events) == {1: (4,), 4: (5, 6)}


# ------------------------------------------- U9: generate until disposition (R19)

DISPOSITIONS = ("home", "facility", "hospice", "expired", "ama", "other", "unknown")
GEM_VOCAB = {
    "<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3,
    "hr=1": 4, "hr=2": 5, "ADMISSION//ed": 6,
    **{f"DISCHARGE//{d}": 10 + i for i, d in enumerate(DISPOSITIONS)},
}
EXPIRED, HOME, EOS, FILLER = 13, 10, 2, 4


class ScriptedEncoder(CLIFEncoder):
    """A tiny trunk whose next-token logits are scripted per batch row: the k-th
    `lm_logits` call (k = 0 after prefill) puts all mass on `scripts[row][k]`, and on
    a filler token once a row's script is exhausted — so "emits DISCHARGE//expired at
    step 5" is exact under greedy decoding, independent of the random weights."""

    def __init__(self, scripts):
        torch.manual_seed(0)
        super().__init__(VOCAB, {"trunk": {"d_model": 32, "n_heads": 4, "n_layers": 1,
                                           "ffn_mult": 2, "dropout": 0.0,
                                           "tied_embeddings": False}})
        self.scripts = [list(script) for script in scripts]
        self.calls = 0
        self.prefills = 0
        self.tok_emb.register_forward_pre_hook(self._on_embed)

    def _on_embed(self, module, inputs):
        if inputs[0].size(1) > 1:  # a multi-token chunk = a new prefill: restart script
            self.calls = 0
            self.prefills += 1

    def lm_logits(self, H):
        out = torch.full((H.size(0), VOCAB), -50.0)
        for row in range(H.size(0)):
            script = self.scripts[row % len(self.scripts)]
            token = script[self.calls] if self.calls < len(script) else FILLER
            out[row, token] = 50.0
        self.calls += 1
        return out


def gem_prompt():
    return [1, 6, 4, 5], [0, 0, 30, 90]  # <bos> ADMISSION//ed hr=1 hr=2


class TestTerminalStopIds:
    def test_every_discharge_token_and_eos_are_stop_ids(self):
        ids = terminal_token_ids(GEM_VOCAB, DISPOSITIONS)
        assert ids == {10 + i: d for i, d in enumerate(DISPOSITIONS)}
        assert stop_token_ids_for(GEM_VOCAB) == tuple(sorted([*ids, EOS]))

    def test_vocab_without_terminal_tokens_is_refused(self):
        plain = {k: v for k, v in GEM_VOCAB.items() if not k.startswith("DISCHARGE//")}
        with pytest.raises(TerminalVocabularyError, match="DISCHARGE//"):
            terminal_token_ids(plain)

    def test_vocab_missing_one_allowlisted_disposition_is_refused(self):
        partial = {k: v for k, v in GEM_VOCAB.items() if k != "DISCHARGE//hospice"}
        with pytest.raises(TerminalVocabularyError, match="hospice"):
            terminal_token_ids(partial, DISPOSITIONS)


class TestRolloutToDisposition:
    def run(self, scripts, n, **kw):
        ids, pos = gem_prompt()
        kw.setdefault("max_new_tokens", 12)
        return rollout_to_disposition(
            ScriptedEncoder(scripts), ids, pos, vocab=GEM_VOCAB, dispositions=DISPOSITIONS,
            n_rollouts=n, seed=0, temperature=0.0, **kw)

    def test_expired_at_step_five_is_a_terminal_stop(self):
        [record] = self.run([[FILLER, 5, FILLER, 5, EXPIRED]], 1)
        assert record["stop_reason"] == "terminal"
        assert record["terminal_type"] == "expired"
        assert record["step"] == 5
        assert record["token_ids"][-1] == EXPIRED and len(record["token_ids"]) == 5

    def test_eos_without_a_disposition_is_an_eos_stop(self):
        [record] = self.run([[FILLER, EOS]], 1)
        assert record["stop_reason"] == "eos"
        assert record["terminal_type"] is None
        assert record["step"] == 2

    def test_never_terminating_rollout_is_censored_at_the_cap(self):
        [record] = self.run([[]], 1, max_new_tokens=7)
        assert record["stop_reason"] == "censored"
        assert record["terminal_type"] is None  # never read as survival (home)
        assert record["step"] == 7

    def test_time_to_terminal_is_unavailable_without_a_time_model(self):
        """The sampler advances a FIXED pos_step_min clock — it does not predict time,
        so a rollout's elapsed minutes are not reported as a model estimate."""
        [record] = self.run([[FILLER, EXPIRED]], 1)
        assert record["elapsed_min"] is None

    def test_rows_in_one_batch_stop_independently(self):
        records = self.run([[EXPIRED], [FILLER, FILLER, HOME], []], 3, max_new_tokens=5)
        assert [r["rollout"] for r in records] == [0, 1, 2]
        assert [(r["stop_reason"], r["terminal_type"], r["step"]) for r in records] == [
            ("terminal", "expired", 1), ("terminal", "home", 3), ("censored", None, 5)]

    def test_batch_size_chunks_do_not_change_rollouts(self):
        whole = self.run([[EXPIRED], [FILLER, HOME]], 4)
        # each chunk is a fresh prefill, so the scripted rows replay per chunk
        chunked = self.run([[EXPIRED], [FILLER, HOME]], 4, batch_size=2)
        assert [(r["stop_reason"], r["terminal_type"], r["step"]) for r in whole] == \
            [(r["stop_reason"], r["terminal_type"], r["step"]) for r in chunked]

    def test_rollouts_are_deterministic_under_a_seed(self):
        enc = tiny_encoder()
        ids, pos = gem_prompt()

        def run(seed):
            return rollout_to_disposition(
                enc, ids, pos, vocab=GEM_VOCAB, dispositions=DISPOSITIONS, n_rollouts=6,
                seed=seed, max_new_tokens=10, temperature=1.0,
                allowed_token_ids=tuple(sorted(GEM_VOCAB.values())))

        assert run(3) == run(3)

    def test_driver_refuses_a_vocab_without_terminal_tokens(self):
        plain = {k: v for k, v in GEM_VOCAB.items() if not k.startswith("DISCHARGE//")}
        ids, pos = gem_prompt()
        enc = ScriptedEncoder([[EXPIRED]])
        with pytest.raises(TerminalVocabularyError):
            rollout_to_disposition(enc, ids, pos, vocab=plain, dispositions=DISPOSITIONS,
                                   n_rollouts=2, seed=0, max_new_tokens=4)
        assert enc.prefills == 0  # refused before any generation

    def test_driver_refuses_a_prompt_that_already_holds_the_outcome(self):
        ids, pos = gem_prompt()
        with pytest.raises(ValueError, match="terminal"):
            rollout_to_disposition(
                ScriptedEncoder([[EXPIRED]]), [*ids, EXPIRED], [*pos, 120],
                vocab=GEM_VOCAB, dispositions=DISPOSITIONS, n_rollouts=1, seed=0,
                max_new_tokens=4)

    def test_classify_rollouts_from_a_generate_result(self):
        result = {"tokens": torch.tensor([[4, 13, 0], [4, 2, 0], [4, 5, 4]]),
                  "lengths": torch.tensor([2, 2, 3]),
                  "stopped": torch.tensor([True, True, False])}
        records = classify_rollouts(result, terminal_ids=terminal_token_ids(GEM_VOCAB),
                                    end_ids=(EOS,))
        assert [(r["stop_reason"], r["terminal_type"], r["step"]) for r in records] == [
            ("terminal", "expired", 2), ("eos", None, 2), ("censored", None, 3)]


class TestGemAnchorPrompt:
    def windows(self):
        # stay stream: <bos> ADM hr=1 hr=2 | hr=1 DISCHARGE//expired <eos>, anchor_idx=3
        return [
            {"continuation_index": 1, "source_start": 4, "token": [4, 13, 2],
             "pos_min": [200, 400, 400], "anchor_idx": 3},
            {"continuation_index": 0, "source_start": 0, "token": [1, 6, 4, 5],
             "pos_min": [0, 0, 30, 90], "anchor_idx": 3},
        ]

    def test_prefix_is_truncated_at_the_anchor_across_windows(self):
        assert gem_anchor_prompt(self.windows()) == ([1, 6, 4, 5], [0, 0, 30, 90])

    def test_a_prefix_reaching_the_terminal_is_refused(self):
        windows = self.windows()
        for window in windows:
            window["anchor_idx"] = 5
        with pytest.raises(ValueError, match="terminal"):
            gem_anchor_prompt(windows, terminal_ids=terminal_token_ids(GEM_VOCAB))
