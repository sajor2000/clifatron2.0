"""U5: full-hospitalization training path and batching (R31, R34, R37, R39; KTD1, KTD2).

`pretrain.build_loaders(representation="gem")` reads one or more site directories that
share ONE frozen vocabulary, side by side in one loader (sites are never pooled on disk),
labels each stay in-stream with the `gem_tte` target builder on the WHOLE stay (KTD1), and
batches under DDP with `DistributedTokenBudgetBatchSampler`: batches formed globally
(deterministic per seed and epoch) and dealt to ranks, every rank yielding the same
number of batches.

What is proven here, all on synthetic data:
  - two sites with the same vocabulary load together, their stays kept apart by a
    per-site key; a site or shard bound to another vocabulary is refused, and so is a
    vocabulary fit on a sample;
  - the sampler's contract: equal per-rank length, disjoint batches apart from at most
    world-size minus one repeated batches, full coverage, `set_epoch` reshuffling, and a
    padding bound on a skewed length distribution — also inside a two-process gloo run
    that trains a few updates of the full objective and leaves both ranks identical;
  - a stay longer than one window keeps every window for labelling;
  - the continuous-fused value channel loads on this representation;
  - the measured parameter count lands in the manifest; run length in passes resolves to
    the expected update count; three CPU updates on the synthetic site are finite for every
    head; the loader reports its resident memory per event.
"""

import copy
import json
import math
import os
import random
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import polars as pl
import torch
import torch.distributed as dist
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from src.data.collate import collate_model_samples  # noqa: E402
from src.data.dataset import DistributedTokenBudgetBatchSampler, ModelDataset  # noqa: E402
from src.data.segments import artifact_binding, n_value_bins  # noqa: E402
from src.data.targets import TargetContractError  # noqa: E402
from src.data.threshold_grid import THRESHOLD_KINDS, load_thresholds  # noqa: E402

CPU = torch.device("cpu")
MODEL_CFG = yaml.safe_load((ROOT / "configs/model.yaml").read_text())
SITES = ("site_a", "site_b")
VALIDATION_STAYS = {"synth-001", "synth-002"}     # moved to the validation partition


# ------------------------------------------------------------------ synthetic corpus

def _registry() -> dict:
    full = load_thresholds()
    return {**full, **{kind: tuple(t for t in full[kind] if t.concept == "map")
                       for kind in THRESHOLD_KINDS}}


def tiny_mcfg(**weights) -> dict:
    mcfg = {
        "trunk": {"d_model": 16, "n_layers": 1, "n_heads": 2, "ffn_mult": 2, "dropout": 0.0,
                  "rope_base": 10000.0, "tied_embeddings": False, "target_vocab": 10000,
                  # As the committed config: the synthetic shards are cut for it, and
                  # training refuses a shard cut the other way.
                  "continuation_header": MODEL_CFG["trunk"]["continuation_header"],
                  "max_tokens": MODEL_CFG["trunk"]["max_tokens"]},
        "heads": copy.deepcopy(MODEL_CFG["heads"]),
        "in_stream": {"anchors_per_window": 3, "queries_per_anchor": 2},
    }
    mcfg["heads"]["threshold_hazard"]["threshold_embed_dim"] = 4
    for name, weight in weights.items():
        mcfg["heads"][name]["weight"] = weight
    return mcfg


def tiny_tcfg(ckpt_dir, *, token_budget: int = 256, per_gpu: int = 4, total_steps: int = 3,
              passes=None) -> dict:
    schedule = {"warmup_steps": 1, "total_steps": total_steps, "cosine_decay": True}
    if passes is not None:
        schedule["passes"] = passes
    return {
        "optimizer": {"lr": 1e-2, "weight_decay": 0.1, "betas": [0.9, 0.95], "grad_clip": 1.0},
        "schedule": schedule,
        "batch": {"per_gpu": per_gpu, "grad_accum": 1},
        "runtime": {"precision": "bf16", "num_workers": 0, "log_every": 1000,
                    "ckpt_every": 1000, "ckpt_dir": str(ckpt_dir),
                    "token_budget": token_budget},
        "eval_schedule": {"val_every": 1000},
    }


def _set_partitions(path: Path, keep: set[str] | None = None) -> None:
    """Move VALIDATION_STAYS to the validation partition (all windows of a stay move
    together); `keep` restricts the shard to those stays."""
    frame = pl.read_parquet(path)
    if keep is not None:
        frame = frame.filter(pl.col("hosp_id").is_in(sorted(keep)))
    frame = frame.with_columns(
        pl.when(pl.col("hosp_id").is_in(sorted(VALIDATION_STAYS)))
        .then(pl.lit("validation")).otherwise(pl.col("partition")).alias("partition"))
    frame.write_parquet(path)


def build_corpus(work: Path) -> dict:
    """Two development sites tokenized with ONE frozen vocabulary (site A builds it, site
    B imports it, as a non-reference site does), a site with its own vocabulary, a site
    whose vocabulary is marked as fit on a sample, and value stats from site A's train
    stays. Site B carries a subset of the stays, with the SAME raw identifiers as site A:
    only the per-site key keeps the two apart."""
    from test_gem_artifact import MAX_TOKENS, _build_site

    from src.data.tokenize import tokenize_site
    from src.data.value_stats import compute_value_stats_from_events, write_value_stats
    from src.eval.synthetic_bundle import FIXTURE_DATA_CONFIG, FIXTURE_POLICY, SYNTHETIC_SITE

    old_cwd = os.getcwd()
    os.chdir(work)
    try:
        raw, episodes, cfg = _build_site(work)
        kw = {"episodes": episodes, "artifact_policy": FIXTURE_POLICY}
        out = work / "output/intermediate_phi"
        dirs = {name: out / name for name in (*SITES, "other_vocab")}
        tokenize_site(cfg, SYNTHETIC_SITE, raw, dirs["site_a"], None, **kw)
        blob = json.loads((dirs["site_a"] / "vocab.json").read_text())
        for name in SITES:
            if name != "site_a":
                tokenize_site(cfg, SYNTHETIC_SITE, raw, dirs[name], blob, **kw)
            tokenize_site(cfg, SYNTHETIC_SITE, raw, dirs[name], blob,
                          trajectory="hospitalization", max_tokens=MAX_TOKENS, **kw)
        other = copy.deepcopy(cfg)
        other["value_binning"]["n_bins"] = 8
        tokenize_site(other, SYNTHETIC_SITE, raw, dirs["other_vocab"], None, **kw)
        other_blob = json.loads((dirs["other_vocab"] / "vocab.json").read_text())
        tokenize_site(other, SYNTHETIC_SITE, raw, dirs["other_vocab"], other_blob,
                      trajectory="hospitalization", max_tokens=MAX_TOKENS, **kw)
    finally:
        os.chdir(old_cwd)
    gem = {name: d / "gem_events.parquet" for name, d in dirs.items()}
    _set_partitions(gem["site_a"])
    _set_partitions(gem["site_b"], keep={f"synth-{i:03d}" for i in range(12)})
    _set_partitions(gem["other_vocab"])
    # A vocabulary fit on a verification sample: same content (same binding), but its
    # provenance says sample — training must refuse it (KTD9).
    sample = out / "sample_vocab"
    shutil.copytree(dirs["site_b"], sample)
    sample_blob = copy.deepcopy(blob)
    sample_blob["manifest"]["provenance"].update({"sample": True, "sample_size": 12})
    (sample / "vocab.json").write_text(json.dumps(sample_blob))
    stats = compute_value_stats_from_events(gem["site_a"], partition="train", min_count=1)
    stats_path = write_value_stats(stats, out / "value_stats.json", vocab=blob["vocab"],
                                   segments=blob["segments"], fit_partition_name="train")
    return {"blob": blob, "other_blob": other_blob, "gem": gem,
            "sample": sample / "gem_events.parquet", "value_stats": stats_path,
            "dcfg": FIXTURE_DATA_CONFIG, "registry": _registry()}


def gem_loaders(corpus: dict, sites=SITES, *, tcfg=None, blob=None, dry_run=False,
                value_channel=False, mcfg=None, td="."):
    from src.train.pretrain import build_loaders, embedding_vocab_size

    blob = corpus["blob"] if blob is None else blob
    mcfg = tiny_mcfg() if mcfg is None else mcfg
    paths = {name: corpus["gem"].get(name, name) for name in sites} \
        if not isinstance(sites, dict) else sites
    return build_loaders(
        paths, representation="gem", binding=artifact_binding(blob), vocab_blob=blob,
        tcfg=tcfg or tiny_tcfg(td), mcfg=mcfg, vocab_size=embedding_vocab_size(blob, mcfg),
        value_stats_path=corpus["value_stats"], dry_run=dry_run, dcfg=corpus["dcfg"],
        thresholds=corpus["registry"], value_channel=value_channel)


def skewed_lengths(n: int = 400, seed: int = 0) -> list[int]:
    """Heavy-tailed (log-normal) lengths, like CLIF stays: most short, a few very long."""
    rng = random.Random(seed)
    return [max(1, min(4096, int(rng.lognormvariate(4.5, 1.1)))) for _ in range(n)]


# --------------------------------------------------------------------- sampler (unit)

def rank_batches(lengths, world, *, epoch=0, **kw) -> list[list[list[int]]]:
    out = []
    for rank in range(world):
        sampler = DistributedTokenBudgetBatchSampler(
            lengths, num_replicas=world, rank=rank, seed=3, **kw)
        sampler.set_epoch(epoch)
        batches = list(sampler)
        if len(batches) != len(sampler):
            raise AssertionError("len() disagrees with the batches yielded")
        out.append(batches)
    return out


class RankAwareSamplerTest(unittest.TestCase):
    KW = {"max_batch_tokens": 1024, "max_batch_size": 8}

    def test_every_rank_yields_the_same_number_of_batches_on_skewed_lengths(self):
        lengths = skewed_lengths()
        for world in (2, 3, 4):
            per_rank = rank_batches(lengths, world, **self.KW)
            with self.subTest(world=world):
                self.assertEqual(len({len(b) for b in per_rank}), 1)

    def test_batches_are_disjoint_apart_from_the_evening_out_and_cover_every_sample(self):
        lengths = skewed_lengths()
        for world in (2, 3):
            per_rank = rank_batches(lengths, world, **self.KW)
            sampler = DistributedTokenBudgetBatchSampler(lengths, num_replicas=world,
                                                         rank=0, seed=3, **self.KW)
            global_batches = sampler.global_batches()
            repeats = sampler.padding_batches
            with self.subTest(world=world):
                self.assertLessEqual(repeats, world - 1)
                self.assertEqual((len(global_batches) + repeats) % world, 0)
                flat = [i for batches in per_rank for batch in batches for i in batch]
                self.assertEqual(set(flat), set(range(len(lengths))))
                # Each index appears once, except the samples of the repeated batches.
                extra = len(flat) - len(lengths)
                self.assertEqual(extra, sum(len(b) for b in global_batches[:repeats]))
                # Ranks share nothing outside those repeated batches.
                repeated = {i for b in global_batches[:repeats] for i in b}
                seen = [set(i for b in batches for i in b) - repeated for batches in per_rank]
                for r in range(world):
                    for s in range(r + 1, world):
                        self.assertEqual(seen[r] & seen[s], set())

    def test_global_batches_respect_the_token_budget_and_size_cap(self):
        lengths = skewed_lengths()
        sampler = DistributedTokenBudgetBatchSampler(lengths, num_replicas=2, rank=0,
                                                     seed=3, **self.KW)
        for batch in sampler.global_batches():
            self.assertLessEqual(len(batch), 8)
            if len(batch) > 1:      # a single over-budget sequence batches alone
                self.assertLessEqual(len(batch) * max(lengths[i] for i in batch), 1024)

    def test_padding_stays_under_the_bound_on_a_skewed_distribution(self):
        """Length grouping keeps pad tokens below 15% of the padded batch slots, against
        about 60% for the same budget filled in arrival order."""
        lengths = skewed_lengths(2000, seed=1)
        sampler = DistributedTokenBudgetBatchSampler(lengths, num_replicas=2, rank=0,
                                                     seed=3, max_batch_tokens=4096,
                                                     max_batch_size=16)
        batches = sampler.global_batches()
        slots = sum(len(b) * max(lengths[i] for i in b) for b in batches)
        real = sum(lengths)
        self.assertLess(1 - real / slots, 0.15)

    def test_set_epoch_reshuffles_deterministically_and_ranks_agree(self):
        lengths = skewed_lengths()
        e0 = rank_batches(lengths, 2, epoch=0, **self.KW)
        e1 = rank_batches(lengths, 2, epoch=1, **self.KW)
        self.assertNotEqual(e0, e1)
        self.assertEqual(e0, rank_batches(lengths, 2, epoch=0, **self.KW))

    def test_fewer_batches_than_ranks_still_gives_every_rank_one(self):
        per_rank = rank_batches([5, 5], 4, max_batch_tokens=100, max_batch_size=8)
        self.assertEqual([len(b) for b in per_rank], [1, 1, 1, 1])

    def test_invalid_arguments_are_refused(self):
        with self.assertRaises(ValueError):
            DistributedTokenBudgetBatchSampler([3], num_replicas=2, rank=2,
                                               max_batch_size=4)
        with self.assertRaises(ValueError):
            DistributedTokenBudgetBatchSampler([3], num_replicas=1, rank=0)
        with self.assertRaises(ValueError):
            DistributedTokenBudgetBatchSampler([], num_replicas=1, rank=0, max_batch_size=4)


# ------------------------------------------------------------------ loader (corpus)

class GemLoaderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._td = tempfile.TemporaryDirectory()
        cls.work = Path(cls._td.name)
        cls.corpus = build_corpus(cls.work)
        cls.loaders = gem_loaders(cls.corpus, td=cls.work / "ckpt")

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    def test_two_sites_with_one_vocabulary_load_together_kept_apart_by_site(self):
        dataset = self.loaders.train_dataset
        frames = {name: pl.read_parquet(self.corpus["gem"][name]) for name in SITES}
        expected = sum(f.filter(pl.col("partition") == "train").height
                       for f in frames.values())
        self.assertEqual(len(dataset), expected)
        keys = set(dataset._gem_streams)
        for name in SITES:
            stays = set(frames[name].filter(pl.col("partition") == "train")["hosp_id"])
            self.assertEqual({k for k in keys if k.startswith(f"{name}:")},
                             {f"{name}:{s}" for s in stays})
        # The same raw identifier at both sites is two stays, never one merged stream.
        self.assertIn("site_a:synth-000", keys)
        self.assertIn("site_b:synth-000", keys)
        val = self.loaders.validation_dataset
        self.assertEqual({k.split(":", 1)[1] for k in val._gem_streams}, VALIDATION_STAYS)
        self.assertEqual(self.loaders.sites, list(SITES))

    def test_a_site_bound_to_another_vocabulary_is_refused(self):
        with self.assertRaisesRegex(Exception, "vocabulary|binding|hash"):
            gem_loaders(self.corpus, ("site_a", "other_vocab"), td=self.work)

    def test_a_shard_row_bound_to_another_vocabulary_is_refused(self):
        foreign = self.work / "foreign"
        shutil.copytree(self.corpus["gem"]["site_b"].parent, foreign)
        frame = pl.read_parquet(foreign / "gem_events.parquet")
        other_hashes = pl.read_parquet(self.corpus["gem"]["other_vocab"])["artifact_hashes"][0]
        frame = frame.with_columns(
            pl.struct(**{k: pl.lit(v) for k, v in other_hashes.items()})
            .alias("artifact_hashes"))
        frame.write_parquet(foreign / "gem_events.parquet")
        sites = {"site_a": self.corpus["gem"]["site_a"],
                 "foreign": foreign / "gem_events.parquet"}
        with self.assertRaises(TargetContractError) as ctx:
            gem_loaders(self.corpus, sites, td=self.work)
        self.assertRegex(str(ctx.exception), "hash|binding|vocabulary")

    def _recut(self, name: str, **cut) -> Path:
        """A copy of site_b's shard whose rows record another window-cut mode (`cut`: the
        fields to set; a None value drops the field, as a shard cut before they existed)."""
        copy_dir = self.work / name
        shutil.rmtree(copy_dir, ignore_errors=True)
        shutil.copytree(self.corpus["gem"]["site_b"].parent, copy_dir)
        shutil.rmtree(copy_dir / "gem_cache", ignore_errors=True)
        path = copy_dir / "gem_events.parquet"
        frame = pl.read_parquet(path)
        hashes = dict(frame["artifact_hashes"][0])
        for key, value in cut.items():
            hashes.pop(key) if value is None else hashes.__setitem__(key, value)
        frame.with_columns(pl.struct(**{k: pl.lit(v) for k, v in hashes.items()})
                           .alias("artifact_hashes")).write_parquet(path)
        return path

    def test_the_shard_records_how_its_windows_were_cut(self):
        hashes = pl.read_parquet(self.corpus["gem"]["site_a"])["artifact_hashes"].unique()
        self.assertEqual(hashes.len(), 1)
        recorded = hashes[0]
        self.assertEqual(recorded["continuation_header"], "on")
        self.assertGreaterEqual(int(recorded["header_length"]), 2)    # <bos>, ADMISSION//

    def test_a_shard_cut_without_the_header_is_refused_when_training_with_it(self):
        # The cut flag is bound into the shard: a shard cut at full length (or one that
        # predates the flag) cannot take the header, and is refused when the loader is
        # built - not trained, and not discovered at the first over-budget window.
        for label, cut in (("off", {"continuation_header": "off", "header_length": "0"}),
                           ("legacy", {"continuation_header": None, "header_length": None})):
            with self.subTest(shard=label):
                path = self._recut(f"cut_{label}", **cut)
                with self.assertRaisesRegex(TargetContractError, "continuation_header"):
                    gem_loaders(self.corpus, {"site_a": path}, td=self.work)

    def test_a_shard_cut_with_the_header_is_refused_when_training_without_it(self):
        mcfg = tiny_mcfg()
        mcfg["trunk"]["continuation_header"] = False
        with self.assertRaisesRegex(TargetContractError, "continuation_header"):
            gem_loaders(self.corpus, mcfg=mcfg, td=self.work)
        # Evaluation may still read it without the header (no require_cut_match).
        loaded = self.loaders.train_dataset
        ModelDataset(loaded.corpus, representation="gem", target_builder=loaded.target_builder,
                     expected_hashes=loaded.expected_hashes)

    def test_the_cut_mode_is_part_of_the_corpus_cache_key(self):
        from src.data.dataset import GemCorpus, _shard_cut_mode

        modes = {label: _shard_cut_mode(self._recut(f"key_{label}", **cut))
                 for label, cut in (("on", {}),
                                    ("off", {"continuation_header": "off", "header_length": "0"}),
                                    ("legacy", {"continuation_header": None,
                                                "header_length": None}))}
        self.assertEqual(modes["off"], modes["legacy"])      # no field means cut at full length
        self.assertNotEqual(modes["on"], modes["off"])
        path = self._recut("key_cache")
        GemCorpus.from_parquet(path, partition="train", cache=True)
        (cache,) = [c for c in (path.parent / "gem_cache").iterdir() if c.is_dir()]
        meta = cache / "meta.json"
        self.assertEqual(json.loads(meta.read_text())["cut_mode"], modes["on"])
        # A cache recording another cut mode is refused, never read.
        meta.write_text(meta.read_text().replace('\\"on\\"', '\\"off\\"'))
        with self.assertRaisesRegex(TargetContractError, "stale"):
            GemCorpus.from_parquet(path, partition="train", cache=True)

    def test_a_vocabulary_built_from_a_sample_is_refused(self):
        sites = {"site_a": self.corpus["gem"]["site_a"], "sample": self.corpus["sample"]}
        with self.assertRaisesRegex(SystemExit, "sample"):
            gem_loaders(self.corpus, sites, td=self.work)
        sample_blob = json.loads((self.corpus["sample"].parent / "vocab.json").read_text())
        with self.assertRaisesRegex(SystemExit, "sample"):
            gem_loaders(self.corpus, {"sample": self.corpus["sample"]}, blob=sample_blob,
                        td=self.work)
        # A dry run (smoke only) may read it.
        gem_loaders(self.corpus, sites, dry_run=True, td=self.work)

    def test_duplicate_or_empty_site_names_are_refused(self):
        with self.assertRaisesRegex(ValueError, "site"):
            gem_loaders(self.corpus, {"": self.corpus["gem"]["site_a"]}, td=self.work)
        with self.assertRaisesRegex(ValueError, "site"):
            gem_loaders(self.corpus, {"a:b": self.corpus["gem"]["site_a"]}, td=self.work)

    def test_samples_carry_in_stream_anchors(self):
        sample = self.loaders.train_dataset[0]
        self.assertIn("anchors", sample["segments"][0])
        batch = next(iter(self.loaders.train))
        self.assertGreater(batch["anchor_idx"].numel(), 0)
        self.assertGreater(int(batch["th_mask"].sum()), 0)

    def test_a_stay_longer_than_one_window_keeps_all_its_windows_for_labelling(self):
        from test_gem_artifact import LONG_STAY

        dataset = self.loaders.train_dataset
        key = f"site_a:{LONG_STAY}"
        n_windows = pl.read_parquet(self.corpus["gem"]["site_a"]).filter(
            pl.col("hosp_id") == LONG_STAY).height
        self.assertGreater(n_windows, 100)
        stream = dataset._gem_streams[key]
        self.assertEqual(len(stream["windows"]), n_windows)
        self.assertEqual(stream["windows"][-1][1], len(stream["token"]))
        built = dataset.target_builder.build(stream, epoch=0)
        indices = [i for i, r in enumerate(dataset.records)
                   if r.get("episode_key") == key]
        total = sum(len(dataset[i]["segments"][0]["anchors"]) for i in indices)
        self.assertEqual(total, len(built["anchors"]))

    def test_continuation_windows_lack_the_header_unless_it_is_reinserted(self):
        """Product authority, 2026-10-03: the second and later windows of a long stay start
        mid-stream (no <bos>, ADMISSION//, static tokens); `continuation_header` puts the
        stay's header back without changing labels."""
        from test_gem_artifact import LONG_STAY

        from src.data.dataset import ModelDataset, header_token_ids, stay_header_length

        loaded = self.loaders.train_dataset
        # The header-less view of the same corpus (evaluation reads a header-cut shard
        # without the header; training would refuse that, `require_cut_match`).
        base = ModelDataset(loaded.corpus, representation="gem",
                            target_builder=loaded.target_builder,
                            expected_hashes=loaded.expected_hashes)
        header = header_token_ids(self.corpus["blob"]["vocab"])
        key = f"site_a:{LONG_STAY}"
        indices = [i for i, r in enumerate(base.records) if r.get("episode_key") == key]
        first, second = base[indices[0]], base[indices[1]]
        h = stay_header_length(first["input_ids"], header)
        self.assertGreaterEqual(h, 2)                         # <bos>, ADMISSION//, statics
        self.assertEqual(stay_header_length(second["input_ids"], header), 0)   # the gap

        fixed = ModelDataset(base.corpus, representation="gem", target_builder=base.target_builder,
                             expected_hashes=base.expected_hashes, continuation_header=header)
        again_first, again_second = fixed[indices[0]], fixed[indices[1]]
        self.assertEqual(again_first["input_ids"], first["input_ids"])          # window 1 as is
        self.assertEqual(again_second["input_ids"][:h], first["input_ids"][:h])
        self.assertEqual(again_second["input_ids"][h:], second["input_ids"])
        self.assertEqual(again_second["pos_min"][:h], first["pos_min"][:h])
        self.assertFalse(any(again_second["ntp_mask"][:h]))
        self.assertEqual(again_second["ntp_target"][h:], second["ntp_target"])
        seg, old = again_second["segments"][0], second["segments"][0]
        self.assertEqual((seg["packed_start"], seg["packed_end"]), (0, h + len(second["input_ids"])))
        self.assertEqual([a["offset"] - h for a in seg["anchors"]], [a["offset"] for a in old["anchors"]])
        self.assertEqual([a["cr"] for a in seg["anchors"]], [a["cr"] for a in old["anchors"]])
        self.assertEqual(fixed.sample_lengths()[indices[1]], h + len(second["input_ids"]))
        # Deterministic.
        self.assertEqual(fixed[indices[1]]["input_ids"], again_second["input_ids"])
        # Counted against the budget: a full window plus its header is refused.
        from src.data.targets import TargetContractError
        capped = ModelDataset(base.corpus, representation="gem", target_builder=base.target_builder,
                              expected_hashes=base.expected_hashes, continuation_header=header,
                              max_tokens=len(second["input_ids"]))
        with self.assertRaisesRegex(TargetContractError, "max_tokens"):
            capped[indices[1]]
        # ...and refused when the loader is built (the batch sampler reads
        # sample_lengths), not hours into an epoch: shards cut without room for the header.
        with self.assertRaisesRegex(TargetContractError, "gem_window_bounds_with_header"):
            capped.sample_lengths()

    def test_header_aware_window_bounds_keep_every_sample_within_the_budget(self):
        from src.data.dataset import gem_window_bounds_with_header

        bounds = gem_window_bounds_with_header(1000, 64, 6)
        self.assertEqual(bounds[0], (0, 64))
        self.assertEqual(bounds[-1][1], 1000)
        self.assertTrue(all(b - a <= 58 for a, b in bounds[1:]))
        self.assertTrue(all(b1 == a2 for (_, b1), (a2, _) in zip(bounds, bounds[1:])))
        self.assertGreater(bounds[-1][1] - bounds[-1][0], 1)

    def test_continuous_fused_value_channel_loads_on_the_full_hospitalization_shards(self):
        loaders = gem_loaders(self.corpus, value_channel=True, td=self.work)
        dataset = loaders.train_dataset
        stats = dataset.target_builder.value_stats
        some_value = False
        for index in range(len(dataset)):
            sample = dataset[index]
            record = dataset.records[index]
            n = len(sample["input_ids"])
            self.assertEqual(len(sample["input_value"]), n)
            self.assertEqual(len(sample["input_value_mask"]), n)
            for token, value, z, mask in zip(record["token"], record["value"],
                                             sample["input_value"],
                                             sample["input_value_mask"]):
                if value is None or token not in stats:
                    self.assertEqual((z, mask), (0.0, False))
                elif mask:
                    some_value = True
                    center, scale = stats[token]
                    self.assertAlmostEqual(z, (value - center) / scale, places=5)
            if some_value:
                break
        self.assertTrue(some_value)
        batch = collate_model_samples([dataset[0], dataset[1]])
        self.assertIn("input_value", batch)
        self.assertEqual(batch["input_value"].shape, batch["input_ids"].shape)

    def test_loader_reports_resident_memory_per_event(self):
        memory = self.loaders.memory
        events = sum(len(r["token"]) for r in self.loaders.train_dataset.records) + sum(
            len(r["token"]) for r in self.loaders.validation_dataset.records)
        self.assertEqual(memory["events"], events)
        self.assertGreater(memory["rss_after_bytes"], 0)
        self.assertTrue(math.isfinite(memory["bytes_per_event"]))

    def test_twenty_four_hour_representation_is_unchanged_by_default(self):
        from src.train.pretrain import build_loaders

        with self.assertRaisesRegex(ValueError, "representation"):
            build_loaders(self.corpus["gem"]["site_a"], representation="bogus",
                          binding=artifact_binding(self.corpus["blob"]),
                          vocab_blob=self.corpus["blob"], tcfg=tiny_tcfg(self.work),
                          mcfg=tiny_mcfg(), vocab_size=64)


class EmbeddingAndRunLengthTest(unittest.TestCase):
    def test_embedding_size_is_derived_from_the_vocabulary(self):
        from src.train.pretrain import embedding_vocab_size

        blob = {"vocab": {"<pad>": 0, "a": 1, "b": 7}}
        self.assertEqual(embedding_vocab_size(blob, tiny_mcfg()), 8)
        capped = tiny_mcfg()
        capped["trunk"]["target_vocab"] = 5
        with self.assertRaisesRegex(SystemExit, "target_vocab"):
            embedding_vocab_size(blob, capped)

    def test_run_length_in_passes_resolves_to_the_update_count(self):
        from src.train.engine import TrainConfig, resolve_total_steps

        tcfg = tiny_tcfg(".", total_steps=999, passes=2.5)
        tcfg["batch"]["grad_accum"] = 4
        # 37 batches per pass at accumulation 4 -> 10 updates per pass (the last partial).
        self.assertEqual(resolve_total_steps(tcfg, batches_per_pass=37), 25)
        cfg = TrainConfig({}, tcfg, {}, resolve_total_steps(tcfg, batches_per_pass=37))
        self.assertEqual((cfg.total_steps, cfg.passes), (25, 2.5))
        # Without passes, total_steps is the run length as before.
        self.assertEqual(resolve_total_steps(tiny_tcfg(".", total_steps=7), 37), 7)
        for bad in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                resolve_total_steps(tiny_tcfg(".", passes=bad), 37)


class SyntheticTrainingTest(unittest.TestCase):
    """Three CPU updates of the full objective through `engine.train` on the two-site
    loader; the manifest records the measured parameter count."""

    @classmethod
    def setUpClass(cls):
        cls._td = tempfile.TemporaryDirectory()
        cls.work = Path(cls._td.name)
        cls.corpus = build_corpus(cls.work)

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    def test_three_cpu_updates_of_the_full_objective_and_the_measured_parameter_count(self):
        from src.train.engine import TrainConfig, train
        from src.train.pretrain import (
            Model,
            build_optimizer,
            build_scheduler,
            embedding_vocab_size,
        )

        mcfg = tiny_mcfg()
        tcfg = tiny_tcfg(self.work / "ckpt")
        loaders = gem_loaders(self.corpus, mcfg=mcfg, tcfg=tcfg)
        blob = self.corpus["blob"]
        torch.manual_seed(0)
        model = Model(embedding_vocab_size(blob, mcfg), len(self.corpus["dcfg"]["target_concepts"]),
                      mcfg, n_value_bins=n_value_bins(blob))
        self.assertEqual(model.enc.tok_emb.num_embeddings, max(blob["vocab"].values()) + 1)
        losses = []
        model.register_forward_hook(lambda _m, _i, out: losses.append(
            {k: float(v.detach()) for k, v in out.items()}))
        opt = build_optimizer(model, lr=1e-3, weight_decay=0.1, betas=(0.9, 0.95))
        _, manifest = train(model, loaders.train, None, opt, build_scheduler(opt, 3, 1),
                            TrainConfig({}, tcfg, mcfg, 3), CPU,
                            vocab_binding=artifact_binding(blob))
        self.assertEqual(manifest.ledger["optimizer_updates"], 3)
        self.assertEqual(len(losses), 3)
        for name in ("ntp", "cr", "th", "val", "total"):
            values = [step[name] for step in losses]
            with self.subTest(head=name):
                self.assertTrue(all(math.isfinite(v) for v in values), values)
                self.assertTrue(any(v > 0 for v in values), values)
        heads = sum(p.numel() for module in (model.cr, model.th, model.vr)
                    for p in module.parameters())
        total = sum(p.numel() for p in model.parameters())
        self.assertEqual(manifest.parameters,
                         {"trunk": total - heads, "heads": heads, "total": total})
        self.assertEqual(manifest.to_dict()["parameters"]["total"], total)


# ------------------------------------------------------------- two-process gloo run

def scenario_gem_two_site(local: int, out_dir: Path) -> dict:
    """One rank: the two-site loader under DDP, its batch plan, and `engine.train` of the
    full objective for a few updates."""
    from src.train.engine import TrainConfig, train, wrap_ddp
    from src.train.pretrain import Model, build_optimizer, build_scheduler, embedding_vocab_size

    corpus = torch.load(out_dir / "corpus.pt", weights_only=False)
    mcfg = tiny_mcfg()
    tcfg = tiny_tcfg(out_dir / "ckpt", token_budget=192, per_gpu=4, total_steps=4)
    loaders = gem_loaders(corpus, mcfg=mcfg, tcfg=tcfg)
    sampler = loaders.train.batch_sampler
    sampler.set_epoch(0)
    plan = list(sampler)
    skewed = DistributedTokenBudgetBatchSampler(
        skewed_lengths(), num_replicas=dist.get_world_size(), rank=dist.get_rank(),
        max_batch_tokens=1024, max_batch_size=8, seed=3)
    blob = corpus["blob"]
    torch.manual_seed(0)
    model = wrap_ddp(Model(embedding_vocab_size(blob, mcfg),
                           len(corpus["dcfg"]["target_concepts"]), mcfg,
                           n_value_bins=n_value_bins(blob)), CPU, local)
    opt = build_optimizer(model, lr=1e-2, weight_decay=0.1, betas=(0.9, 0.95))
    _, manifest = train(model, loaders.train, None, opt, build_scheduler(opt, 4, 1),
                        TrainConfig({}, tcfg, mcfg, 4), CPU,
                        vocab_binding=artifact_binding(blob))
    return {"len": len(loaders.train), "plan": plan, "n": len(loaders.train_dataset),
            "global": sampler.global_batches(), "padding": sampler.padding_batches,
            "skewed_len": len(skewed), "skewed": list(skewed),
            "dataset_len": len(loaders.train_dataset),
            "updates": manifest.ledger["optimizer_updates"],
            "state": {k: v.detach().clone() for k, v in model.module.state_dict().items()}}


SCENARIOS = {"gem_two_site": scenario_gem_two_site}


@unittest.skipUnless(dist.is_available() and dist.is_gloo_available(),
                     "torch.distributed gloo backend is not built on this platform")
class TwoRankGemTrainingTest(unittest.TestCase):
    def test_two_ranks_share_the_batch_plan_and_end_with_identical_parameters(self):
        from test_ddp_multihead import launch_two_ranks

        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            corpus = build_corpus(out_dir)
            torch.save(corpus, out_dir / "corpus.pt")
            codes, logs, timed_out = launch_two_ranks("gem_two_site", out_dir,
                                                      script=Path(__file__).resolve())
            if timed_out or any(codes):
                tails = "\n".join(f"--- rank {r} (exit {c}) ---\n{log[-3000:]}"
                                  for r, (c, log) in enumerate(zip(codes, logs)))
                self.fail(f"two-rank gem launch failed:\n{tails}")
            r0, r1 = (torch.load(out_dir / f"rank_{r}.pt", weights_only=False)
                      for r in range(2))
        # Same batch count on both ranks, for the corpus and for a skewed distribution.
        self.assertEqual(r0["len"], r1["len"])
        self.assertEqual(r0["skewed_len"], r1["skewed_len"])
        self.assertEqual(len(r0["skewed"]), len(r1["skewed"]))
        # Every rank holds every window (whole stays for labelling); the sampler only deals.
        self.assertEqual(r0["dataset_len"], r1["dataset_len"])
        self.assertEqual(r0["global"], r1["global"])
        repeats = r0["padding"]
        self.assertLessEqual(repeats, 1)
        repeated = {i for b in r0["global"][:repeats] for i in b}
        s0 = {i for b in r0["plan"] for i in b}
        s1 = {i for b in r1["plan"] for i in b}
        self.assertEqual((s0 & s1) - repeated, set())
        self.assertEqual(s0 | s1, set(range(r0["n"])))
        k0 = {i for b in r0["skewed"] for i in b}
        k1 = {i for b in r1["skewed"] for i in b}
        self.assertEqual(k0 | k1, set(range(len(skewed_lengths()))))
        # The full objective trained a few updates and the replicas did not drift.
        self.assertEqual((r0["updates"], r1["updates"]), (4, 4))
        for name in r0["state"]:
            self.assertTrue(torch.equal(r0["state"][name], r1["state"][name]), name)


def _rank_main(scenario: str, out_dir: str) -> None:
    from src.train.engine import setup_ddp

    torch.set_num_threads(1)
    local, _ = setup_ddp(allow_cpu=True)
    try:
        result = SCENARIOS[scenario](local, Path(out_dir))
        torch.save(result, Path(out_dir) / f"rank_{dist.get_rank()}.pt")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] in SCENARIOS:
        _rank_main(sys.argv[1], sys.argv[2])
    else:
        unittest.main()
