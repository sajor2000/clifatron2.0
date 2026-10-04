"""U5: the full-hospitalization dataset is columnar and yields what the list-of-dicts
implementation yielded.

`ModelDataset(representation="gem")` stores its windows as flat numpy arrays with
per-window and per-stay offsets (`src.data.dataset.GemCorpus`), optionally memory-mapped
from an on-disk cache beside the shard, and assembles a stay's windows on access.
`_reference_samples` below is the previous implementation (whole stays rebuilt as Python
lists, targets built once per stay and sliced per window) kept as the oracle: every sample
of the synthetic two-site corpus must be identical to it, element type for element type
and float bit for float bit, over two epochs, with and without the value channel, and
the collated batch tensors must be byte-identical.
"""

import pickle
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from test_gem_training_path import SITES, build_corpus, gem_loaders  # noqa: E402

from src.data.collate import collate_model_samples  # noqa: E402
from src.data.dataset import PACKED_SCHEMA_VERSION, GemCorpus, ModelDataset  # noqa: E402
from src.data.segments import artifact_binding  # noqa: E402
from src.data.targets import TargetContractError  # noqa: E402
from src.data.tokenize_continuous import normalize_value  # noqa: E402


# ------------------------------------------------- the previous implementation (oracle)

_FIELDS = ("token", "pos_min", "value", "target_eligible")


def _reference_samples(records, builder, *, epoch, value_channel):
    """The list-of-dicts GEM path as it stood before the columnar store, in its order."""
    by_key: dict = {}
    for record in records:
        key = record.get("episode_key") or record.get("hosp_id")
        by_key.setdefault(key, []).append(record)
    samples = []
    for key in sorted(by_key):
        windows = sorted(by_key[key], key=lambda r: int(r["continuation_index"]))
        stream = {field: [] for field in _FIELDS}
        for window in windows:
            for field in _FIELDS:
                stream[field].extend(window[field])
        anchor = windows[0].get("anchor_idx")
        stream.update(episode_key=key, anchor_idx=None if anchor is None else int(anchor),
                      anchor_min=windows[0].get("anchor_min"), outcomes=[],
                      windows=[(int(w["source_start"]), int(w["source_end"]))
                               for w in windows])
        built = builder.build(stream, epoch=epoch)
        for record in windows:
            start, end = int(record["source_start"]), int(record["source_end"])
            length = end - start
            anchor = built["anchor_idx"]
            contains = anchor is not None and start <= anchor < end
            sample = {
                "packed_schema_version": PACKED_SCHEMA_VERSION,
                "input_ids": list(record["token"]),
                "attention_mask": [1] * length,
                "pos_min": list(record["pos_min"]),
                "soft_token": record.get("soft_token"),
                "soft_weight": record.get("soft_weight"),
                "segments": [{
                    "episode_key": key, "source_start": start, "source_end": end,
                    "packed_start": 0, "packed_end": length,
                    "continuation_index": int(record["continuation_index"]),
                    "continues_from_previous": bool(record["continues_from_previous"]),
                    "continues_to_next": bool(record["continues_to_next"]),
                    "anchor_offset": anchor - start if contains else None,
                    "outcome_labels": [], "threshold_query": None,
                }],
            }
            for field in ("ntp_target", "ntp_mask", "ntp_delta_min", "value_target",
                          "value_mask"):
                sample[field] = built[field][start:end]
            if "anchors" in built:
                sample["segments"][0]["anchors"] = [
                    {"offset": a["anchor_idx"] - start, "cr": a["cr"], "queries": a["queries"]}
                    for a in built["anchors"] if start <= a["anchor_idx"] < end]
            if value_channel:
                stats = builder.value_stats
                values = record.get("value") or [None] * len(record["token"])
                channel = [normalize_value(v, t, stats, max_abs_z=builder.max_abs_value_z)
                           for t, v in zip(record["token"], values)]
                sample["input_value"] = [v for v, _ in channel]
                sample["input_value_mask"] = [m for _, m in channel]
            samples.append(sample)
    return samples


def canonical(obj):
    """A structure where equality means: same container shapes and keys, same exact
    element types (int is not numpy int, bool is not int) and float bit patterns."""
    if isinstance(obj, dict):
        return ("dict", tuple((k, canonical(v)) for k, v in obj.items()))
    if isinstance(obj, (list, tuple)):
        return (type(obj).__name__, tuple(canonical(v) for v in obj))
    if isinstance(obj, float):
        return ("float", obj.hex() if obj == obj else "nan")
    if obj is None or isinstance(obj, (bool, int, str)):
        return (type(obj).__name__, obj)
    raise AssertionError(f"unexpected element type {type(obj).__name__}")


def batch_bytes(batch):
    return {k: (str(v.dtype), tuple(v.shape), v.numpy().tobytes())
            for k, v in batch.items() if isinstance(v, torch.Tensor)}


def _window_records(path: Path, site: str, partition: str) -> list[dict]:
    """The rows `pretrain` handed the dataset: one partition, keyed ``<site>:<hosp_id>``."""
    rows = []
    for row in pl.read_parquet(path).filter(pl.col("partition") == partition).iter_rows(
            named=True):
        row["episode_key"] = f"{site}:{row.pop('hosp_id')}"
        rows.append(row)
    return rows


class LeanDatasetEquivalenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._td = tempfile.TemporaryDirectory()
        cls.work = Path(cls._td.name)
        cls.corpus = build_corpus(cls.work)
        cls.loaders = gem_loaders(cls.corpus, td=cls.work / "ckpt")
        cls.builder = cls.loaders.train_dataset.target_builder
        cls.binding = dict(artifact_binding(cls.corpus["blob"]))
        cls.records = {p: [r for s in SITES
                           for r in _window_records(cls.corpus["gem"][s], s, p)]
                       for p in ("train", "validation")}

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    def assert_same_as_reference(self, dataset, records, *, value_channel):
        for epoch in (0, 1):
            dataset.set_epoch(epoch)
            expected = _reference_samples(records, self.builder, epoch=epoch,
                                          value_channel=value_channel)
            got = [dataset[i] for i in range(len(dataset))]
            self.assertEqual(len(got), len(expected))
            for index, (a, b) in enumerate(zip(got, expected)):
                self.assertEqual(canonical(a), canonical(b), f"sample {index}, epoch {epoch}")
            for start in range(0, len(got), 4):
                self.assertEqual(batch_bytes(collate_model_samples(got[start:start + 4])),
                                 batch_bytes(collate_model_samples(expected[start:start + 4])))

    @staticmethod
    def _without_header(dataset):
        """The loader's corpus read as the oracle does: no re-inserted stay header (the
        loaders train with the committed `continuation_header`, which the oracle lacks)."""
        return ModelDataset(dataset.corpus, representation="gem",
                            target_builder=dataset.target_builder,
                            expected_hashes=dataset.expected_hashes)

    def test_loader_samples_equal_the_list_of_dicts_implementation(self):
        self.assert_same_as_reference(self._without_header(self.loaders.train_dataset),
                                      self.records["train"], value_channel=False)
        self.assert_same_as_reference(self._without_header(self.loaders.validation_dataset),
                                      self.records["validation"], value_channel=False)

    def test_value_channel_samples_equal_the_list_of_dicts_implementation(self):
        dataset = ModelDataset(self.records["train"], representation="gem",
                               target_builder=self.builder, expected_hashes=self.binding,
                               value_channel=True)
        self.assert_same_as_reference(dataset, self.records["train"], value_channel=True)

    def test_parquet_corpus_and_its_memory_mapped_cache_equal_the_records(self):
        """Read straight from the shards (no Python rows), first building the cache, then
        memory-mapping it: the same samples as the row path, in the same order."""
        work = self.work / "cache_copy"
        shutil.rmtree(work, ignore_errors=True)
        paths = {}
        for site in SITES:
            (work / site).mkdir(parents=True)
            paths[site] = shutil.copy(self.corpus["gem"][site], work / site)
        for attempt in ("build", "mmap"):
            corpus = GemCorpus.concat([
                GemCorpus.from_parquet(paths[s], site=s, partition="train", cache=True)
                for s in SITES])
            with self.subTest(attempt=attempt):
                if attempt == "mmap":
                    self.assertIsInstance(corpus.parts[0].token, np.memmap)
                dataset = ModelDataset(corpus, representation="gem",
                                       target_builder=self.builder,
                                       expected_hashes=self.binding, value_channel=True)
                self.assert_same_as_reference(dataset, self.records["train"],
                                              value_channel=True)
        caches = sorted((work / "site_a" / "gem_cache").iterdir())
        self.assertEqual(len([c for c in caches if c.is_dir()]), 1)

    def test_a_spawned_worker_reopens_the_cache_instead_of_copying_it(self):
        """DataLoader workers started by spawn/forkserver receive the dataset pickled: a
        memory-mapped part pickles as its cache path, not as its arrays."""
        work = self.work / "pickled"
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir()
        path = shutil.copy(self.corpus["gem"]["site_a"], work)
        corpus = GemCorpus.from_parquet(path, site="site_a", partition="train", cache=True)
        dataset = ModelDataset(corpus, representation="gem", target_builder=self.builder,
                               expected_hashes=self.binding)
        payload = pickle.dumps(corpus)
        self.assertLess(len(payload), corpus.nbytes / 10)
        clone = pickle.loads(payload)
        self.assertIsInstance(clone.parts[0].token, np.memmap)
        twin = ModelDataset(clone, representation="gem", target_builder=self.builder,
                            expected_hashes=self.binding)
        for index in (0, len(dataset) - 1):
            self.assertEqual(canonical(twin[index]), canonical(dataset[index]))

    def test_a_cache_whose_shard_or_vocabulary_changed_is_not_used(self):
        work = self.work / "stale"
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir()
        path = Path(shutil.copy(self.corpus["gem"]["site_a"], work))
        GemCorpus.from_parquet(path, site="a", partition="train", cache=True)
        (cache,) = [c for c in (work / "gem_cache").iterdir() if c.is_dir()]
        # Same directory name, different recorded shard: refused, never read.
        meta = cache / "meta.json"
        meta.write_text(meta.read_text().replace('"shard_sha256": "', '"shard_sha256": "0'))
        with self.assertRaisesRegex(TargetContractError, "stale"):
            GemCorpus.from_parquet(path, site="a", partition="train", cache=True)
        # A rewritten shard hashes differently and gets its own cache.
        frame = pl.read_parquet(path)
        frame.head(frame.height - 1).write_parquet(path)
        GemCorpus.from_parquet(path, site="a", partition="train", cache=True)
        self.assertEqual(len([c for c in (work / "gem_cache").iterdir() if c.is_dir()]), 2)

    def test_records_and_streams_keep_their_shape(self):
        dataset = self.loaders.train_dataset
        reference = sorted(self.records["train"],
                           key=lambda r: (r["episode_key"], r["continuation_index"]))
        self.assertEqual(len(dataset.records), len(reference))
        for view, row in zip(dataset.records, reference):
            for field in ("episode_key", "token", "pos_min", "target_eligible",
                          "source_start", "source_end", "continuation_index"):
                self.assertEqual(canonical(view[field]), canonical(row[field]), field)
            self.assertEqual(
                [None if v is None else ("nan" if v != v else v) for v in view["value"]],
                [None if v is None else ("nan" if v != v else v) for v in row["value"]])
        key = reference[0]["episode_key"]
        stream = dataset._gem_streams[key]
        self.assertEqual(stream["token"],
                         [t for r in reference if r["episode_key"] == key for t in r["token"]])

    def test_columnar_store_costs_far_less_than_the_rows(self):
        dataset = self.loaders.train_dataset
        events = sum(len(r["token"]) for r in self.records["train"])
        self.assertLess(dataset.corpus.nbytes / events, 80)


if __name__ == "__main__":
    unittest.main()
