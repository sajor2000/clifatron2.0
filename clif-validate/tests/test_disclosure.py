"""One implementation, two environments (U9): wheel/repo byte-equivalence.

The vendored evaluator and the in-repo evaluator must produce byte-identical
aggregate payloads (pre-signature) for the same fixture bundle. If they ever
diverge, either the vendor sync is stale (the drift guard catches the textual
case) or something environment-dependent leaked into the payload — a timestamp,
an unordered dict, a host path — which is precisely the class of defect that
turns "same implementation" into a claim instead of a property.

Also pins the policy-in-bundle rule end to end: after the wheel loads a bundle,
every `min_cell_size()` in the process — vendored and repo alike — reads the
bundle's own floor through CLIF_ARTIFACT_POLICY_FILE, not a repo checkout.
"""

import os
import tempfile
import unittest
from pathlib import Path

from clif_validate._vendor.eval import attestation as v_attest
from clif_validate._vendor.eval import schema as v_schema
from clif_validate._vendor.eval.synthetic_bundle import (
    SYNTHETIC_OUTCOME,
    build_synthetic_bundle,
    build_synthetic_site,
)


def _run_pipeline(modules: dict, bundle_dir: Path, site: Path, episodes: Path,
                  shard_tag: str) -> dict:
    """Run bundle → inference → evaluate → build_export with either module set."""
    bundle = modules["bundle"].load_bundle(bundle_dir)
    model = modules["clif_validate"].load_checkpoint(str(bundle_dir))
    cfgs = [{"name": SYNTHETIC_OUTCOME}]
    predict_fn = modules["bundle_inference"].bundle_predict_fn(
        bundle, model, data_path=site, episode_artifact=episodes,
        outcome_cfgs=cfgs, shard_dir=Path("output/intermediate_phi") / shard_tag,
        site_id="SYNTH-A",
    )
    result = modules["clif_validate"].evaluate_site(
        str(bundle_dir), str(site), str(episodes), cfgs, predict_fn=predict_fn,
        cohort_config=bundle.cohort_path, data_config=bundle.data_cfg_path,
    )
    return modules["clif_validate"].build_export(
        result["outcomes"], bundle.provenance, site_id="SYNTH-A",
        site_role="development", partition_role="test", release_id="rel-equivalence",
    )


class WheelRepoEquivalenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.work = Path(cls._tmp.name)
        cls._old_cwd = os.getcwd()
        os.chdir(cls.work)
        cls.site = cls.work / "site"
        cls.episodes = build_synthetic_site(cls.site)
        cls.bundle_dir = build_synthetic_bundle(cls.work / "bundle", cls.site,
                                                cls.episodes)

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls._old_cwd)
        cls._reset_pins()
        cls._tmp.cleanup()

    @staticmethod
    def _reset_pins():
        os.environ.pop(v_schema.POLICY_OVERRIDE_ENV, None)
        v_schema.min_cell_size.cache_clear()
        try:
            from src.eval import schema as r_schema
        except ImportError:
            return
        r_schema.min_cell_size.cache_clear()

    def tearDown(self):
        self._reset_pins()

    @unittest.skipUnless((Path(__file__).resolve().parents[2] / "src" / "eval").is_dir(),
                         "no repository checkout beside the package (installed site)")
    def test_wheel_and_repo_payloads_are_byte_identical_pre_signature(self):
        from clif_validate._vendor.eval import bundle as v_bundle
        from clif_validate._vendor.eval import bundle_inference as v_inference
        from clif_validate._vendor.eval import clif_validate as v_cv
        from src.eval import attestation as r_attest
        from src.eval import bundle as r_bundle
        from src.eval import bundle_inference as r_inference
        from src.eval import clif_validate as r_cv

        vendored = _run_pipeline(
            {"bundle": v_bundle, "bundle_inference": v_inference, "clif_validate": v_cv},
            self.bundle_dir, self.site, self.episodes, "shards_wheel",
        )
        in_repo = _run_pipeline(
            {"bundle": r_bundle, "bundle_inference": r_inference, "clif_validate": r_cv},
            self.bundle_dir, self.site, self.episodes, "shards_repo",
        )
        self.assertEqual(v_attest.canonical_bytes(vendored),
                         r_attest.canonical_bytes(in_repo))
        # The serializers themselves must agree, not just the dicts.
        self.assertEqual(v_attest.canonical_bytes(in_repo),
                         r_attest.canonical_bytes(in_repo))

    def test_loading_a_bundle_pins_its_policy_for_the_whole_process(self):
        from clif_validate._vendor.eval import bundle as v_bundle

        v_bundle.load_bundle(self.bundle_dir)
        pinned = os.environ.get(v_schema.POLICY_OVERRIDE_ENV)
        self.assertEqual(Path(pinned), (self.bundle_dir / "artifact_policy.yaml").resolve())
        self.assertEqual(v_schema.min_cell_size(), 10)


if __name__ == "__main__":
    unittest.main()
