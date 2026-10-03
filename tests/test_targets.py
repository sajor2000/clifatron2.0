import hashlib
import json
import unittest

from src.data.targets import TargetBuilder, TargetContractError


def episode(**overrides):
    value = {
        "episode_key": "opaque-a",
        "token": [3, 9, 4],
        "pos_min": [10, 20, 40],
        "value": [70.0, None, 90.0],
        "target_eligible": [True, False, True],
        "anchor_idx": 2,
        "anchor_min": 40,
        "outcomes": [
            {
                "target_idx": 1,
                "status": "positive",
                "time_from_anchor_hours": 2.5,
                "threshold_bin": 6,
                "direction": "below",
            },
            {
                "target_idx": 2,
                "status": "censored",
                "time_from_anchor_hours": 5.0,
                "threshold_bin": 4,
                "direction": "above",
            },
        ],
    }
    value.update(overrides)
    return value


class TargetBuilderTest(unittest.TestCase):
    def setUp(self):
        self.builder = TargetBuilder(
            vocab_size=16,
            n_time_bins=48,
            horizon_hours=48,
            value_stats={4: (80.0, 5.0)},
            run_seed=7,
        )

    def test_treatments_are_context_but_never_targets(self):
        result = self.builder.build(episode())

        self.assertEqual(result["ntp_target"], [4, 0, 0])
        self.assertEqual(result["ntp_mask"], [True, False, False])
        self.assertEqual(result["ntp_delta_min"], [30, 0, 0])
        self.assertEqual(result["value_target"], [2.0, 0.0, 0.0])
        self.assertEqual(result["value_mask"], [True, False, False])

    def test_extreme_normalized_mark_is_excluded_but_event_remains_target(self):
        result = self.builder.build(episode(value=[70.0, None, 999999.0]))

        self.assertEqual(result["ntp_target"], [4, 0, 0])
        self.assertEqual(result["ntp_mask"], [True, False, False])
        self.assertEqual(result["value_target"], [0.0, 0.0, 0.0])
        self.assertEqual(result["value_mask"], [False, False, False])

    def test_censoring_records_observed_risk_without_inventing_a_cause(self):
        labels = self.builder.build(episode())["outcome_labels"]

        self.assertEqual(labels[0]["event_cause"], 1)
        self.assertEqual(labels[0]["event_bin"], 2)
        self.assertEqual(labels[1]["event_cause"], -1)
        self.assertEqual(labels[1]["event_bin"], -1)
        self.assertEqual(labels[1]["observed_bins"], 5)
        self.assertTrue(labels[1]["censored"])

    def test_threshold_query_is_deterministic_by_sample_epoch_and_seed(self):
        first = self.builder.build(episode(), epoch=3)["threshold_query"]
        second = self.builder.build(episode(), epoch=3)["threshold_query"]

        self.assertEqual(first, second)

    def test_unsupported_and_prevalent_outcomes_are_masked(self):
        outcomes = [
            {
                "target_idx": 1,
                "status": status,
                "time_from_anchor_hours": None,
                "threshold_bin": 2,
                "direction": "below",
            }
            for status in ("unsupported_at_site", "not_ascertainable", "prevalent")
        ]
        labels = self.builder.build(episode(outcomes=outcomes))["outcome_labels"]

        self.assertTrue(all(not label["tte_mask"] for label in labels))
        self.assertIsNone(self.builder.build(episode(outcomes=outcomes))["threshold_query"])

    def test_rejects_post_anchor_features_and_invalid_tokens(self):
        with self.assertRaisesRegex(TargetContractError, "post-anchor"):
            self.builder.build(episode(anchor_min=30))
        with self.assertRaisesRegex(TargetContractError, "vocabulary"):
            self.builder.build(episode(token=[3, 99, 4]))


def _characterization_episode(key, **overrides):
    value = {
        "episode_key": key,
        "token": [3, 9, 4, 5, 4],
        "pos_min": [10, 20, 40, 40, 55],
        "value": [70.0, None, 90.0, float("nan"), 61.5],
        "target_eligible": [True, False, True, True, True],
        "anchor_idx": 4,
        "anchor_min": 55,
        "outcomes": [],
    }
    value.update(overrides)
    return value


def _characterization_outcome(status, hours, *, target=1, tbin=6, direction="below",
                              cause=None):
    row = {"target_idx": target, "status": status, "time_from_anchor_hours": hours,
           "threshold_bin": tbin, "direction": direction}
    if cause is not None:
        row["cause_idx"] = cause
    return row


class Icu24hCharacterizationTest(unittest.TestCase):
    """The 24 h mode is the regression path (KTD1): `TargetBuilder.build` output for it
    is pinned byte for byte. The hashes were captured on the commit BEFORE the in-stream
    (`gem_tte`) mode was added; a change here is a change to the 24 h label contract."""

    # sha256 of the canonical JSON of every build below, per builder time grid.
    PINNED = {
        16: "205e49620054d14bde9fe445c85cdc1184c405f59fd17ae01197dd6f61982ee7",
        48: "5b2e98f48fbef88455aeedc392fb86e74980da938ff16683d331c811e776caa6",
    }

    @staticmethod
    def _episodes():
        ep, out = _characterization_episode, _characterization_outcome
        return [
            ep("opaque-a", outcomes=[out("positive", 2.5),
                                     out("censored", 5.0, target=2, tbin=4,
                                         direction="above")]),
            ep("opaque-b", outcomes=[out("negative", 48.0),
                                     out("competing_event", 12.0, target=2, cause=3),
                                     out("positive", 3.0, target=0, tbin=1,
                                         direction="above")]),
            ep("opaque-c", outcomes=[out("prevalent", None),
                                     out("not_ascertainable", None, target=2),
                                     out("unsupported_at_site", None, target=0)]),
            ep("opaque-d", outcomes=[out("censored", 0.0), out("positive", 47.99, target=2),
                                     out("positive", 60.0, target=0)]),
            ep("opaque-e"),
        ]

    def test_build_output_is_byte_identical_to_the_pinned_baseline(self):
        for n_bins, expected in self.PINNED.items():
            builder = TargetBuilder(vocab_size=16, n_time_bins=n_bins, horizon_hours=48,
                                    value_stats={4: (80.0, 5.0), 5: (1.0, 2.0)}, run_seed=7)
            built = [builder.build(episode, epoch=epoch)
                     for episode in self._episodes() for epoch in (0, 1, 5)]
            payload = json.dumps(built, sort_keys=True, separators=(",", ":"))
            with self.subTest(n_time_bins=n_bins):
                self.assertEqual(hashlib.sha256(payload.encode()).hexdigest(), expected)


if __name__ == "__main__":
    unittest.main()
