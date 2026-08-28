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


if __name__ == "__main__":
    unittest.main()
