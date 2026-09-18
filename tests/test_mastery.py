import sys
import os
import unittest
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import Topic
from src.mastery import record_assessment, apply_forgetting, PASS_THRESHOLD


class TestMastery(unittest.TestCase):
    def setUp(self):
        self.topic = Topic(
            topic_id="t1", name="Test Topic", subject_id="s1",
            difficulty=0.5, importance=0.5,
        )

    def test_initial_mastery_is_midpoint(self):
        self.assertAlmostEqual(self.topic.mastery, 0.5)

    def test_passing_score_increases_mastery(self):
        before = self.topic.mastery
        record_assessment(self.topic, 0.9, "quiz", date(2026, 1, 1))
        self.assertGreater(self.topic.mastery, before)

    def test_failing_score_decreases_mastery(self):
        before = self.topic.mastery
        record_assessment(self.topic, 0.2, "quiz", date(2026, 1, 1))
        self.assertLess(self.topic.mastery, before)

    def test_mastery_stays_within_bounds(self):
        for _ in range(20):
            record_assessment(self.topic, 1.0, "practice", date(2026, 1, 1))
        self.assertLess(self.topic.mastery, 1.0)
        self.assertGreater(self.topic.mastery, 0.0)

    def test_attempts_increment(self):
        record_assessment(self.topic, 0.7, "practice", date(2026, 1, 1))
        record_assessment(self.topic, 0.8, "practice", date(2026, 1, 2))
        self.assertEqual(self.topic.attempts, 2)

    def test_last_studied_updates(self):
        record_assessment(self.topic, 0.7, "practice", date(2026, 1, 5))
        self.assertEqual(self.topic.last_studied, "2026-01-05")

    def test_history_is_recorded(self):
        record_assessment(self.topic, 0.6, "quiz", date(2026, 1, 1))
        self.assertEqual(len(self.topic.history), 1)
        self.assertEqual(self.topic.history[0]["score"], 0.6)

    def test_score_clamped_to_valid_range(self):
        record_assessment(self.topic, 1.5, "quiz", date(2026, 1, 1))
        self.assertEqual(self.topic.history[-1]["score"], 1.0)

    def test_uncertainty_decreases_with_more_attempts(self):
        u1 = self.topic.uncertainty
        for i in range(5):
            record_assessment(self.topic, 0.8, "practice", date(2026, 1, 1 + i))
        u2 = self.topic.uncertainty
        self.assertLess(u2, u1)

    def test_forgetting_pulls_mastery_toward_uncertainty(self):
        record_assessment(self.topic, 0.9, "quiz", date(2026, 1, 1))
        alpha_before = self.topic.alpha
        apply_forgetting(self.topic, date(2026, 3, 1), half_life_days=20)
        self.assertLessEqual(self.topic.alpha, alpha_before)

    def test_pass_threshold_constant(self):
        self.assertEqual(PASS_THRESHOLD, 0.6)


if __name__ == "__main__":
    unittest.main()
