import sys
import os
import unittest
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import StudentProfile, Subject, Topic
from src.ml_model import SuccessPredictor
from src.planner import (
    BeamSearchPlanner, compute_utility, days_until_exam, exam_urgency, revision_need,
)

TEST_DATASET = os.path.join(os.path.dirname(__file__), "_tmp_dataset.csv")


class TestPlanner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = SuccessPredictor(TEST_DATASET)

    @classmethod
    def tearDownClass(cls):
        if os.path.isfile(TEST_DATASET):
            os.remove(TEST_DATASET)

    def make_profile(self):
        profile = StudentProfile(name="Test", daily_minutes=60)
        subject = Subject(subject_id="s1", name="Physics", exam_date="2026-10-01")
        weak_topic = Topic(topic_id="t_weak", name="Weak Topic", subject_id="s1",
                            difficulty=0.6, importance=0.9, alpha=2, beta=8)
        strong_topic = Topic(topic_id="t_strong", name="Strong Topic", subject_id="s1",
                              difficulty=0.4, importance=0.5, alpha=9, beta=2)
        subject.topics = {"t_weak": weak_topic, "t_strong": strong_topic}
        profile.subjects = {"s1": subject}
        return profile

    def test_days_until_exam_future(self):
        days = days_until_exam("2026-10-10", date(2026, 9, 18))
        self.assertEqual(days, 22)

    def test_days_until_exam_none(self):
        self.assertEqual(days_until_exam(None, date(2026, 9, 18)), 60)

    def test_exam_urgency_increases_as_date_nears(self):
        far = exam_urgency(40)
        near = exam_urgency(2)
        self.assertLess(far, near)

    def test_revision_need_bounds(self):
        self.assertEqual(revision_need(0), 0.0)
        self.assertEqual(revision_need(100), 1.0)

    def test_weak_topic_gets_higher_utility_than_strong(self):
        profile = self.make_profile()
        subject = profile.subjects["s1"]
        today = date(2026, 9, 18)
        u_weak, _ = compute_utility(subject, subject.topics["t_weak"], self.predictor, today, 0)
        u_strong, _ = compute_utility(subject, subject.topics["t_strong"], self.predictor, today, 0)
        self.assertGreater(u_weak, u_strong)

    def test_repetition_penalty_reduces_utility(self):
        profile = self.make_profile()
        subject = profile.subjects["s1"]
        today = date(2026, 9, 18)
        u0, _ = compute_utility(subject, subject.topics["t_weak"], self.predictor, today, 0)
        u2, _ = compute_utility(subject, subject.topics["t_weak"], self.predictor, today, 2)
        self.assertLess(u2, u0)

    def test_planner_generates_actions(self):
        profile = self.make_profile()
        planner = BeamSearchPlanner(self.predictor)
        actions, trace = planner.generate_plan(profile, date(2026, 9, 18))
        self.assertGreater(len(actions), 0)
        self.assertGreater(len(trace), 0)

    def test_planner_respects_time_budget(self):
        profile = self.make_profile()
        profile.daily_minutes = 25
        planner = BeamSearchPlanner(self.predictor, slot_minutes=20)
        actions, _ = planner.generate_plan(profile, date(2026, 9, 18))
        total_minutes = sum(a.minutes for a in actions)
        self.assertLessEqual(total_minutes, 25)

    def test_planner_prioritizes_weaker_topic_first(self):
        profile = self.make_profile()
        planner = BeamSearchPlanner(self.predictor)
        actions, _ = planner.generate_plan(profile, date(2026, 9, 18))
        self.assertEqual(actions[0].topic_id, "t_weak")

    def test_empty_profile_returns_no_plan(self):
        profile = StudentProfile(name="Empty", daily_minutes=60)
        planner = BeamSearchPlanner(self.predictor)
        actions, trace = planner.generate_plan(profile, date(2026, 9, 18))
        self.assertEqual(actions, [])
        self.assertEqual(trace, [])


if __name__ == "__main__":
    unittest.main()
