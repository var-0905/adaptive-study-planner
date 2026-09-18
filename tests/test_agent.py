import sys
import os
import unittest
import tempfile
import shutil
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import StudentProfile, Subject, Topic
from src.storage import Storage
from src.ml_model import SuccessPredictor
from src.agent import StudyPlannerAgent

TEST_DATASET = os.path.join(os.path.dirname(__file__), "_tmp_dataset2.csv")


class TestAgent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = SuccessPredictor(TEST_DATASET)

    @classmethod
    def tearDownClass(cls):
        if os.path.isfile(TEST_DATASET):
            os.remove(TEST_DATASET)

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.profile = StudentProfile(name="Agent Test", daily_minutes=60)
        subject = Subject(subject_id="s1", name="Chemistry", exam_date="2026-10-15")
        topic = Topic(topic_id="t1", name="Stoichiometry", subject_id="s1",
                       difficulty=0.5, importance=0.7)
        subject.topics = {"t1": topic}
        self.profile.subjects = {"s1": subject}
        self.agent = StudyPlannerAgent(self.profile, self.predictor)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_generate_plan_creates_actions(self):
        actions, trace = self.agent.generate_plan(date(2026, 9, 18))
        self.assertGreater(len(actions), 0)

    def test_record_assessment_updates_mastery(self):
        before = self.profile.subjects["s1"].topics["t1"].mastery
        self.agent.record_assessment_result("s1", "t1", 0.9, "quiz", date(2026, 9, 18))
        after = self.profile.subjects["s1"].topics["t1"].mastery
        self.assertGreater(after, before)

    def test_replan_after_assessment_changes_plan(self):
        self.agent.generate_plan(date(2026, 9, 18))
        for _ in range(5):
            self.agent.record_assessment_result("s1", "t1", 0.95, "quiz", date(2026, 9, 18))
        actions_before = self.agent.last_plan
        self.agent.generate_plan(date(2026, 9, 18))
        self.assertIsNotNone(self.agent.last_plan)

    def test_explain_latest_returns_none_without_plan(self):
        fresh_agent = StudyPlannerAgent(self.profile, self.predictor)
        fresh_agent.last_plan = None
        self.assertIsNone(fresh_agent.explain_latest())

    def test_explain_latest_after_plan(self):
        self.agent.generate_plan(date(2026, 9, 18))
        explanation = self.agent.explain_latest(date(2026, 9, 18))
        self.assertIn("chosen", explanation)
        self.assertGreater(len(explanation["chosen"]), 0)

    def test_performance_measure_within_bounds(self):
        score = self.agent.performance_measure()
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_invalid_subject_raises_keyerror(self):
        with self.assertRaises(KeyError):
            self.agent.record_assessment_result("bad_subject", "t1", 0.5)

    def test_storage_round_trip(self):
        path = os.path.join(self.tmpdir, "profile.json")
        storage = Storage(path)
        storage.save(self.profile)
        self.assertTrue(storage.exists())
        loaded = storage.load()
        self.assertEqual(loaded.name, "Agent Test")
        self.assertEqual(loaded.subjects["s1"].topics["t1"].name, "Stoichiometry")

    def test_storage_missing_file_returns_empty_profile(self):
        path = os.path.join(self.tmpdir, "does_not_exist.json")
        storage = Storage(path)
        profile = storage.load()
        self.assertEqual(profile.subjects, {})


if __name__ == "__main__":
    unittest.main()
