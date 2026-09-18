from datetime import date
from src.mastery import record_assessment, apply_forgetting, mastery_summary
from src.planner import BeamSearchPlanner


class StudyPlannerAgent:
    """
    Intelligent agent environment:
      Percepts  -> student profile state (mastery, scores, dates, time budget)
      Actions   -> study / revise / practice recommendations
      Goal      -> maximize expected exam readiness within available time
      Rationality -> performance measure below guides every planning decision
    """

    def __init__(self, profile, predictor, beam_width=4):
        self.profile = profile
        self.predictor = predictor
        self.planner = BeamSearchPlanner(predictor, beam_width=beam_width)
        self.last_plan = None
        self.last_trace = None
        self.previous_plan_snapshot = None

    def apply_forgetting_curve(self, today: date = None):
        if today is None:
            today = date.today()
        for _, topic in self.profile.all_topics():
            apply_forgetting(topic, today)

    def generate_plan(self, today: date = None):
        if today is None:
            today = date.today()
        self.apply_forgetting_curve(today)
        actions, trace = self.planner.generate_plan(self.profile, today)
        self.previous_plan_snapshot = self.last_plan
        self.last_plan = actions
        self.last_trace = trace
        return actions, trace

    def record_assessment_result(self, subject_id, topic_id, score, kind="practice", today: date = None):
        if today is None:
            today = date.today()
        subject = self.profile.subjects[subject_id]
        topic = subject.topics[topic_id]
        record_assessment(topic, score, kind, today)
        return mastery_summary(topic)

    def explain_latest(self, today: date = None):
        if not self.last_plan:
            return None
        rejected = self.planner.explain_rejected(self.profile, self.last_plan, today)
        return {
            "chosen": [
                {
                    "topic": a.topic_name,
                    "subject": a.subject_name,
                    "action": a.action_type,
                    "minutes": a.minutes,
                    "utility": round(a.utility, 3),
                    "reasons": a.reasons,
                }
                for a in self.last_plan
            ],
            "rejected_top": [
                {"topic": t, "subject": s, "utility": round(u, 3), "reasons": r}
                for t, s, u, r in rejected
            ],
        }

    def performance_measure(self):
        topics = self.profile.all_topics()
        if not topics:
            return 0.0
        return sum(t.mastery * t.importance for _, t in topics) / sum(t.importance for _, t in topics)

    def plan_diff(self):
        if not self.previous_plan_snapshot:
            return None
        before = {a.topic_id: a.action_type for a in self.previous_plan_snapshot}
        after = {a.topic_id: a.action_type for a in self.last_plan}
        added = [tid for tid in after if tid not in before]
        removed = [tid for tid in before if tid not in after]
        return {"added_topics": added, "removed_topics": removed}
