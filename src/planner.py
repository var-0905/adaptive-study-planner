import heapq
from datetime import date, timedelta
from dataclasses import dataclass, field
from typing import List


DEFAULT_SLOT_MINUTES = 20


@dataclass
class Action:
    topic_id: str
    topic_name: str
    subject_name: str
    action_type: str
    minutes: int
    utility: float
    reasons: dict


@dataclass
class PlanState:
    remaining_minutes: int
    used_topic_counts: dict
    actions: List[Action] = field(default_factory=list)

    def total_utility(self):
        return sum(a.utility for a in self.actions)

    def key(self):
        return (self.remaining_minutes, tuple(sorted(self.used_topic_counts.items())), len(self.actions))


def days_until_exam(exam_date_str, today: date):
    if not exam_date_str:
        return 60
    try:
        exam_date = date.fromisoformat(exam_date_str)
    except ValueError:
        return 60
    return max((exam_date - today).days, 0)


def exam_urgency(days_left: int) -> float:
    if days_left <= 0:
        return 1.0
    return max(0.0, min(1.0, 1.0 - (days_left / 45)))


def revision_need(days_since_study: int) -> float:
    return max(0.0, min(1.0, days_since_study / 30))


def compute_utility(subject, topic, predictor, today: date, repetition_count: int):
    days_left = days_until_exam(subject.exam_date, today)
    urgency = exam_urgency(days_left)
    days_since = topic.days_since_last_study(today)
    need = revision_need(days_since)

    features = {
        "mastery": topic.mastery,
        "recent_avg_score": topic.recent_avg_score,
        "latest_score": topic.latest_score,
        "difficulty": topic.difficulty,
        "days_since_study": min(days_since, 60),
        "attempts": topic.attempts,
        "exam_urgency": urgency,
    }
    predicted_success = predictor.predict_success_probability(features)

    mastery_gap = 1.0 - topic.mastery
    repetition_penalty = 0.18 * repetition_count

    utility = (
        0.30 * mastery_gap
        + 0.20 * predicted_success
        + 0.20 * urgency
        + 0.15 * topic.importance
        + 0.15 * need
        - repetition_penalty
    )

    reasons = {
        "mastery": round(topic.mastery, 2),
        "mastery_gap": round(mastery_gap, 2),
        "predicted_success": round(predicted_success, 2),
        "exam_urgency": round(urgency, 2),
        "days_to_exam": days_left,
        "importance": round(topic.importance, 2),
        "revision_need": round(need, 2),
        "days_since_study": days_since,
        "repetition_penalty": round(repetition_penalty, 2),
        "utility": round(utility, 3),
    }
    return utility, reasons


def action_type_for(topic, days_since_study):
    if topic.attempts == 0:
        return "study"
    if topic.mastery < 0.5:
        return "study"
    if days_since_study >= 14:
        return "revise"
    return "practice"


class BeamSearchPlanner:
    def __init__(self, predictor, beam_width=4, slot_minutes=DEFAULT_SLOT_MINUTES):
        self.predictor = predictor
        self.beam_width = beam_width
        self.slot_minutes = slot_minutes

    def generate_plan(self, profile, today: date = None, max_actions=8):
        if today is None:
            today = date.today()

        candidates = []
        for subject in profile.subjects.values():
            for topic in subject.topics.values():
                candidates.append((subject, topic))

        if not candidates:
            return [], []

        initial = PlanState(remaining_minutes=profile.daily_minutes, used_topic_counts={})
        frontier = [initial]
        trace = []

        for step in range(max_actions):
            expanded = []
            for state in frontier:
                if state.remaining_minutes < self.slot_minutes:
                    expanded.append(state)
                    continue
                state_had_expansion = False
                for subject, topic in candidates:
                    rep = state.used_topic_counts.get(topic.topic_id, 0)
                    if rep >= 2:
                        continue
                    state_had_expansion = True
                    utility, reasons = compute_utility(subject, topic, self.predictor, today, rep)
                    action_type = action_type_for(topic, topic.days_since_last_study(today))
                    new_action = Action(
                        topic_id=topic.topic_id,
                        topic_name=topic.name,
                        subject_name=subject.name,
                        action_type=action_type,
                        minutes=self.slot_minutes,
                        utility=utility,
                        reasons=reasons,
                    )
                    new_state = PlanState(
                        remaining_minutes=state.remaining_minutes - self.slot_minutes,
                        used_topic_counts=dict(state.used_topic_counts),
                        actions=list(state.actions) + [new_action],
                    )
                    new_state.used_topic_counts[topic.topic_id] = rep + 1
                    expanded.append(new_state)
                if not state_had_expansion:
                    expanded.append(state)

            expanded.sort(key=lambda s: s.total_utility(), reverse=True)
            frontier = expanded[: self.beam_width]
            trace.append({
                "step": step + 1,
                "best_utility": round(frontier[0].total_utility(), 3) if frontier else 0,
                "frontier_size": len(frontier),
            })
            if all(s.remaining_minutes < self.slot_minutes for s in frontier):
                break

        if not frontier:
            return [], trace
        best_state = max(frontier, key=lambda s: s.total_utility())
        return best_state.actions, trace

    def explain_rejected(self, profile, chosen_actions, today: date = None, top_n=3):
        if today is None:
            today = date.today()
        chosen_ids = {a.topic_id for a in chosen_actions}
        scored = []
        for subject in profile.subjects.values():
            for topic in subject.topics.values():
                if topic.topic_id in chosen_ids:
                    continue
                utility, reasons = compute_utility(subject, topic, self.predictor, today, 0)
                scored.append((topic.name, subject.name, utility, reasons))
        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:top_n]
