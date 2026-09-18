from datetime import date
from src.models import Topic


PASS_THRESHOLD = 0.6


def record_assessment(topic: Topic, score: float, kind: str = "practice", today: date = None):
    if today is None:
        today = date.today()
    score = max(0.0, min(1.0, score))

    if score >= PASS_THRESHOLD:
        strength = 0.5 + score
        topic.alpha += strength
    else:
        strength = 0.5 + (1 - score)
        topic.beta += strength

    topic.attempts += 1
    topic.last_studied = today.isoformat()
    topic.history.append({"date": today.isoformat(), "score": score, "kind": kind})
    return topic


def apply_forgetting(topic: Topic, today: date = None, half_life_days: int = 20):
    if today is None:
        today = date.today()
    days = topic.days_since_last_study(today)
    if days <= 0 or topic.attempts == 0:
        return topic
    decay_steps = days / half_life_days
    if decay_steps <= 0:
        return topic
    pull_alpha = (topic.alpha - 1) * (1 - 0.5 ** decay_steps) * 0.1
    pull_beta = (topic.beta - 1) * (1 - 0.5 ** decay_steps) * 0.1
    topic.alpha = max(1.0, topic.alpha - pull_alpha)
    topic.beta = max(1.0, topic.beta - pull_beta)
    return topic


def mastery_summary(topic: Topic):
    return {
        "topic": topic.name,
        "mastery": round(topic.mastery, 3),
        "uncertainty": round(topic.uncertainty, 3),
        "alpha": round(topic.alpha, 2),
        "beta": round(topic.beta, 2),
        "attempts": topic.attempts,
    }
