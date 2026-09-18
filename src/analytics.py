from datetime import date
from src.planner import days_until_exam


def progress_report(profile):
    rows = []
    for subject in profile.subjects.values():
        for topic in subject.topics.values():
            rows.append({
                "subject": subject.name,
                "topic": topic.name,
                "mastery": round(topic.mastery, 3),
                "uncertainty": round(topic.uncertainty, 3),
                "attempts": topic.attempts,
                "last_studied": topic.last_studied or "never",
            })
    return rows


def weak_topics(profile, threshold=0.5, top_n=5):
    all_topics = []
    for subject in profile.subjects.values():
        for topic in subject.topics.values():
            if topic.mastery < threshold:
                all_topics.append((subject.name, topic.name, round(topic.mastery, 3)))
    all_topics.sort(key=lambda x: x[2])
    return all_topics[:top_n]


def upcoming_exams(profile, today: date = None):
    if today is None:
        today = date.today()
    exams = []
    for subject in profile.subjects.values():
        if subject.exam_date:
            days_left = days_until_exam(subject.exam_date, today)
            exams.append((subject.name, subject.exam_date, days_left))
    exams.sort(key=lambda x: x[2])
    return exams


def mastery_changes(profile):
    changes = []
    for subject in profile.subjects.values():
        for topic in subject.topics.values():
            if len(topic.history) >= 2:
                first = topic.history[0]["score"]
                last = topic.history[-1]["score"]
                changes.append((subject.name, topic.name, round(last - first, 3)))
    changes.sort(key=lambda x: x[2], reverse=True)
    return changes
