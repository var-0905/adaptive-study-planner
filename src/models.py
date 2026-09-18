from dataclasses import dataclass, field, asdict
from datetime import date
from typing import Optional


@dataclass
class AssessmentRecord:
    date: str
    score: float
    kind: str = "practice"


@dataclass
class Topic:
    topic_id: str
    name: str
    subject_id: str
    difficulty: float
    importance: float
    alpha: float = 2.0
    beta: float = 2.0
    attempts: int = 0
    last_studied: Optional[str] = None
    history: list = field(default_factory=list)

    @property
    def mastery(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def uncertainty(self) -> float:
        a, b = self.alpha, self.beta
        variance = (a * b) / (((a + b) ** 2) * (a + b + 1))
        return variance ** 0.5

    @property
    def recent_avg_score(self) -> float:
        if not self.history:
            return 0.0
        recent = self.history[-3:]
        return sum(r["score"] for r in recent) / len(recent)

    @property
    def latest_score(self) -> float:
        if not self.history:
            return 0.0
        return self.history[-1]["score"]

    def days_since_last_study(self, today: date) -> int:
        if not self.last_studied:
            return 999
        last = date.fromisoformat(self.last_studied)
        return max((today - last).days, 0)

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        d = dict(d)
        history = d.pop("history", [])
        t = Topic(**{k: v for k, v in d.items() if k in Topic.__dataclass_fields__})
        t.history = history
        return t


@dataclass
class Subject:
    subject_id: str
    name: str
    exam_date: Optional[str] = None
    topics: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "subject_id": self.subject_id,
            "name": self.name,
            "exam_date": self.exam_date,
            "topics": {tid: t.to_dict() for tid, t in self.topics.items()},
        }

    @staticmethod
    def from_dict(d):
        s = Subject(subject_id=d["subject_id"], name=d["name"], exam_date=d.get("exam_date"))
        for tid, td in d.get("topics", {}).items():
            s.topics[tid] = Topic.from_dict(td)
        return s


@dataclass
class StudentProfile:
    name: str = "Student"
    daily_minutes: int = 60
    subjects: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "name": self.name,
            "daily_minutes": self.daily_minutes,
            "subjects": {sid: s.to_dict() for sid, s in self.subjects.items()},
        }

    @staticmethod
    def from_dict(d):
        p = StudentProfile(name=d.get("name", "Student"), daily_minutes=d.get("daily_minutes", 60))
        for sid, sd in d.get("subjects", {}).items():
            p.subjects[sid] = Subject.from_dict(sd)
        return p

    def all_topics(self):
        topics = []
        for subject in self.subjects.values():
            for topic in subject.topics.values():
                topics.append((subject, topic))
        return topics
