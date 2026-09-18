import json
import os
from src.models import StudentProfile


class Storage:
    def __init__(self, path):
        self.path = path

    def exists(self):
        return os.path.isfile(self.path)

    def load(self) -> StudentProfile:
        if not self.exists():
            return StudentProfile()
        with open(self.path, "r") as f:
            data = json.load(f)
        return StudentProfile.from_dict(data)

    def save(self, profile: StudentProfile):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(profile.to_dict(), f, indent=2)

    def load_demo(self, demo_path) -> StudentProfile:
        with open(demo_path, "r") as f:
            data = json.load(f)
        return StudentProfile.from_dict(data)
