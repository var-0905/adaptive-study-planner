import csv
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = [
    "mastery",
    "recent_avg_score",
    "latest_score",
    "difficulty",
    "days_since_study",
    "attempts",
    "exam_urgency",
]


def generate_synthetic_dataset(path, n_samples=400, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_samples):
        mastery = rng.uniform(0, 1)
        recent_avg = np.clip(mastery + rng.normal(0, 0.12), 0, 1)
        latest = np.clip(recent_avg + rng.normal(0, 0.1), 0, 1)
        difficulty = rng.uniform(0, 1)
        days_since = rng.integers(0, 40)
        attempts = rng.integers(0, 15)
        urgency = rng.uniform(0, 1)

        score = (
            0.45 * mastery
            + 0.25 * recent_avg
            + 0.15 * latest
            - 0.20 * difficulty
            - 0.01 * days_since
            + 0.015 * attempts
            - 0.05 * urgency
        )
        prob_success = 1 / (1 + np.exp(-6 * (score - 0.15)))
        label = 1 if rng.uniform(0, 1) < prob_success else 0
        rows.append([mastery, recent_avg, latest, difficulty, days_since, attempts, urgency, label])

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(FEATURE_NAMES + ["success"])
        writer.writerows(rows)


def load_dataset(path):
    X, y = [], []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            X.append([float(row[c]) for c in FEATURE_NAMES])
            y.append(int(row["success"]))
    return np.array(X), np.array(y)


class SuccessPredictor:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.model = LogisticRegression(max_iter=1000)
        self.scaler = StandardScaler()
        self.metrics = {}
        self.is_synthetic = True
        self._train()

    def _train(self):
        if not os.path.isfile(self.dataset_path):
            generate_synthetic_dataset(self.dataset_path)
        X, y = load_dataset(self.dataset_path)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        self.scaler.fit(X_train)
        X_train_s = self.scaler.transform(X_train)
        X_test_s = self.scaler.transform(X_test)

        self.model.fit(X_train_s, y_train)
        preds = self.model.predict(X_test_s)

        self.metrics = {
            "accuracy": round(accuracy_score(y_test, preds), 3),
            "precision": round(precision_score(y_test, preds, zero_division=0), 3),
            "recall": round(recall_score(y_test, preds, zero_division=0), 3),
            "f1": round(f1_score(y_test, preds, zero_division=0), 3),
            "train_size": len(y_train),
            "test_size": len(y_test),
        }

    def predict_success_probability(self, features: dict) -> float:
        vec = np.array([[features[name] for name in FEATURE_NAMES]])
        vec_s = self.scaler.transform(vec)
        prob = self.model.predict_proba(vec_s)[0][1]
        return float(prob)

    def feature_weights(self):
        return dict(zip(FEATURE_NAMES, self.model.coef_[0].round(3).tolist()))
