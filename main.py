import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.storage import Storage
from src.models import StudentProfile, Subject, Topic
from src.ml_model import SuccessPredictor
from src.agent import StudyPlannerAgent
from src.analytics import progress_report, weak_topics, upcoming_exams, mastery_changes

DATA_PATH = os.path.join("data", "profile.json")
DEMO_PATH = os.path.join("data", "demo_data.json")
DATASET_PATH = os.path.join("data", "training_data.csv")


def clear_prompt():
    print()


def read_float(prompt, low=0.0, high=1.0, default=None):
    while True:
        raw = input(prompt).strip()
        if raw == "" and default is not None:
            return default
        try:
            val = float(raw)
            if low <= val <= high:
                return val
            print(f"Enter a number between {low} and {high}.")
        except ValueError:
            print("Please enter a valid number.")


def read_int(prompt, low=1, high=1000, default=None):
    while True:
        raw = input(prompt).strip()
        if raw == "" and default is not None:
            return default
        try:
            val = int(raw)
            if low <= val <= high:
                return val
            print(f"Enter a whole number between {low} and {high}.")
        except ValueError:
            print("Please enter a valid whole number.")


def read_date(prompt, allow_blank=True):
    while True:
        raw = input(prompt).strip()
        if raw == "" and allow_blank:
            return None
        try:
            date.fromisoformat(raw)
            return raw
        except ValueError:
            print("Please use YYYY-MM-DD format.")


def read_nonempty(prompt):
    while True:
        raw = input(prompt).strip()
        if raw:
            return raw
        print("This field cannot be empty.")


def choose_subject(profile):
    if not profile.subjects:
        print("No subjects yet. Add one first.")
        return None
    items = list(profile.subjects.values())
    for i, s in enumerate(items, 1):
        print(f"  {i}. {s.name} ({len(s.topics)} topics)")
    idx = read_int("Choose subject number: ", 1, len(items)) - 1
    return items[idx]


def choose_topic(subject):
    if not subject.topics:
        print("No topics in this subject yet.")
        return None
    items = list(subject.topics.values())
    for i, t in enumerate(items, 1):
        print(f"  {i}. {t.name} (mastery {t.mastery:.2f})")
    idx = read_int("Choose topic number: ", 1, len(items)) - 1
    return items[idx]


def setup_profile(profile):
    clear_prompt()
    name = read_nonempty("Student name: ")
    minutes = read_int("Available study minutes per day: ", 10, 600, default=60)
    profile.name = name
    profile.daily_minutes = minutes
    print(f"Profile saved for {name}, {minutes} minutes/day available.")


def add_subject(profile):
    clear_prompt()
    name = read_nonempty("Subject name: ")
    exam_date = read_date("Exam date (YYYY-MM-DD, blank if unknown): ")
    subject_id = f"sub_{name.lower().replace(' ', '_')}"
    profile.subjects[subject_id] = Subject(subject_id=subject_id, name=name, exam_date=exam_date)
    print(f"Added subject '{name}'.")


def add_topic(profile):
    clear_prompt()
    subject = choose_subject(profile)
    if subject is None:
        return
    name = read_nonempty("Topic name: ")
    difficulty = read_float("Difficulty (0-1): ", default=0.5)
    importance = read_float("Importance (0-1): ", default=0.5)
    topic_id = f"t_{subject.subject_id}_{name.lower().replace(' ', '_')}"
    subject.topics[topic_id] = Topic(
        topic_id=topic_id, name=name, subject_id=subject.subject_id,
        difficulty=difficulty, importance=importance,
    )
    print(f"Added topic '{name}' to {subject.name}.")


def print_plan(actions, trace):
    if not actions:
        print("No plan could be generated. Add subjects and topics first.")
        return
    total_minutes = sum(a.minutes for a in actions)
    print(f"\nGenerated study plan ({total_minutes} minutes, {len(actions)} sessions):\n")
    for i, a in enumerate(actions, 1):
        print(f"  {i}. [{a.action_type.upper()}] {a.topic_name} ({a.subject_name}) - {a.minutes} min "
              f"| utility {a.utility:.3f}")
    print(f"\nSearch trace: explored {len(trace)} beam steps, "
          f"final best utility {trace[-1]['best_utility'] if trace else 0}")


def generate_plan(agent):
    clear_prompt()
    actions, trace = agent.generate_plan()
    print_plan(actions, trace)


def record_assessment_flow(agent, profile):
    clear_prompt()
    subject = choose_subject(profile)
    if subject is None:
        return
    topic = choose_topic(subject)
    if topic is None:
        return
    score = read_float("Score achieved (0-1, e.g. 0.75 for 75%): ")
    kind = input("Assessment type (quiz/practice/exam) [practice]: ").strip() or "practice"
    summary = agent.record_assessment_result(subject.subject_id, topic.topic_id, score, kind)
    print(f"\nUpdated mastery for '{summary['topic']}': "
          f"mastery={summary['mastery']}, uncertainty={summary['uncertainty']} "
          f"(alpha={summary['alpha']}, beta={summary['beta']})")


def replan(agent):
    clear_prompt()
    print("Replanning based on latest assessment results...")
    actions, trace = agent.generate_plan()
    print_plan(actions, trace)
    diff = agent.plan_diff()
    if diff:
        print(f"\nChanges from previous plan: added {diff['added_topics']}, removed {diff['removed_topics']}")


def view_progress(profile, agent):
    clear_prompt()
    print(f"Overall performance measure (importance-weighted mastery): {agent.performance_measure():.3f}\n")
    print("Topic mastery:")
    for row in progress_report(profile):
        print(f"  {row['subject']:<18} {row['topic']:<30} mastery={row['mastery']:<6} "
              f"uncertainty={row['uncertainty']:<6} attempts={row['attempts']:<3} last={row['last_studied']}")

    print("\nWeak topics (mastery < 0.5):")
    weak = weak_topics(profile)
    if not weak:
        print("  None. Good standing across topics.")
    for subj, topic, mastery in weak:
        print(f"  {subj} - {topic}: mastery {mastery}")

    print("\nUpcoming exams:")
    exams = upcoming_exams(profile)
    if not exams:
        print("  No exam dates set.")
    for name, edate, days_left in exams:
        print(f"  {name}: {edate} ({days_left} days left)")

    print("\nMastery change since first recorded assessment:")
    changes = mastery_changes(profile)
    if not changes:
        print("  Not enough history yet.")
    for subj, topic, delta in changes:
        sign = "+" if delta >= 0 else ""
        print(f"  {subj} - {topic}: {sign}{delta}")


def explain_latest(agent):
    clear_prompt()
    explanation = agent.explain_latest()
    if not explanation:
        print("No plan generated yet. Choose option 4 first.")
        return
    print("Why these topics were chosen:\n")
    for c in explanation["chosen"]:
        print(f"  {c['topic']} ({c['subject']}) -> {c['action']} | utility {c['utility']}")
        r = c["reasons"]
        print(f"     mastery={r['mastery']} gap={r['mastery_gap']} predicted_success={r['predicted_success']} "
              f"urgency={r['exam_urgency']} (exam in {r['days_to_exam']}d) importance={r['importance']} "
              f"revision_need={r['revision_need']} repetition_penalty={r['repetition_penalty']}")
    print("\nTop topics considered but not selected:\n")
    for r in explanation["rejected_top"]:
        print(f"  {r['topic']} ({r['subject']}) | utility {r['utility']} | reasons {r['reasons']}")


def print_menu():
    print("\nAdaptive Study Planner")
    print("1. Setup profile")
    print("2. Add subject")
    print("3. Add topic")
    print("4. Generate study plan")
    print("5. Record assessment")
    print("6. Replan")
    print("7. View progress")
    print("8. Explain latest decision")
    print("9. Load demo dataset")
    print("10. Show ML model info")
    print("0. Exit")


def show_ml_info(predictor):
    clear_prompt()
    print("ML component: Logistic Regression (scikit-learn)")
    print("Trained on bundled SYNTHETIC demo data (not real student records).")
    print(f"Held-out test metrics: {predictor.metrics}")
    print(f"Learned feature weights (standardized): {predictor.feature_weights()}")


def main():
    print("Loading Adaptive Study Planner...")
    storage = Storage(DATA_PATH)
    predictor = SuccessPredictor(DATASET_PATH)
    print(f"ML model trained on synthetic data. Test accuracy: {predictor.metrics['accuracy']}, "
          f"f1: {predictor.metrics['f1']}")

    if storage.exists():
        profile = storage.load()
    else:
        profile = StudentProfile()
        print("No saved profile found. Use option 9 to load the demo dataset, or set up your own.")

    agent = StudyPlannerAgent(profile, predictor)

    while True:
        print_menu()
        choice = input("\nChoose an option: ").strip()

        if choice == "1":
            setup_profile(agent.profile)
        elif choice == "2":
            add_subject(agent.profile)
        elif choice == "3":
            add_topic(agent.profile)
        elif choice == "4":
            generate_plan(agent)
        elif choice == "5":
            record_assessment_flow(agent, agent.profile)
        elif choice == "6":
            replan(agent)
        elif choice == "7":
            view_progress(agent.profile, agent)
        elif choice == "8":
            explain_latest(agent)
        elif choice == "9":
            agent.profile = storage.load_demo(DEMO_PATH)
            print("Demo dataset loaded: 2 subjects, 6 topics, with existing history.")
        elif choice == "10":
            show_ml_info(predictor)
        elif choice == "0":
            storage.save(agent.profile)
            print("Progress saved. Goodbye.")
            break
        else:
            print("Invalid option, please choose a number from the menu.")

        storage.save(agent.profile)


if __name__ == "__main__":
    main()
