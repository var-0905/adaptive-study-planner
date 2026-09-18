# Problem Statement and Scope

## Problem statement

Most students study without a clear sense of which topics actually need attention. They revise what
feels comfortable, ignore weak areas until an exam is close, and rarely track how their understanding
changes over time. A planner that only lets you write down "study chapter 3 on Monday" does not solve
this, because it has no model of how well you actually know chapter 3 or how urgently it needs review.

This project builds an agent that keeps a running estimate of what a student knows, how confident that
estimate is, and how urgent each topic is given exam dates and forgetting, and then uses that estimate
to generate a study plan through search rather than a fixed rule or a static checklist.

## Project scope

The scope is a single user, offline, terminal based planner. It covers:

- storing subjects, topics and a daily time budget
- a Bayesian model of topic mastery that updates from assessment results
- a small ML classifier that predicts the probability of a successful learning session for a topic
- a best first / beam search planner that picks a sequence of study actions under a time budget
- basic analytics: progress, weak topics, upcoming exams, and an explanation of why the planner
  chose what it chose

It does not cover multi user accounts, content delivery (actual practice questions), calendar sync,
notifications, or anything that requires an internet connection or an external LLM API. The reasoning
here is entirely local: Bayesian updates and a scikit-learn model trained on a small bundled synthetic
dataset.

## Target users

The primary target is a student preparing for exams across a handful of subjects who wants a simple
tool to decide what to study each day, and to see that recommendation update as their performance
changes. It is also meant to work as a course project that demonstrates the AI/ML concepts from the
Fundamentals of AI and ML syllabus in a working, testable system rather than slides.

## High level features

- add subjects with exam dates, add topics with difficulty and importance
- record quiz/practice/exam scores per topic
- Beta-Bayesian mastery tracking with an explicit uncertainty value per topic
- logistic regression model predicting learning success probability from topic features
- beam search planner producing a ranked daily plan under a time budget, with a printed reasoning
  trace for every chosen and rejected topic
- replanning after new assessment results, with a before/after comparison
- progress view: mastery per topic, weak topics, upcoming exams, mastery trend
- bundled demo dataset so the whole pipeline can be seen working immediately
