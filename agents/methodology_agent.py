"""Methodology Agent.

Retrieves relevant past lessons for this domain, shows them, then proposes
an approach (model family + hyperparameters), a dataset note, and a
confidence level. When an LLM key is configured it is asked to reason over
the retrieved lessons; either way, a deterministic rule-based candidate
list guarantees the agent always returns something runnable for the
Experiment Agent, so a flaky LLM call can never stall the loop.
"""
from llm import client as llm_client
from memory.memory_store import MemoryStore

# Candidate approaches, roughly ordered from simplest to most capable.
# The rule-based picker walks this list, skipping anything a retrieved
# lesson says already failed.
CANDIDATES = [
    {"model": "logistic_regression", "params": {"max_iter": 500}},
    {"model": "svm_rbf", "params": {"C": 1.0, "gamma": "scale"}},
    {"model": "random_forest", "params": {"n_estimators": 100, "max_depth": None}},
    {"model": "random_forest", "params": {"n_estimators": 300, "max_depth": 12}},
    {"model": "gradient_boosting", "params": {"n_estimators": 200, "learning_rate": 0.05}},
]


def _rule_based_pick(attempt: int, avoid_models: set):
    for candidate in CANDIDATES:
        if candidate["model"] not in avoid_models:
            return candidate
    # Every simple candidate was already tried and failed; escalate to the
    # strongest option with wider hyperparameters as a last resort.
    return {"model": "gradient_boosting", "params": {"n_estimators": 400, "learning_rate": 0.03}}


def _confidence_for(candidate: dict, attempt: int, num_relevant_lessons: int) -> float:
    base = {
        "logistic_regression": 0.45,
        "svm_rbf": 0.55,
        "random_forest": 0.65,
        "gradient_boosting": 0.7,
    }.get(candidate["model"], 0.5)
    # More relevant lessons available -> more informed choice -> higher confidence.
    base += 0.05 * num_relevant_lessons
    # Slight caution discount on very first attempt (nothing learned yet).
    if attempt == 1:
        base -= 0.05
    return round(min(base, 0.95), 2)


def run(domain: str, goal: str, attempt: int, memory: MemoryStore, use_llm_reasoning: bool = True):
    query = f"{domain} {goal}"
    relevant_lessons = memory.retrieve_relevant(domain, query, top_k=3)
    avoid_models = {
        l["decision"].split(" ")[0]
        for l in relevant_lessons
        if "failed" in l.get("outcome", "").lower()
    }

    candidate = _rule_based_pick(attempt, avoid_models)
    confidence = _confidence_for(candidate, attempt, len(relevant_lessons))
    reasoning = (
        f"Rule-based selection: picked '{candidate['model']}' after checking "
        f"{len(relevant_lessons)} retrieved lesson(s); avoided {sorted(avoid_models) or 'nothing'}."
    )

    if use_llm_reasoning:
        lessons_text = "\n".join(
            f"- attempt {l['attempt']}: decision={l['decision']} outcome={l['outcome']} lesson={l['lesson']}"
            for l in relevant_lessons
        ) or "(no relevant past lessons yet)"
        prompt = (
            f"Goal: {goal}\nDomain: {domain}\nAttempt number: {attempt}\n"
            f"Retrieved lessons:\n{lessons_text}\n\n"
            "Given this, is the following approach reasonable? Respond with JSON keys "
            "'agree' (true/false), 'note' (one short sentence)."
            f"\nProposed approach: {candidate['model']} with params {candidate['params']}"
        )
        result = llm_client.complete_json(prompt, system="You are the Methodology Agent for ReflexAgent.")
        if not result.get("parse_error"):
            reasoning += f" LLM check ({result.get('_provider', 'unknown')}): {result.get('note', '')}"

    return {
        "attempt": attempt,
        "approach": candidate,
        "confidence": confidence,
        "reasoning": reasoning,
        "relevant_lessons": relevant_lessons,
    }
