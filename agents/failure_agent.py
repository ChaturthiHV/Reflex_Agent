"""Failure and Improvement Agent.

When an experiment misses its target, this agent diagnoses a likely cause,
recommends a specific fix, and writes a structured lesson (context,
decision, outcome, lesson) to memory. The next Methodology Agent call
retrieves and visibly applies this lesson — this loop is the core
self-improvement mechanism described in the proposal.
"""
from llm import client as llm_client
from memory.memory_store import MemoryStore

# Rule-based diagnosis guarantees a usable lesson even with no LLM key
# configured, or if the LLM call fails outright.
_RULE_BASED_FIXES = {
    "logistic_regression": "Linear decision boundary is likely too simple for this class structure; move to a model that can fit non-linear boundaries (SVM with an RBF kernel or a tree ensemble).",
    "svm_rbf": "The kernel may be under- or over-fit; try a tree ensemble which handles noisy, mixed-scale features more robustly without kernel tuning.",
    "random_forest": "The forest may need more trees or deeper trees to capture the class boundaries; alternatively try gradient boosting, which often needs fewer estimators for the same accuracy.",
    "gradient_boosting": "Already using a strong model; the likely limiter is feature quality or class overlap rather than model choice — consider more estimators and a lower learning rate for a finer fit.",
}


def _rule_based_lesson(experiment_result: dict) -> str:
    model = experiment_result["model"]
    gap = round(experiment_result["target_value"] - experiment_result["measured_value"], 4)
    fix = _RULE_BASED_FIXES.get(model, "Try a different model family or adjust hyperparameters.")
    return f"Missed target by {gap}. {fix}"


def run(domain: str, goal: str, attempt: int, experiment_result: dict, memory: MemoryStore, use_llm_reasoning: bool = True):
    outcome = (
        f"failed: {experiment_result['target_metric']}="
        f"{experiment_result['measured_value']} (target {experiment_result['target_value']})"
    )
    decision = f"{experiment_result['model']} {experiment_result['params']}"
    lesson_text = _rule_based_lesson(experiment_result)

    if use_llm_reasoning:
        prompt = (
            f"Goal: {goal}\nDomain: {domain}\nAttempt: {attempt}\n"
            f"Model tried: {experiment_result['model']} with params {experiment_result['params']}\n"
            f"Result: {experiment_result['target_metric']} = {experiment_result['measured_value']} "
            f"(target {experiment_result['target_value']})\n\n"
            "Diagnose the most likely reason this missed the target and recommend one concrete fix "
            "for the next attempt. Respond in JSON with keys 'diagnosis' and 'fix', each one sentence."
        )
        result = llm_client.complete_json(prompt, system="You are the Failure and Improvement Agent for ReflexAgent.")
        if not result.get("parse_error") and (result.get("diagnosis") or result.get("fix")):
            diagnosis = result.get("diagnosis", "").strip()
            fix = result.get("fix", "").strip()
            lesson_text = f"{diagnosis} Fix: {fix}".strip() if diagnosis else f"Fix: {fix}"

    entry = memory.save_lesson(
        domain=domain,
        context=goal,
        decision=decision,
        outcome=outcome,
        lesson=lesson_text,
        attempt=attempt,
    )
    return entry
