"""Orchestrator.

Runs the plan -> act -> observe -> evaluate -> replan loop:

    1. (once) Literature Agent looks up relevant approaches.
    2. Methodology Agent proposes an approach + confidence, using any
       relevant lessons already in memory.
    3. Experiment Agent runs it for real and measures the result.
    4. If the target is met -> done.
       If confidence was below the floor, or the attempt limit is reached
       -> hand off to a human with the full reasoning trace.
       Otherwise -> Failure Agent diagnoses the miss, writes a lesson, and
       the loop goes back to step 2 (which will retrieve that lesson).

Every attempt is appended to `trace` so the caller (CLI demo or dashboard)
can display full transparency: what was tried, why, what happened, and
what was learned.
"""
import config
from agents import experiment_agent, failure_agent, literature_agent, methodology_agent
from memory.memory_store import MemoryStore


def run_domain_task(domain: str, memory: MemoryStore = None, use_llm_reasoning: bool = True, verbose: bool = True):
    if domain not in config.DOMAIN_TASKS:
        raise ValueError(f"Unknown domain '{domain}'. Registered: {list(config.DOMAIN_TASKS)}")

    task = config.DOMAIN_TASKS[domain]
    memory = memory or MemoryStore()

    def log(msg):
        if verbose:
            print(msg)

    log(f"\n=== ReflexAgent: {domain} ===")
    log(f"Goal: {task['goal']}")

    lit_result = literature_agent.run(task["literature_query"])
    source = "arXiv (live)" if lit_result["live"] else "cached fallback (offline)"
    log(f"\n[Literature Agent] {len(lit_result['papers'])} papers via {source}")
    for p in lit_result["papers"][:3]:
        log(f"  - {p['title']}")

    trace = []
    handoff = None
    final_result = None

    for attempt in range(1, config.MAX_ATTEMPTS + 1):
        log(f"\n--- Attempt {attempt}/{config.MAX_ATTEMPTS} ---")

        methodology = methodology_agent.run(
            domain=domain,
            goal=task["goal"],
            attempt=attempt,
            memory=memory,
            use_llm_reasoning=use_llm_reasoning,
        )
        log(f"[Methodology Agent] approach={methodology['approach']['model']} "
            f"confidence={methodology['confidence']}")
        if methodology["relevant_lessons"]:
            log(f"  retrieved {len(methodology['relevant_lessons'])} lesson(s), most recent: "
                f"\"{methodology['relevant_lessons'][0]['lesson']}\"")
        log(f"  reasoning: {methodology['reasoning']}")

        if methodology["confidence"] < config.CONFIDENCE_FLOOR:
            handoff = {
                "reason": "confidence_below_floor",
                "attempt": attempt,
                "confidence": methodology["confidence"],
                "methodology": methodology,
            }
            log(f"[Orchestrator] Confidence {methodology['confidence']} below floor "
                f"{config.CONFIDENCE_FLOOR} -> human handoff.")
            break

        experiment = experiment_agent.run(
            dataset_loader=task["dataset_loader"],
            approach=methodology["approach"],
            target_metric=task["target_metric"],
            target_value=task["target_value"],
        )
        log(f"[Experiment Agent] {experiment['target_metric']}={experiment['measured_value']} "
            f"(target {experiment['target_value']}) -> {'PASS' if experiment['passed'] else 'miss'}")

        step = {"attempt": attempt, "methodology": methodology, "experiment": experiment}

        if experiment["passed"]:
            final_result = experiment
            trace.append(step)
            log(f"\n[Orchestrator] Target reached on attempt {attempt}. Done.")
            break

        lesson = failure_agent.run(
            domain=domain,
            goal=task["goal"],
            attempt=attempt,
            experiment_result=experiment,
            memory=memory,
            use_llm_reasoning=use_llm_reasoning,
        )
        log(f"[Failure Agent] lesson stored: \"{lesson['lesson']}\"")
        step["lesson"] = lesson
        trace.append(step)

        if attempt == config.MAX_ATTEMPTS:
            handoff = {
                "reason": "attempt_limit_reached",
                "attempt": attempt,
                "last_experiment": experiment,
            }
            log(f"\n[Orchestrator] Attempt limit ({config.MAX_ATTEMPTS}) reached without meeting "
                f"target -> human handoff.")

    return {
        "domain": domain,
        "goal": task["goal"],
        "literature": lit_result,
        "trace": trace,
        "final_result": final_result,
        "handoff": handoff,
    }


def run_baseline(domain: str):
    """Single-shot 'plain AI' baseline: one attempt, no memory, no retries.

    Used by the dashboard/demo to show the value of the ReflexAgent loop
    against the simplest possible alternative.
    """
    task = config.DOMAIN_TASKS[domain]
    approach = methodology_agent.CANDIDATES[0]  # always the simplest default, no lesson retrieval
    experiment = experiment_agent.run(
        dataset_loader=task["dataset_loader"],
        approach=approach,
        target_metric=task["target_metric"],
        target_value=task["target_value"],
    )
    return experiment
