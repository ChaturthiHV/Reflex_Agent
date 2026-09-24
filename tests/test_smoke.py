"""Smoke tests: run without any API keys (offline mock mode) and check the
core loop, agents and memory store all work end to end.

Run with:
    python -m pytest tests/test_smoke.py -v
or simply:
    python tests/test_smoke.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import experiment_agent, literature_agent, methodology_agent
from memory.memory_store import MemoryStore
from orchestrator import run_baseline, run_domain_task
import config


def test_literature_agent_returns_papers():
    result = literature_agent.run("plant disease classification", max_results=2)
    assert len(result["papers"]) > 0
    assert "title" in result["papers"][0]


def test_memory_store_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "lessons.json")
        store = MemoryStore(path=path)
        store.save_lesson(
            domain="test",
            context="unit test context",
            decision="random_forest",
            outcome="failed: accuracy=0.5",
            lesson="try a stronger model",
            attempt=1,
        )
        lessons = store.get_lessons("test")
        assert len(lessons) == 1
        relevant = store.retrieve_relevant("test", "context", top_k=1)
        assert len(relevant) == 1


def test_experiment_agent_runs_real_model():
    X_y_loader = config.DOMAIN_TASKS["agriculture"]["dataset_loader"]
    result = experiment_agent.run(
        dataset_loader=X_y_loader,
        approach={"model": "random_forest", "params": {"n_estimators": 50}},
        target_metric="accuracy",
        target_value=0.01,  # trivially low so this always passes
    )
    assert result["passed"] is True
    assert 0.0 <= result["metrics"]["accuracy"] <= 1.0


def test_methodology_agent_offline():
    store = MemoryStore(path=tempfile.mktemp(suffix=".json"))
    result = methodology_agent.run(
        domain="agriculture",
        goal="test goal",
        attempt=1,
        memory=store,
        use_llm_reasoning=False,  # keep the test offline and fast
    )
    assert "approach" in result
    assert 0.0 <= result["confidence"] <= 1.0


def test_full_loop_offline_runs_to_completion_or_handoff():
    store = MemoryStore(path=tempfile.mktemp(suffix=".json"))
    result = run_domain_task("agriculture", memory=store, use_llm_reasoning=False, verbose=False)
    assert result["final_result"] is not None or result["handoff"] is not None
    assert len(result["trace"]) >= 1


def test_baseline_runs():
    baseline = run_baseline("agriculture")
    assert "measured_value" in baseline


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        print(f"Running {t.__name__} ...")
        t()
        print("  OK")
    print(f"\nAll {len(tests)} smoke tests passed.")
