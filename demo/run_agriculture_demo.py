"""End-to-end CLI demo of the agriculture use case.

Run from the project root:
    python demo/run_agriculture_demo.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.memory_store import MemoryStore
from orchestrator import run_baseline, run_domain_task


def main():
    memory = MemoryStore()
    memory.clear(domain="agriculture")  # fresh run for a clean demo

    result = run_domain_task("agriculture", memory=memory, use_llm_reasoning=True, verbose=True)

    print("\n=== Summary ===")
    if result["final_result"]:
        fr = result["final_result"]
        print(f"Target reached with {fr['model']}: {fr['target_metric']}={fr['measured_value']} "
              f"in {len(result['trace'])} attempt(s).")
    else:
        print(f"Target not reached. Handed off to a human. Reason: {result['handoff']['reason']}")

    baseline = run_baseline("agriculture")
    print(f"\nPlain single-shot baseline ({baseline['model']}): "
          f"{baseline['target_metric']}={baseline['measured_value']} "
          f"(target {baseline['target_value']}) -> {'PASS' if baseline['passed'] else 'miss'}")

    reflex_iterations = len(result["trace"])
    reflex_final = result["final_result"]["measured_value"] if result["final_result"] else result["trace"][-1]["experiment"]["measured_value"]
    print(f"ReflexAgent needed {reflex_iterations} attempt(s) and reached {reflex_final}; "
          f"the plain baseline took 1 attempt and reached {baseline['measured_value']}.")

    print("\nStored lessons for 'agriculture':")
    for lesson in memory.get_lessons("agriculture"):
        print(f"  attempt {lesson['attempt']}: {lesson['lesson']}")


if __name__ == "__main__":
    main()
