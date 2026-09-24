"""Streamlit dashboard for ReflexAgent.

Run with:
    streamlit run dashboard/app.py

Shows the agent-by-agent trace for the selected domain, the lesson
retrieved at each attempt, and a live comparison against a plain
single-shot AI baseline — the evidence the proposal promises.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import streamlit as st

import config
from memory.memory_store import MemoryStore
from orchestrator import run_baseline, run_domain_task

st.set_page_config(page_title="ReflexAgent", layout="wide")
st.title("ReflexAgent — plan, act, evaluate, learn, retry")

domain = st.sidebar.selectbox("Domain", list(config.DOMAIN_TASKS.keys()))
use_llm = st.sidebar.checkbox("Use LLM reasoning (falls back to offline mock if no key set)", value=True)
reset_memory = st.sidebar.checkbox("Clear memory before this run", value=True)

if st.sidebar.button("Run ReflexAgent"):
    memory = MemoryStore()
    if reset_memory:
        memory.clear(domain=domain)

    with st.spinner("Running the loop..."):
        result = run_domain_task(domain, memory=memory, use_llm_reasoning=use_llm, verbose=False)
        baseline = run_baseline(domain)

    st.session_state["result"] = result
    st.session_state["baseline"] = baseline

result = st.session_state.get("result")
baseline = st.session_state.get("baseline")

if not result:
    st.info("Choose a domain in the sidebar and click **Run ReflexAgent**.")
    st.stop()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Literature Agent")
    source = "arXiv (live)" if result["literature"]["live"] else "cached fallback (offline)"
    st.caption(f"Source: {source}")
    for p in result["literature"]["papers"][:3]:
        with st.expander(p["title"]):
            st.write(p["summary"])
            st.caption(f"Gap: {p.get('gap', 'n/a')}")

    st.subheader("Attempt-by-attempt trace")
    for step in result["trace"]:
        m, e = step["methodology"], step["experiment"]
        header = f"Attempt {step['attempt']}: {m['approach']['model']} → " \
                 f"{e['target_metric']}={e['measured_value']} ({'PASS' if e['passed'] else 'miss'})"
        with st.expander(header, expanded=step["attempt"] == len(result["trace"])):
            st.write(f"**Confidence:** {m['confidence']}")
            st.write(f"**Reasoning:** {m['reasoning']}")
            if m["relevant_lessons"]:
                st.write("**Lessons retrieved before this attempt:**")
                for l in m["relevant_lessons"]:
                    st.write(f"- (attempt {l['attempt']}) {l['lesson']}")
            else:
                st.write("_No relevant lessons yet — first attempt on this approach._")
            st.json(e["metrics"])
            if "lesson" in step:
                st.warning(f"New lesson stored: {step['lesson']['lesson']}")

    if result["handoff"]:
        st.error(f"Handed off to a human. Reason: {result['handoff']['reason']}")
    elif result["final_result"]:
        st.success(f"Target reached in {len(result['trace'])} attempt(s).")

with col2:
    st.subheader("ReflexAgent vs. plain single-shot AI")
    reflex_values = [step["experiment"]["measured_value"] for step in result["trace"]]
    target = config.DOMAIN_TASKS[domain]["target_value"]

    fig, ax = plt.subplots()
    ax.plot(range(1, len(reflex_values) + 1), reflex_values, marker="o", label="ReflexAgent")
    ax.axhline(baseline["measured_value"], linestyle="--", color="gray", label="Plain AI (1 shot)")
    ax.axhline(target, linestyle=":", color="green", label="Target")
    ax.set_xlabel("Attempt")
    ax.set_ylabel(config.DOMAIN_TASKS[domain]["target_metric"])
    ax.set_ylim(0, 1)
    ax.legend()
    st.pyplot(fig)

    st.metric("ReflexAgent final", reflex_values[-1])
    st.metric("Plain AI baseline", baseline["measured_value"])
    st.metric("Attempts used", f"{len(result['trace'])} / {config.MAX_ATTEMPTS}")

    st.subheader("All stored lessons")
    lessons = MemoryStore().get_lessons(domain)
    if lessons:
        for l in lessons:
            st.write(f"**Attempt {l['attempt']}:** {l['lesson']}")
    else:
        st.caption("No lessons stored yet for this domain.")
