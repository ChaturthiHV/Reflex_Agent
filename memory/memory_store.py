"""Reflexive memory: stores and retrieves structured lessons.

Each lesson has the shape:
    {
        "domain": "agriculture",
        "context": "...",     # what was being attempted
        "decision": "...",    # what approach/hyperparameters were chosen
        "outcome": "...",     # what happened (metric value, pass/fail)
        "lesson": "...",      # the actionable takeaway for next time
        "attempt": 2,
        "timestamp": "..."
    }

Backed by a plain JSON file by default (works everywhere, easy to inspect
and demo). If chromadb is installed, `use_semantic=True` layers a vector
index on top for similarity retrieval instead of keyword overlap — this is
the "optional upgrade" mentioned in the proposal's tech stack.
"""
import json
import os
import time

import config


class MemoryStore:
    def __init__(self, path: str = config.LESSONS_PATH, use_semantic: bool = False):
        self.path = path
        self.use_semantic = use_semantic
        self._collection = None
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if not os.path.exists(self.path):
            self._write([])

        if self.use_semantic:
            try:
                import chromadb

                client = chromadb.Client()
                self._collection = client.get_or_create_collection("reflexagent_lessons")
                self._sync_semantic_index()
            except Exception:
                # Silently degrade to keyword search if chromadb isn't usable
                # in this environment (e.g. no compiled deps available).
                self.use_semantic = False

    # --- storage plumbing ---------------------------------------------------
    def _read(self):
        with open(self.path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write(self, lessons):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(lessons, f, indent=2)

    def _sync_semantic_index(self):
        lessons = self._read()
        if not lessons:
            return
        ids = [str(i) for i in range(len(lessons))]
        docs = [f"{l['context']} {l['decision']} {l['lesson']}" for l in lessons]
        self._collection.upsert(ids=ids, documents=docs)

    # --- public API ----------------------------------------------------------
    def save_lesson(self, domain: str, context: str, decision: str, outcome: str, lesson: str, attempt: int):
        entry = {
            "domain": domain,
            "context": context,
            "decision": decision,
            "outcome": outcome,
            "lesson": lesson,
            "attempt": attempt,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        lessons = self._read()
        lessons.append(entry)
        self._write(lessons)
        if self.use_semantic:
            try:
                self._collection.add(
                    ids=[str(len(lessons) - 1)],
                    documents=[f"{context} {decision} {lesson}"],
                )
            except Exception:
                pass
        return entry

    def get_lessons(self, domain: str = None):
        lessons = self._read()
        if domain:
            return [l for l in lessons if l["domain"] == domain]
        return lessons

    def retrieve_relevant(self, domain: str, query: str, top_k: int = 3):
        """Return the most relevant past lessons for this domain + query.

        Uses chromadb similarity search when available, otherwise a simple
        keyword-overlap score. Either way the retrieval — and the fact that
        it happened — is meant to be shown to the user for transparency.
        """
        domain_lessons = self.get_lessons(domain)
        if not domain_lessons:
            return []

        if self.use_semantic and self._collection is not None:
            try:
                res = self._collection.query(query_texts=[query], n_results=min(top_k, len(domain_lessons)))
                idxs = [int(i) for i in res["ids"][0]]
                all_lessons = self._read()
                return [all_lessons[i] for i in idxs]
            except Exception:
                pass  # fall through to keyword search

        query_words = set(query.lower().split())

        def score(lesson):
            text_words = set((lesson["context"] + " " + lesson["decision"] + " " + lesson["lesson"]).lower().split())
            return len(query_words & text_words)

        ranked = sorted(domain_lessons, key=score, reverse=True)
        return [l for l in ranked[:top_k]]

    def clear(self, domain: str = None):
        """Mainly for tests/demos: wipe lessons (optionally scoped to one domain)."""
        if domain is None:
            self._write([])
        else:
            remaining = [l for l in self._read() if l["domain"] != domain]
            self._write(remaining)
