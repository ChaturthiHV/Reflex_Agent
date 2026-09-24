"""Literature Agent.

Retrieves real papers from arXiv (no API key needed), summarizes what each
paper does, and notes an open gap. If the network call fails for any
reason, it falls back to a small curated JSON cache so the demo never
stalls on a bad connection.
"""
import json

import feedparser
import requests

import config

ARXIV_API = "http://export.arxiv.org/api/query"


def _search_arxiv(query: str, max_results: int = 3):
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    resp = requests.get(ARXIV_API, params=params, timeout=config.REQUEST_TIMEOUT_SECONDS)
    resp.raise_for_status()
    feed = feedparser.parse(resp.text)
    if not feed.entries:
        raise RuntimeError("arXiv returned no entries")

    papers = []
    for entry in feed.entries:
        summary = entry.summary.strip().replace("\n", " ")
        papers.append(
            {
                "title": entry.title.strip().replace("\n", " "),
                "summary": summary[:400],
                "gap": "See full abstract for limitations and future work.",
                "source": "arxiv",
                "link": entry.link,
            }
        )
    return papers


def _load_cache():
    with open(config.CACHED_PAPERS_PATH, "r", encoding="utf-8") as f:
        cached = json.load(f)
    for p in cached:
        p["source"] = "cache"
    return cached


def run(query: str, max_results: int = 3):
    """Return a list of paper dicts and whether the live source was used."""
    try:
        papers = _search_arxiv(query, max_results)
        return {"papers": papers, "live": True}
    except Exception as e:  # network down, arXiv rate-limited, etc.
        papers = _load_cache()[:max_results]
        return {"papers": papers, "live": False, "fallback_reason": str(e)}
