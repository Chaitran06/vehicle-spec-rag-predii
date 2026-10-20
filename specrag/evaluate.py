"""Golden-set evaluation — retrieval and extraction, measured separately.

Same lesson as the DSA project's evaluate.py: without a golden set, "it looks
right on the three queries I keep typing" is the only signal, and it lies.

Two golden files:

  eval/handwritten.json   ~20 questions phrased the way a technician would
                          ask, with answers checked by hand against the
                          manual, including several where the right answer
                          is "not in this manual".
  eval/generated.json     one question per distinct torque-table component,
                          generated from the deterministic parser
                          (`specrag golden`). Wide coverage, literal phrasing.

Metrics:
  retrieval  row recall@k — share of the expected source rows that appear
             in the retrieved chunks (the LLM cannot extract what it never
             sees). Run per method: dense, bm25, hybrid.
  extraction precision / recall over (value, section) pairs, so returning
             the RWD value for a 4WD row counts as wrong, plus
             "refusal accuracy" on the not-in-manual questions.
"""

import json
import random
import re
from collections import defaultdict
from pathlib import Path

from specrag.config import PROJECT_ROOT, TOP_K

EVAL_DIR = PROJECT_ROOT / "eval"
HANDWRITTEN = EVAL_DIR / "handwritten.json"
GENERATED = EVAL_DIR / "generated.json"


def _norm_value(v: str) -> str:
    return re.sub(r"[\s,]", "", str(v)).replace("–", "-").replace("—", "-")


def _norm_line(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def load_golden(which: str = "all") -> list[dict]:
    items = []
    if which in ("all", "handwritten") and HANDWRITTEN.exists():
        items += [dict(e, origin="handwritten") for e in json.loads(HANDWRITTEN.read_text(encoding="utf-8"))
                  if not str(e.get("q", "")).startswith("_")]
    if which in ("all", "generated") and GENERATED.exists():
        items += [dict(e, origin="generated") for e in json.loads(GENERATED.read_text(encoding="utf-8"))]
    return items


def make_generated(rows: list[dict], path: Path = GENERATED) -> int:
    """One query per distinct torque-table component name."""
    by_component: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["spec_type"] == "Torque" and r["source"] == "table":
            by_component[r["component"].lower()].append(r)
    items = []
    for component, matches in sorted(by_component.items()):
        items.append({
            "q": f"torque for {component}",
            "expect": [
                {"value": m["value"], "unit": m["unit"], "section": m["section_id"], "line": m["line"]}
                for m in matches
            ],
        })
    path.write_text(json.dumps(items, indent=1, ensure_ascii=False), encoding="utf-8")
    return len(items)


# ------------------------------------------------------------------
# Retrieval
# ------------------------------------------------------------------
def eval_retrieval(items: list[dict], k: int = TOP_K, methods=("dense", "bm25", "hybrid")) -> dict:
    from specrag.retrieve import search

    report: dict = {"k": k, "n": 0, "methods": {}}
    scored = [e for e in items if e.get("expect")]
    report["n"] = len(scored)
    for method in methods:
        recalls, hits, misses = [], 0, []
        by_origin: dict[str, list[float]] = defaultdict(list)
        for e in scored:
            retrieved = search(e["q"], top_k=k, method=method)
            texts = [(h.chunk.section_id, _norm_line(h.chunk.text)) for h in retrieved]
            found = 0
            for exp in e["expect"]:
                line = _norm_line(exp.get("line", ""))
                if any(sec == exp["section"] and (line in text if line else _norm_value(exp["value"]) in _norm_value(text))
                       for sec, text in texts):
                    found += 1
            recall = found / len(e["expect"])
            recalls.append(recall)
            by_origin[e.get("origin", "?")].append(recall)
            hits += found > 0
            if recall < 1:
                misses.append({"q": e["q"], "recall": round(recall, 2)})
        report["methods"][method] = {
            "row_recall": round(sum(recalls) / len(recalls), 4) if recalls else 0.0,
            "hit_rate": round(hits / len(scored), 4) if scored else 0.0,
            # Reported separately: generated queries reuse the manual's exact
            # wording, which flatters BM25. Handwritten ones are the honest test.
            "row_recall_by_origin": {o: round(sum(v) / len(v), 4) for o, v in by_origin.items()},
            "misses": misses,
        }
    return report


# ------------------------------------------------------------------
# Extraction
# ------------------------------------------------------------------
def eval_extraction(items: list[dict], mode: str, k: int = TOP_K, sample: int | None = None, seed: int = 7) -> dict:
    from specrag.extractor import extract

    if sample is not None:
        hand = [e for e in items if e["origin"] == "handwritten"]
        gen = [e for e in items if e["origin"] == "generated"]
        random.Random(seed).shuffle(gen)
        items = hand + gen[:sample]

    tp = fp = fn = 0
    refusal_ok = refusal_n = 0
    details = []
    # Per origin, because the generated gold is strict exact-name matching:
    # "torque for brake tube fitting" expects only rows named exactly that, so
    # a correct "Primary brake tube fitting" row counts against precision.
    # The handwritten set, with hand-checked answers, is the honest number.
    per: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    errors = []
    for e in items:
        try:
            result = extract(e["q"], top_k=k, mode=mode)
        except Exception as exc:  # rate limit outlasted the wait, network...
            # Recorded and excluded from the metrics rather than crashing the
            # run and losing every answer collected so far.
            errors.append({"q": e["q"], "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
            continue
        predicted = {(_norm_value(s.value), s.section) for s in result.specs}

        if e.get("expect_none"):
            refusal_n += 1
            ok = not predicted
            refusal_ok += ok
            details.append({"q": e["q"], "expect_none": True, "ok": ok,
                            "got": sorted(f"{s.value} {s.unit} [{s.section}]" for s in result.specs)})
            continue

        gold = {(_norm_value(x["value"]), x["section"]) for x in e["expect"]}
        # Same value stated again in another section (e.g. 206-00 General
        # Information repeating the front disc limit) is not wrong, just not
        # required.
        acceptable = gold | {(_norm_value(x["value"]), x["section"]) for x in e.get("accept_also", [])}
        # A prediction is correct if its (value, section) is expected. Extra
        # rows from the same table that the query did not ask for count as
        # false positives — returning the whole table is not an answer.
        q_tp = len(predicted & gold)
        q_fp = len(predicted - acceptable)
        q_fn = len(gold - predicted)
        tp, fp, fn = tp + q_tp, fp + q_fp, fn + q_fn
        o = per[e.get("origin", "?")]
        o[0], o[1], o[2] = o[0] + q_tp, o[1] + q_fp, o[2] + q_fn
        details.append({
            "q": e["q"], "tp": q_tp, "fp": q_fp, "fn": q_fn,
            "missing": sorted(f"{v} [{s}]" for v, s in gold - predicted),
            "extra": sorted(f"{v} [{s}]" for v, s in predicted - gold),
            "rejected": len(result.rejected),
        })

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "mode": mode,
        "n": len(items),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0,
        "refusal_accuracy": round(refusal_ok / refusal_n, 4) if refusal_n else None,
        "errors": errors,
        "by_origin": {
            origin: {
                "precision": round(t / (t + f), 4) if t + f else 0.0,
                "recall": round(t / (t + n), 4) if t + n else 0.0,
            }
            for origin, (t, f, n) in per.items()
        },
        "details": details,
    }
