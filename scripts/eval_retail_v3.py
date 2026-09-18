#!/usr/bin/env python3
"""Evaluate the typed calculator planner on the same development evidence."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "retrieval"))
from retail_scoped import ScopedRetailSearch, query_scope, VERSION as RETRIEVER_VERSION
from retail_evidence import add_catalog, VERSION as EVIDENCE_VERSION
from retail_planner_v3 import build_plan_messages, parse_plan, execute, contract, aliases, VERSION as CALCULATOR_VERSION, SYSTEM as PLAN_SYSTEM
from retail_scoring import score_v2, VERSION as SCORER_VERSION
from eval_retail_baseline import (build_messages, request_model, parse_prediction, evidence_metrics,
                                  score_prediction, summarize, SYSTEM)
from retail_pilot import read_json, read_jsonl, write_json, sha256


def build_context(search, question, as_of, top_k=6):
    hits = search.search(question, as_of, top_k)
    # Fixed identical evidence budget for both arms: raw context first, then
    # catalog records in the remaining space. All omissions are recorded.
    raw = search.context(hits, as_of, max_chars=14000)
    return hits, add_catalog(raw, search.by_id, max_chars=18000, question=question)


def extended_summary(rows, mode):
    summary = summarize(rows, mode)
    summary["all_annotated_cells_in_catalog_count"] = sum(r["all_annotated_cells_in_catalog"] is True for r in rows)
    summary["calculator_calls"] = sum(r["calculator_calls"] for r in rows)
    summary["guard_rejections"] = sum(bool(r.get("guard_rejection")) for r in rows)
    if mode == "generate":
        count = sum(r.get("scores_v2", {}).get("answer_content_correct", False) for r in rows)
        summary.update({"answer_content_correct_count": count, "answer_content_accuracy": count / len(rows),
                        "formatting_only_mismatches": sum(r.get("scores_v2", {}).get("answer_content_correct", False)
                            and not r.get("scores", {}).get("strict_answer_correct", False) for r in rows)})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["retrieve", "generate"])
    parser.add_argument("--arm", choices=["scoped-calc"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data/retail_pilot")
    parser.add_argument("--base-url", default=os.environ.get("PILOT_LLM_BASE_URL"))
    parser.add_argument("--model", default=os.environ.get("PILOT_LLM_MODEL"))
    args = parser.parse_args()
    if args.mode == "generate":
        if not args.base_url or not args.model:
            parser.error("Generation needs --base-url and --model")
        url = urlsplit(args.base_url)
        if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password or url.query or url.fragment:
            parser.error("Invalid endpoint URL")
    qpath = args.data / "splits/train/questions.draft.jsonl"
    cpath = args.data / "splits/train/corpus.json"
    questions, corpus = read_jsonl(qpath), read_json(cpath)
    if len(questions) != 27 or any(q["split"] != "train" for q in questions) or any(c["split"] != "train" for c in corpus):
        raise ValueError("Only the 27-question development split is allowed")
    args.output.mkdir(parents=True, exist_ok=False)
    files = [Path(__file__).resolve(), ROOT / "scripts/retail_pilot.py", ROOT / "scripts/eval_retail_baseline.py",
             ROOT / "scripts/retail_planner_v3.py", ROOT / "scripts/retail_scoring.py", ROOT / "scripts/retail_calculator.py", ROOT / "scripts/retail_evidence.py",
             ROOT / "retrieval/retail_scoped.py", ROOT / "retrieval/retail_search.py"]
    config = {"arm": args.arm, "mode": args.mode, "split": "train", "retriever": RETRIEVER_VERSION,
              "scorer": SCORER_VERSION, "evidence_adapter": EVIDENCE_VERSION,
              "calculator": CALCULATOR_VERSION if args.arm == "scoped-calc" else None,
              "started_at": datetime.now(timezone.utc).isoformat(), "questions_sha256": sha256(qpath),
              "corpus_sha256": sha256(cpath), "source_hashes": {str(p.relative_to(ROOT)): sha256(p) for p in files},
              "model": args.model, "base_url": args.base_url, "temperature": 0, "disable_thinking": True,
              "top_k": 6, "max_raw_context_chars": 14000, "max_total_context_chars": 18000, "max_tokens": 2048,
              "timeout_seconds": 180, "system_prompt": PLAN_SYSTEM if args.arm == "scoped-calc" else SYSTEM,
              "notes": "v3 named input slots and short aliases. Same v2 evidence and scoring; original source guards retained. One request per question, no retries. Development only."}
    write_json(args.output / "config.json", config)
    start = time.perf_counter()
    search = ScopedRetailSearch(corpus)
    config["index_seconds"] = time.perf_counter() - start
    rows = []
    with (args.output / "results.jsonl").open("w", encoding="utf-8") as out:
        for q in questions:
            start = time.perf_counter()
            scope = query_scope(q["question"], corpus)
            hits, context = build_context(search, q["question"], q["as_of"])
            retrieval_seconds = time.perf_counter() - start
            ids = [s["chunk_id"] for s in context["sections"]]
            # Reference evidence is touched only after retrieval/context creation.
            catalog_ids = {c["cell_id"] for c in context["cell_catalog"]}
            gold_ids = {f"{e['table_id']}:r{e['row_index']}:c{e['column_index']}" for e in q["evidence"]}
            row = {"id": q["id"], "question": q["question"], "question_type": q["question_type"],
                   "answerable": q["answerable"], "as_of": q["as_of"], "reference_answer": q["answer"], "scope": scope,
                   "ranked_hits": hits, "context": context,
                   "ranked_evidence": evidence_metrics(q, [h["chunk_id"] for h in hits], search.by_id),
                   "presented_evidence": evidence_metrics(q, ids, search.by_id),
                   "all_annotated_cells_in_catalog": gold_ids <= catalog_ids if gold_ids else None,
                   "missing_annotated_cells": sorted(gold_ids - catalog_ids),
                   "date_filter_violations": sum(bool(q["as_of"] and search.by_id[i]["published_date"] > q["as_of"]) for i in ids),
                   "retrieval_seconds": retrieval_seconds, "retrieval_calls": 1, "llm_calls": 0, "calculator_calls": 0}
            if row["date_filter_violations"]:
                raise AssertionError("Future source entered context")
            if args.mode == "generate":
                start = time.perf_counter()
                row["llm_calls"] = 1
                row["planning_contract"] = contract(q["question"], scope)
                row["cell_aliases"] = aliases(context)
                try:
                    messages = (build_plan_messages(q["question"], q["as_of"], context, scope) if args.arm == "scoped-calc"
                                else build_messages(q["question"], q["as_of"], context["text"]))
                    response = request_model(args.base_url, args.model, messages, 2048, 180, True)
                    row["response"] = response
                    if response["finish_reason"] != "stop":
                        raise ValueError("Model did not finish normally")
                    if args.arm == "scoped-calc":
                        plan = parse_plan(response["raw"])
                        row["plan"] = plan
                        row["calculator_calls"] = 1
                        try:
                            prediction, trace = execute(plan, q["question"], q["as_of"], scope, context, search.by_id)
                            row["calculation_trace"] = trace
                        except ValueError as exc:
                            row["guard_rejection"] = str(exc)
                            prediction = {"answer": "所提供证据不足以通过核验。", "abstain": True, "citations": []}
                    else:
                        prediction = parse_prediction(response["raw"])
                    row["prediction"] = prediction
                    row["scores"] = score_prediction(q, prediction, ids, search.by_id)
                    row["scores_v2"] = score_v2(q, prediction, ids)
                    row["canonical_citation_evidence"] = evidence_metrics(q,
                        [c for c in row["scores_v2"]["canonical_citations"] if c in set(ids)], search.by_id)
                except Exception as exc:
                    row["prediction"] = None
                    row["error"] = f"{type(exc).__name__}: {exc}"
                row["generation_seconds"] = time.perf_counter() - start
            rows.append(row)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            print(f"{len(rows)}/27 {q['id']} catalog_complete={row['all_annotated_cells_in_catalog']} "
                  f"correct={row.get('scores_v2', {}).get('answer_content_correct', 'not scored')} "
                  f"guard={row.get('guard_rejection', '')}" + (" ERROR" if row.get("error") else ""), flush=True)
    summary = extended_summary(rows, args.mode)
    summary["by_question_type"] = {kind: extended_summary([r for r in rows if r["question_type"] == kind], args.mode)
                                   for kind in sorted({r["question_type"] for r in rows})}
    write_json(args.output / "summary.json", summary)
    config["completed_at"] = datetime.now(timezone.utc).isoformat()
    write_json(args.output / "config.json", config)
    lines = ["# Retail v3 development evaluation: " + args.arm, "",
             "Development-only, provisional references. Content accuracy does not certify citation support. "
             "Annotated source coverage misses some valid alternative evidence. Guard rejection is a system abstention, not a correct answer by itself.", "",
             "| Metric | Value |", "| --- | --- |"]
    for k, v in summary.items():
        if k != "by_question_type":
            lines.append(f"| {k} | {v} |")
    lines += ["", "| Question | Content correct | Guard rejection / error |", "| --- | --- | --- |"]
    for r in rows:
        lines.append(f"| {r['id']} | {r.get('scores_v2', {}).get('answer_content_correct', 'not run')} | {r.get('guard_rejection', r.get('error', ''))} |")
    (args.output / "report.md").write_text("\n".join(lines) + "\n")
    return 2 if args.mode == "generate" and summary["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
