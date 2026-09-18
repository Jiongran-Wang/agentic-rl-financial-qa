#!/usr/bin/env python3
"""Reproducible, evidence-first retail dataset pilot. No model/API key required.

Use a Python environment with requirements-data.txt installed. See
docs/retail_pilot.md for source, split, annotation and release contracts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import subprocess
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "retail_pilot"
VERSION = "retail-pilot-v1"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compact(value):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(value or "")))


def parse_number(value):
    """Only a whole numeric cell is accepted; do not silently parse footnotes."""
    value = compact(value).replace(",", "").replace("−", "-")
    if value.startswith("(") and value.endswith(")"):
        value = "-" + value[1:-1]
    if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", value):
        raise ValueError(f"Not a numeric cell: {value!r}")
    return Decimal(value)


def fetch(data):
    """Download original documents atomically; retain checksums and URLs."""
    sources = read_json(data / "sources.json")
    log_path = data / "downloads.json"
    old = {r["doc_id"]: r for r in read_json(log_path)} if log_path.exists() else {}
    log = []
    for source in sources:
        dest = data / "raw" / (source["doc_id"] + ".pdf")
        dest.parent.mkdir(parents=True, exist_ok=True)
        previous = old.get(source["doc_id"], {})
        if not (dest.exists() and previous.get("sha256") == sha256(dest)
                and previous.get("source_url") == source["source_url"]):
            temporary = dest.with_suffix(".download")
            subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error",
                            "--retry", "2", "--connect-timeout", "15", "--max-time", "90",
                            source["source_url"], "--output", str(temporary)], check=True)
            if not temporary.read_bytes().startswith(b"%PDF-"):
                raise ValueError(f"Non-PDF response for {source['doc_id']}")
            temporary.replace(dest)
            previous = {}
        row = {"doc_id": source["doc_id"], "source_url": source["source_url"],
               "path": str(dest.relative_to(data)), "sha256": sha256(dest),
               "bytes": dest.stat().st_size,
               "retrieved_at": previous.get("retrieved_at", datetime.now(timezone.utc).isoformat())}
        log.append(row)
        write_json(log_path, log)
        print(f"Downloaded/verified {source['doc_id']}: {row['bytes']:,} bytes", flush=True)


def extract(data, limit_pages=0):
    """Full-document text plus structured tables on the first 15 pages.

    Keep raw cell matrices and bounding boxes, never invent merged-cell content.
    Later financial-note tables remain raw PDF/page text in this pilot. A page
    with no extracted text is explicitly flagged as requiring OCR.
    """
    import pdfplumber
    from pypdf import PdfReader

    sources = read_json(data / "sources.json")
    for source in sources:
        dest = data / "parsed" / (source["doc_id"] + ".json")
        pdf_path = data / "raw" / (source["doc_id"] + ".pdf")
        digest = sha256(pdf_path)
        if dest.exists():
            previous = read_json(dest)
            if previous.get("sha256") == digest and previous.get("extractor_version") == VERSION:
                print(f"Cached {source['doc_id']}", flush=True)
                continue
        reader = PdfReader(pdf_path)
        cover = compact(reader.pages[0].extract_text())
        if compact(source["legal_name"]) not in cover or source["period"][:4] not in cover:
            raise ValueError(f"Cover does not match manifest: {source['doc_id']}")
        is_half = "半年度报告" in cover
        if is_half != (source["report_type"] == "half_year") or "年度报告摘要" in cover:
            raise ValueError(f"Report type does not match: {source['doc_id']}")
        pages = []
        with pdfplumber.open(pdf_path) as pdf:
            count = min(len(reader.pages), limit_pages) if limit_pages else len(reader.pages)
            for index in range(count):
                tables = []
                if index < 15:
                    page = pdf.pages[index]
                    text = page.extract_text() or ""
                    layout = page.extract_text(layout=True) or ""
                    for n, table in enumerate(page.find_tables()):
                        tables.append({"table_id": f"{source['doc_id']}_p{index+1:04d}_t{n:02d}",
                                       "bbox": list(table.bbox), "cells": table.extract()})
                    page.close()
                else:
                    text = reader.pages[index].extract_text() or ""
                    layout = None
                pages.append({"page": index + 1, "text": text, "layout_text": layout,
                              "tables": tables, "tables_extracted": index < 15,
                              "needs_ocr": len(compact(text)) < 20})
        write_json(dest, {**source, "sha256": digest, "extractor_version": VERSION,
                          "page_count": len(reader.pages), "extracted_page_count": len(pages),
                          "table_extraction_scope": "first_15_pages", "pages": pages})
        print(f"Parsed {source['doc_id']}: {len(pages)} pages, "
              f"{sum(len(p['tables']) for p in pages)} summary tables", flush=True)


def corpus_chunks(document, max_chars=1100):
    """Page-bounded chunks, with table matrices separate from page-text chunks."""
    result = []
    source = {k: document[k] for k in ["doc_id", "company_id", "company", "period",
                                      "period_start", "period_end", "published_date", "source_url", "split"]}
    for page in document["pages"]:
        paragraphs = page["text"].splitlines()
        buffers, buffer = [], ""
        for line in paragraphs:
            if len(buffer) + len(line) + 1 > max_chars and buffer:
                buffers.append(buffer)
                buffer = ""
            # Explicitly split unusually long lines rather than creating unbounded chunks.
            while len(line) > max_chars:
                if buffer:
                    buffers.append(buffer)
                    buffer = ""
                buffers.append(line[:max_chars])
                line = line[max_chars:]
            buffer += ("\n" if buffer else "") + line
        if buffer.strip():
            buffers.append(buffer)
        for i, text in enumerate(buffers):
            result.append({**source, "chunk_id": f"{document['doc_id']}_p{page['page']:04d}_c{i:02d}",
                           "title": document["title"], "pages": [page["page"]],
                           "kind": "page_text", "text": text,
                           "text_sha256": hashlib.sha256(compact(text).encode()).hexdigest()})
        for table in page["tables"]:
            text = "\n".join(" | ".join(str(c or "").replace("\n", " ") for c in row)
                             for row in table["cells"])
            result.append({**source, "chunk_id": table["table_id"], "title": document["title"],
                           "pages": [page["page"]], "kind": "table", "text": text,
                           "cells": table["cells"], "bbox": table["bbox"],
                           "header_status": "raw_matrix_requires_review"})
    return result


METRICS = {
    "revenue": ("营业收入", "CNY"),
    "net_profit": ("归属于上市公司股东的净利润", "CNY"),
    "operating_cash_flow": ("经营活动产生的现金流量净额", "CNY"),
    "total_assets": ("总资产", "CNY"),
    "net_assets": ("归属于上市公司股东的净资产", "CNY"),
    "eps": ("基本每股收益", "CNY/share"),
}


def metric_of(label):
    label = compact(label)
    # Labels must match exactly after removing unit suffixes. Avoid e.g. net
    # profit excluding nonrecurring items, or revenue after deductions.
    label = re.sub(r"\([^)]*\)", "", label)
    aliases = {"归属于上市公司股东的所有者权益": "net_assets"}
    if label in aliases:
        return aliases[label]
    return next((key for key, (name, _) in METRICS.items() if label == name), None)


def logical_rows(cells):
    """Collapse PDF grid artefacts while preserving every raw cell coordinate.

    A single logical label may be split over several rows; only append a
    following row containing ONE nonempty, nonnumeric cell. Period/header rows
    are never appended. Values always point back to the untouched cell matrix.
    """
    output = []
    for ri, row in enumerate(cells):
        values = [(ci, compact(c)) for ci, c in enumerate(row) if compact(c)]
        if not values:
            continue
        numbers = []
        for ci, c in values:
            try:
                parse_number(c)
                numbers.append((ci, c))
            except ValueError:
                pass
        if numbers:
            col = numbers[0][0]
            label = "".join(c for ci, c in values if ci < col)
            # Some PDF grids split just a later comparison-column value onto
            # its own physical row. It is not a new current-period metric.
            if (len(values) == 1 and not label and output and output[-1]["kind"] == "data"
                    and col > output[-1]["column_index"]):
                output[-1].setdefault("continuation_value_rows", []).append(ri)
                continue
            output.append({"kind": "data", "label": label, "row_index": ri,
                           "column_index": col, "raw_value": row[col],
                           "raw_row": row, "label_row_indices": [ri]})
        elif (len(values) == 1 and output and output[-1]["kind"] == "data"
              and not re.search(r"\d{4}|调整|本期|报告期|单位", values[0][1])):
            output[-1]["label"] += values[0][1]
            output[-1]["label_row_indices"].append(ri)
        else:
            output.append({"kind": "header", "values": [c for _, c in values],
                           "raw_row": row, "row_index": ri})
    return output


def extract_facts(document):
    """Conservative current-period summary-cell candidates, not gold labels.

    Read only the first data column of explicit current-year/current-period
    summary tables. Retain header evidence across adjacent-page continuations.
    Fail closed when the unit/header cannot be established.
    """
    found = {}
    header_context = None
    for page in document["pages"][:15]:
        for table in page["tables"]:
            rows = table["cells"]
            if not rows:
                continue
            logical = logical_rows(rows)
            current_marker = document["period"][:4] + "年"
            continuation = (header_context and page["page"] - header_context["page"] <= 1
                            and logical and logical[0]["kind"] == "data")
            if not continuation:
                header_context = None
            for logical_row in logical:
                if logical_row["kind"] == "header":
                    periods = [v for v in logical_row["values"]
                               if re.search(r"\d{4}年|本报告期|本期|季度", v)]
                    if periods:
                        first_period = periods[0]
                        valid = ((current_marker in first_period or "本报告期" in first_period or first_period == "本期")
                                 and "季度" not in first_period and "增减" not in first_period)
                        header_context = {"table_id": table["table_id"], "page": page["page"],
                            "rows": [logical_row["raw_row"]], "first_period": first_period} if valid else None
                    continue
                metric = metric_of(logical_row["label"])
                if not header_context or not metric or metric in found:
                    continue
                value = parse_number(logical_row["raw_value"])
                # Never assume a scale from magnitude. Use row unit or nearby
                # explicit currency/unit declaration, retain it for review.
                label = logical_row["label"]
                unit_evidence = label if "元" in label else ""
                if not unit_evidence:
                    contexts = [p["text"] for p in document["pages"]
                                if header_context["page"] - 1 <= p["page"] <= page["page"]]
                    declarations = re.findall(r"单位\s*[:：]\s*(?:人民币)?\s*(?:亿元|万元|千元|元)", "\n".join(contexts))
                    if declarations:
                        unit_evidence = compact(declarations[-1])
                if not unit_evidence:
                    continue
                scale = Decimal("100000000") if "亿元" in unit_evidence else Decimal("10000") if "万元" in unit_evidence else Decimal("1000") if "千元" in unit_evidence else Decimal(1)
                found[metric] = {
                    "fact_id": f"{document['doc_id']}__{metric}",
                    "doc_id": document["doc_id"], "company_id": document["company_id"],
                    "company": document["company"], "period": document["period"],
                    "split": document["split"], "metric": metric, "label": METRICS[metric][0],
                    "unit": METRICS[metric][1], "value": str(value * scale),
                    "raw_value": logical_row["raw_value"], "unit_evidence": unit_evidence,
                    "source_url": document["source_url"], "published_date": document["published_date"],
                    "page": page["page"], "table_id": table["table_id"],
                    "row_index": logical_row["row_index"], "column_index": logical_row["column_index"],
                    "raw_row": logical_row["raw_row"],
                    "label_row_indices": logical_row["label_row_indices"], "extracted_label": label,
                    "header_evidence": header_context.copy(),
                    "review_status": "pending", "extraction_method": "summary_table_first_current_column",
                }
    return list(found.values())


def prepare(data):
    docs = [read_json(data / "parsed" / (s["doc_id"] + ".json")) for s in read_json(data / "sources.json")]
    corpus = [c for d in docs for c in corpus_chunks(d)]
    facts = [f for d in docs for f in extract_facts(d)]
    write_json(data / "corpus.json", corpus)
    write_jsonl(data / "facts.jsonl", facts)
    print(f"Prepared {len(corpus)} chunks and {len(facts)} candidate facts")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["fetch", "extract", "prepare"])
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    {"fetch": fetch, "extract": extract, "prepare": prepare}[args.command](args.data_dir)


if __name__ == "__main__":
    main()
