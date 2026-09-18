"""Build a current-column cell catalog solely from presented corpus sources.

This restricted PDF-table adapter covers explicit annual/current-period headers
and unambiguous currency declarations. It does not read the pilot fact labels.
"""
import json
import re
from decimal import Decimal
from retail_pilot import compact, logical_rows, parse_number

VERSION = "retail-presented-cells-v2"
UNIT = re.compile(r"单位[:：](?:人民币)?(亿元|万元|千元|元)")


def label_key(label):
    return re.sub(r"\([^)]*\)", "", compact(label))


def cell_catalog(sections, by_id):
    present = {s["chunk_id"] for s in sections}
    chunks = [by_id[i] for i in present]
    tables = sorted((c for c in chunks if c["kind"] == "table"), key=lambda c: (c["doc_id"], min(c["pages"]), c["chunk_id"]))
    output, header = [], None
    for table in tables:
        page = min(table["pages"])
        logical = logical_rows(table["cells"])
        if not (header and header["doc_id"] == table["doc_id"] and 0 <= page - header["page"] <= 1
                and logical and logical[0]["kind"] == "data"):
            header = None
        for row in logical:
            if row["kind"] == "header":
                periods = [v for v in row["values"] if re.search(r"20\d{2}年|本报告期|本期|季度", v)]
                if periods:
                    first = periods[0]
                    valid = (table["period"][:4] + "年" in first or first in {"本期", "本报告期"}) and not re.search(r"季度|增减", first)
                    header = {"doc_id": table["doc_id"], "page": page, "chunk_id": table["chunk_id"],
                              "row": row["row_index"], "text": first} if valid else None
                continue
            if not header or not row["label"]:
                continue
            label = compact(row["label"])
            unit_match = re.search(r"亿元|万元|千元|元/股|元", label)
            if unit_match:
                unit, unit_id, unit_quote = unit_match[0], table["chunk_id"], label
            else:
                declarations = []
                for c in chunks:
                    if c["kind"] == "page_text" and c["doc_id"] == table["doc_id"] and header["page"] - 1 <= min(c["pages"]) <= page:
                        for m in UNIT.finditer(compact(c["text"])):
                            declarations.append((m[1], c["chunk_id"], m[0]))
                if len({d[0] for d in declarations}) != 1:
                    continue
                unit, unit_id, unit_quote = sorted(declarations)[0]
            value = parse_number(row["raw_value"])
            scale = {"亿元": 100000000, "万元": 10000, "千元": 1000}.get(unit, 1)
            output.append({"cell_id": f"{table['chunk_id']}:r{row['row_index']}:c{row['column_index']}",
                           "chunk_id": table["chunk_id"], "row": row["row_index"], "column": row["column_index"],
                           "company": table["company"], "period": table["period"], "label": label_key(label),
                           "value": str(value * Decimal(scale)), "unit": "元/股" if unit == "元/股" else "元",
                           "raw_value": row["raw_value"], "header": header.copy(),
                           "unit_source": {"chunk_id": unit_id, "quote": unit_quote}})
    return output


def add_catalog(context, by_id, max_chars=18000, question=""):
    candidates = cell_catalog(context["sections"], by_id)
    # Spend the catalog budget on explicitly requested labels and one copy of
    # each company/period/metric/value before repeated disclosures. This uses
    # question text and raw data only, not annotated facts or reference values.
    seen = set()
    prioritized = []
    for c in candidates:
        key = (c["company"], c["period"], c["label"], c["value"], c["unit"])
        prioritized.append((c["label"] not in question, key in seen, c))
        seen.add(key)
    candidates = [c for _, _, c in sorted(prioritized, key=lambda x: (x[0], x[1]))]
    suffix = "\n\n可调用计算器的当期原始单元格（仅来自上方证据；value已按明确单位换算）：\n"
    catalog, omitted = [], []
    for c in candidates:
        line = json.dumps(c, ensure_ascii=False, separators=(",", ":")) + "\n"
        if len(context["text"]) + len(suffix) + len(line) > max_chars:
            omitted.append(c["cell_id"])
            continue
        suffix += line
        catalog.append(c)
    text = context["text"] + suffix if catalog else context["text"]
    return {**context, "text": text, "chars": len(text), "cell_catalog": catalog, "omitted_cell_ids": omitted}
