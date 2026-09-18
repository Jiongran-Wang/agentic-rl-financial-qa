"""Versioned raw-source adapter: explicit FY/H1 headers and source precision.

No annotation inputs. Prior adapters remain unchanged.
"""
import json
import re
from decimal import Decimal
from retail_pilot import compact, logical_rows, parse_number

VERSION = "retail-presented-cells-v10"
UNIT = re.compile(r"单位[:：](?:人民币)?(亿元|万元|千元|元)")


def label_key(label):
    return re.sub(r"\([^)]*\)", "", compact(label))


def current_header(text, period):
    text = compact(text).replace('－', '-').replace('—', '-').replace('–', '-')
    # Reject comparative, quarterly and wrong-duration columns, even if they
    # also contain a current-period substring.
    if re.search(r'上年|上期|期初|年初|增减|季度|调整|同期', text):
        return False
    if re.fullmatch(r'本(?:报告)?期(?:末|数)?', text):
        return True
    if period.endswith('H1') and re.fullmatch(r'本(?:报告)?期\(1-6月\)', text):
        return True
    year = period[:4]
    if period.endswith('FY'):
        return bool(re.fullmatch(year + r'年(?:度|末|12月31日)?', text))
    return bool(re.fullmatch(year + r'年(?:上半年|半年度|6月30日|1-6月)', text))


def cell_catalog(sections, by_id):
    present = {s["chunk_id"] for s in sections}
    chunks = [by_id[i] for i in present]
    tables = sorted((c for c in chunks if c["kind"] == "table"), key=lambda c: (c["doc_id"], min(c["pages"]), c["chunk_id"]))
    output, header, previous_table = [], None, None
    for table in tables:
        page = min(table["pages"])
        logical = logical_rows(table["cells"])
        # A page can start with the tail of the preceding metric label.
        # Carry its header only when adjacent raw fragments join to a known
        # metric; do not infer from page proximity alone or alter raw cells.
        continuation = False
        if previous_table and logical and logical[0]['kind'] == 'header':
            prior = logical_rows(previous_table['cells'])
            tail = [compact(v) for v in logical[0]['raw_row'] if compact(v)]
            if (prior and prior[-1]['kind'] == 'data' and len(tail) == 1
                    and previous_table['doc_id'] == table['doc_id']
                    and page - min(previous_table['pages']) == 1):
                from retail_challenge_agent_v3 import metric
                prefix = prior[-1].get('label', '')
                continuation = metric(prefix) is None and metric(prefix+tail[0]) is not None
        if continuation:
            logical = logical[1:]
        previous_table = table
        if not (header and header["doc_id"] == table["doc_id"] and 0 <= page - header["page"] <= 1
                and logical and logical[0]["kind"] == "data"):
            header = None
        for row in logical:
            if row["kind"] == "header":
                nonempty = [(i, compact(v)) for i, v in enumerate(row['raw_row']) if compact(v)]
                if (header and header.get('current_column_limit') is not None and nonempty
                        and all(i > header['current_column_limit'] for i, _ in nonempty)
                        and all(re.search(r'增|减|调整|比上年|本年末', v) for _, v in nonempty)):
                    # Wrapped comparison-column labels at the far right do
                    # not replace the current-column header above them.
                    continue
                periods = [v for v in row["values"] if re.search(r"20\d{2}年|本报告期|本期|季度|上年|上期|期初|年初", v)]
                if periods:
                    first = periods[0]
                    valid = current_header(first, table["period"])
                    header = {"doc_id": table["doc_id"], "page": page, "chunk_id": table["chunk_id"],
                              "row": row["row_index"], "text": first} if valid else None
                    if header:
                        cols = [i for i, v in enumerate(row['raw_row']) if re.search(r'20\d{2}年|本报告期|本期|季度|上年|上期|期初|年初', compact(v))]
                        header['current_column_limit'] = (cols[0] + cols[1]) / 2 if len(cols) > 1 else None
                continue
            if not header or not row["label"]:
                continue
            if header.get('current_column_limit') is not None and row['column_index'] > header['current_column_limit']:
                # A missing current cell must not shift selection to the prior
                # column. Midpoints tolerate centered PDF header grid cells.
                continue
            label = compact(row["label"])
            # A period-end header supports balance stocks, not a six-month
            # flow. This also prevents inheriting a stock header into flows.
            stock = bool(re.search(r'资产|负债|权益', label))
            header_stock = bool(re.search(r'末|月\d+日', header['text']))
            if header_stock and not stock:
                continue
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
                           "source_unit": unit,
                           "resolution": str(Decimal(10) ** value.as_tuple().exponent * Decimal(scale)),
                           "unit_source": {"chunk_id": unit_id, "quote": unit_quote}})
    return output
