"""Query-derived company/period scopes with balanced lexical retrieval.

No QA annotations, fact IDs or expected document IDs are accepted here.
"""
from collections import Counter
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from retail_pilot import logical_rows
from retail_evidence import label_key
from retail_search import RetailSearch, source_text, tokens

VERSION = "retail-scoped-bm25-v2"


def query_scope(question, corpus):
    companies = sorted({c["company"] for c in corpus if c["company"] in question}, key=question.find)
    # Dates such as 2025-04-24 are cutoff dates, not report periods.
    periods = []
    for m in re.finditer(r"(20\d{2})(年度|年上半年|年半年度)", question):
        period = m[1] + ("FY" if m[2] == "年度" else "H1")
        if period not in periods:
            periods.append(period)
    return {"companies": companies, "periods": periods}


class ScopedRetailSearch(RetailSearch):
    def __init__(self, corpus):
        super().__init__(corpus)
        self.groups = {}
        for c in corpus:
            self.groups.setdefault((c["company"], c["period"]), []).append(c)
        self.indices = {}
        self.labels = {}
        for key, records in self.groups.items():
            index = RetailSearch(records)
            # Rejoin raw PDF label fragments for search, retaining original
            # corpus text and cell coordinates in the returned context.
            for c in records:
                if c["kind"] == "table":
                    normalized = [label_key(r.get("label", "")) for r in logical_rows(c["cells"]) if r.get("label")]
                    self.labels[c["chunk_id"]] = normalized
                    labels = " ".join(normalized)
                    index.counts[c["chunk_id"]] = Counter(tokens(source_text(c) + "\n" + labels))
            self.indices[key] = index

    def search(self, question, as_of=None, top_k=6):
        if top_k < 1:
            raise ValueError("top_k must be positive")
        scope = query_scope(question, self.corpus)
        if not scope["companies"] or not scope["periods"]:
            # Unrecognized queries retain an explicit generic fallback.
            return super().search(question, as_of, top_k)
        queues = []
        for company in scope["companies"]:
            for period in scope["periods"]:
                index = self.indices.get((company, period))
                hits = index.search(question, as_of, len(index.corpus)) if index else []
                # Exact normalized row labels are stronger field matches than
                # a metric mentioned in narrative prose. The labels come from
                # every raw table, never the six annotated pilot metric IDs.
                for hit in hits:
                    hit["matched_row_labels"] = sorted({label for label in self.labels.get(hit["chunk_id"], []) if label in question})
                hits.sort(key=lambda h: (-len(h["matched_row_labels"]), -h["score"], h["chunk_id"]))
                hits = hits[:top_k]
                queues.append(hits)
        # Never fill an unavailable requested report with another company/year.
        # Round-robin order makes both sides visible before one side gets more.
        result = []
        for rank in range(top_k):
            for queue in queues:
                if rank < len(queue):
                    result.append({**queue[rank], "scope_balanced": True})
                    if len(result) == top_k:
                        return result
        return result
