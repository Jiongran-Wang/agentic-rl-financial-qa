"""Dependency-free lexical baseline for the isolated retail pilot.

Only corpus records enter this module: no reference answers or fact annotations.
Dates are applied before scoring, including collection statistics and expansion.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date
import math
import re
import unicodedata

VERSION = "retail-bm25-v1"


def tokens(text):
    text = re.sub(r"\s+", "", unicodedata.normalize("NFKC", text)).lower()
    result = []
    for run in re.findall(r"[\u4e00-\u9fff]+|[a-z]+|\d+(?:\.\d+)?", text):
        if "\u4e00" <= run[0] <= "\u9fff":
            result.extend(run)
            result.extend(run[i:i + 2] for i in range(len(run) - 1))
        else:
            result.append(run)
    return result


def source_text(chunk):
    period = chunk["period"][:4] + ("年上半年" if chunk["period"].endswith("H1") else "年度")
    return f"{chunk['company']} {period} {chunk['title']}\n{chunk['text']}"


def render_chunk(chunk):
    # Content is data; the model system prompt explicitly says not to follow it.
    return (f"[{chunk['chunk_id']}] 公司={chunk['company']} 报告期={chunk['period']} "
            f"发布日期={chunk['published_date']} PDF页={','.join(map(str, chunk['pages']))} "
            f"类型={chunk['kind']}\n来源={chunk['source_url']}\n{chunk['text']}")


class RetailSearch:
    def __init__(self, corpus):
        self.corpus = list(corpus)
        self.by_id = {c["chunk_id"]: c for c in self.corpus}
        if len(self.by_id) != len(self.corpus):
            raise ValueError("Duplicate corpus chunk IDs")
        self.by_page = defaultdict(list)
        for c in self.corpus:
            date.fromisoformat(c["published_date"])
            for page in c["pages"]:
                self.by_page[c["doc_id"], page].append(c)
        self.counts = {c["chunk_id"]: Counter(tokens(source_text(c))) for c in self.corpus}
        self._indices = {}

    def _index(self, as_of):
        if as_of is not None:
            date.fromisoformat(as_of)
        if as_of not in self._indices:
            eligible = [c for c in self.corpus if as_of is None or c["published_date"] <= as_of]
            df = Counter()
            for c in eligible:
                df.update(self.counts[c["chunk_id"]].keys())
            avg = sum(sum(self.counts[c["chunk_id"]].values()) for c in eligible) / max(1, len(eligible))
            self._indices[as_of] = (eligible, df, avg)
        return self._indices[as_of]

    def search(self, question, as_of=None, top_k=6):
        if top_k < 1:
            raise ValueError("top_k must be positive")
        eligible, df, avg = self._index(as_of)
        query = set(tokens(question))
        scored = []
        for c in eligible:
            counts = self.counts[c["chunk_id"]]
            length = sum(counts.values())
            score = 0.0
            for term in query.intersection(counts):
                freq = counts[term]
                idf = math.log(1 + (len(eligible) - df[term] + 0.5) / (df[term] + 0.5))
                score += idf * freq * 2.5 / (freq + 1.5 * (0.25 + 0.75 * length / max(avg, 1)))
            if score > 0:
                scored.append({"chunk_id": c["chunk_id"], "score": score})
        return sorted(scored, key=lambda r: (-r["score"], r["chunk_id"]))[:top_k]

    def context(self, hits, as_of=None, max_chars=18000, expand_tables=True):
        """Keep ranked chunks whole; append table page/continuation context.

        Expansion uses only geometry/kind/header text from the corpus. It is a
        provenance aid, not a claim that every malformed PDF header was repaired.
        Every included source remains independently citable. Nothing is sliced.
        """
        if max_chars < 1:
            raise ValueError("max_chars must be positive")
        candidates, seen = [], set()

        def add(c, reason):
            if c["chunk_id"] not in seen and (as_of is None or c["published_date"] <= as_of):
                candidates.append((c, reason))
                seen.add(c["chunk_id"])

        for hit in hits:
            add(self.by_id[hit["chunk_id"]], "ranked")
        if expand_tables:
            for hit in hits:
                c = self.by_id[hit["chunk_id"]]
                if c["kind"] != "table" or (as_of and c["published_date"] > as_of):
                    continue
                page = min(c["pages"])
                # Same page text retains nearby unit declarations and headings.
                for neighbor in self.by_page[c["doc_id"], page]:
                    if neighbor["kind"] == "page_text":
                        add(neighbor, "table_page_context")
                first_rows = "\n".join(c["text"].splitlines()[:2])
                if not re.search(r"20\d{2}\s*年|本报告期|本期|本年度", first_rows):
                    # A continuation may need the preceding page's header/unit.
                    for neighbor in self.by_page[c["doc_id"], page - 1]:
                        add(neighbor, "possible_table_continuation_context")
        included, skipped, size = [], [], 0
        for chunk, reason in candidates:
            block = render_chunk(chunk)
            cost = len(block) + (2 if included else 0)
            if size + cost > max_chars:
                skipped.append(chunk["chunk_id"])
                continue
            included.append({"chunk_id": chunk["chunk_id"], "reason": reason, "text": block})
            size += cost
        return {"sections": included, "skipped_chunk_ids": skipped, "chars": size,
                "text": "\n\n".join(c["text"] for c in included)}
