import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "retrieval"))
from retail_scoring import content_score, score_v2
from retail_scoped import ScopedRetailSearch, query_scope
from retail_search import RetailSearch
from retail_evidence import add_catalog, cell_catalog
from retail_calculator import execute, parse_plan, build_plan_messages


def fixture(company="甲超市", doc="a", year="2024", revenue="100", cash="12", unit="元"):
    meta = {"doc_id": doc, "company": company, "period": year + "FY", "published_date": str(int(year)+1) + "-04-01",
            "pages": [1], "title": company + year + "年年度报告", "source_url": "https://example.org/report"}
    cells = [["指标", year + "年", str(int(year)-1)+"年"], ["营业收入", revenue, "999"],
             ["总资产", "800", "900"], ["归属于上市公司股东的净资产", "200", "180"],
             ["经营活动产生的现金流量净额", cash, "10"]]
    return [{**meta, "chunk_id": doc + "_table", "kind": "table", "cells": cells,
             "text": "\n".join(" | ".join(row) for row in cells)},
            {**meta, "chunk_id": doc + "_page", "kind": "page_text", "text": "单位：" + unit}]


def context_for(chunks):
    search = RetailSearch(chunks)
    context = search.context([{"chunk_id": c["chunk_id"]} for c in chunks])
    return add_catalog(context, search.by_id, max_chars=100000), search.by_id


def prediction(answer):
    return {"answer": answer, "abstain": False, "citations": []}


class ScoringTests(unittest.TestCase):
    def test_sentence_is_accepted_but_wrong_subject_unit_sign_and_alternatives_are_not(self):
        q = {"question": "甲超市2024年度的净利润是多少？", "question_type": "lookup", "answerable": True, "answer": "-123400.00元"}
        self.assertTrue(content_score(q, prediction("甲超市2024年度的净利润为-12.34万元。"))["answer_content_correct"])
        for bad in ["乙超市2024年度的净利润为-123400元", "甲超市2023年度的净利润为-123400元",
                    "甲超市2024年度的营业收入为-123400元", "不是-123400元", "-123400元或-12元",
                    "123400元", "-123400元/股", "-123400", "可能为-123400元"]:
            self.assertFalse(content_score(q, prediction(bad))["answer_content_correct"], bad)

    def test_precision_is_not_relaxed_to_make_close_answers_correct(self):
        q = {"question": "甲超市2024年度资产比例是多少？", "question_type": "ratio", "answerable": True, "answer": "17.86%"}
        self.assertFalse(content_score(q, prediction("17.84%"))["answer_content_correct"])
        q.update(question_type="lookup", answer="100.35元")
        self.assertFalse(content_score(q, prediction("100元"))["answer_content_correct"])

    def test_comparison_is_selection_only_and_dependent_lookup_checks_both_parts(self):
        q = {"question": "甲超市与乙超市哪家更高？", "question_type": "comparison", "answerable": True, "answer": "甲超市"}
        self.assertTrue(content_score(q, prediction("甲超市；100.00"))["answer_content_correct"])
        self.assertFalse(content_score(q, prediction("乙超市；100.00"))["answer_content_correct"])
        q.update(question_type="dependent_lookup", answer="甲超市；100.00元")
        self.assertTrue(content_score(q, prediction("甲超市；0.01万元"))["answer_content_correct"])
        self.assertFalse(content_score(q, prediction("乙超市；100.00元"))["answer_content_correct"])

    def test_citation_canonicalization_still_rejects_unpresented_ids(self):
        q = {"answerable": False}
        p = {"answer": "不确定", "abstain": True, "citations": ["[source]", "[invented]"]}
        scored = score_v2(q, p, ["source"])
        self.assertEqual(scored["canonical_invalid_citations"], ["invented"])
        self.assertEqual(scored["canonical_citation_validity"], 0.5)


class ScopeTests(unittest.TestCase):
    def test_cutoff_year_is_not_report_year_and_unavailable_scope_does_not_fall_back(self):
        corpus = fixture()
        q = "截至2025-03-31，甲超市2024年度的营业收入是多少？"
        self.assertEqual(query_scope(q, corpus), {"companies": ["甲超市"], "periods": ["2024FY"]})
        search = ScopedRetailSearch(corpus)
        self.assertEqual(search.search(q, "2025-03-31"), [])
        self.assertEqual(search.search("甲超市2025年度的营业收入是多少？"), [])
        self.assertTrue(search.search(q, "2025-04-01"))

    def test_two_companies_receive_slots_even_with_one_dominant_document(self):
        corpus = fixture() + fixture("乙超市", "b", revenue="200")
        search = ScopedRetailSearch(corpus)
        ids = [h["chunk_id"] for h in search.search("2024年度甲超市与乙超市哪家的营业收入更高？", top_k=2)]
        self.assertEqual({search.by_id[i]["company"] for i in ids}, {"甲超市", "乙超市"})


class CalculatorTests(unittest.TestCase):
    def setUp(self):
        self.corpus = fixture() + fixture("乙超市", "b", revenue="200", cash="30")
        self.context, self.by_id = context_for(self.corpus)
        self.scope = {"companies": ["甲超市", "乙超市"], "periods": ["2024FY"]}

    def run_plan(self, op, ids, q, scope=None, context=None, as_of=None):
        return execute({"op": op, "inputs": ids}, q, as_of, scope or self.scope,
                       context or self.context, self.by_id)[0]

    def test_difference_preserves_negative_sign_and_rejects_reversed_order(self):
        q = "2024年度甲超市的营业收入减去乙超市是多少元？"
        self.assertEqual(self.run_plan("difference", ["a_table:r1:c1", "b_table:r1:c1"], q)["answer"], "-100.00元")
        with self.assertRaises(ValueError):
            self.run_plan("difference", ["b_table:r1:c1", "a_table:r1:c1"], q)

    def test_ratio_reads_cells_and_does_not_trust_catalog_value(self):
        scope = {"companies": ["甲超市"], "periods": ["2024FY"]}
        q = "甲超市2024年度归属于上市公司股东的净资产占总资产的比例是多少？"
        tampered = copy.deepcopy(self.context)
        for cell in tampered["cell_catalog"]:
            cell["value"] = "999999"
        pred = self.run_plan("ratio", ["a_table:r3:c1", "a_table:r2:c1"], q, scope, tampered)
        self.assertEqual(pred["answer"], "25.00%")
        with self.assertRaises(ValueError):
            self.run_plan("ratio", ["a_table:r2:c1", "a_table:r3:c1"], q, scope)

    def test_unknown_prior_period_and_future_cells_are_rejected(self):
        q = "2024年度甲超市的营业收入减去乙超市是多少元？"
        for bad in ["invented", "a_table:r1:c2"]:
            with self.assertRaises(ValueError):
                self.run_plan("difference", [bad, "b_table:r1:c1"], q)
        with self.assertRaises(ValueError):
            self.run_plan("difference", ["a_table:r1:c1", "b_table:r1:c1"], q, as_of="2025-03-31")

    def test_dependent_lookup_selects_then_reads_the_followup_metric(self):
        q = "2024年度甲超市与乙超市中营业收入更高的是哪家？该公司经营活动产生的现金流量净额是多少？"
        pred = self.run_plan("select_then_lookup", ["a_table:r1:c1", "b_table:r1:c1", "a_table:r4:c1", "b_table:r4:c1"], q)
        self.assertEqual(pred["answer"], "乙超市；30.00元")
        with self.assertRaises(ValueError):
            self.run_plan("select_then_lookup", ["a_table:r1:c1", "b_table:r1:c1", "a_table:r2:c1", "b_table:r2:c1"], q)

    def test_missing_company_wrong_operation_and_raw_number_inputs_fail(self):
        q = "2024年度甲超市与乙超市哪家营业收入更高？"
        with self.assertRaises(ValueError):
            self.run_plan("lookup", ["a_table:r1:c1"], q)
        with self.assertRaises(ValueError):
            parse_plan('{"op":"ratio","inputs":[100,200]}')
        with self.assertRaises(ValueError):
            parse_plan('{"op":"ratio","inputs":[],"answer":"25%"}')

    def test_scale_conversion_and_missing_or_conflicting_units(self):
        context, by_id = context_for(fixture(unit="万元"))
        pred, trace = execute({"op": "lookup", "inputs": ["a_table:r1:c1"]}, "甲超市2024年度营业收入是多少？",
            None, {"companies": ["甲超市"], "periods": ["2024FY"]}, context, by_id)
        self.assertEqual(pred["answer"], "1000000.00元")
        corpus = fixture()
        corpus[1]["text"] = "没有单位"
        self.assertEqual(context_for(corpus)[0]["cell_catalog"], [])
        corpus[1]["text"] = "单位：元\n单位：万元"
        self.assertEqual(context_for(corpus)[0]["cell_catalog"], [])

    def test_quarterly_headers_are_not_annual_current_cells(self):
        corpus = fixture()
        corpus[0]["cells"][0][1] = "2024年第一季度"
        self.assertEqual(context_for(corpus)[0]["cell_catalog"], [])

    def test_zero_denominator_and_tie_are_explicit_failures(self):
        self.by_id["a_table"]["cells"][2][1] = "0"
        q = "甲超市2024年度归属于上市公司股东的净资产占总资产的比例是多少？"
        with self.assertRaises(ValueError):
            self.run_plan("ratio", ["a_table:r3:c1", "a_table:r2:c1"], q, {"companies": ["甲超市"], "periods": ["2024FY"]})
        self.by_id["b_table"]["cells"][1][1] = "100"
        q = "2024年度甲超市与乙超市中营业收入更高的是哪家？该公司经营活动产生的现金流量净额是多少？"
        with self.assertRaises(ValueError):
            self.run_plan("select_then_lookup", ["a_table:r1:c1", "b_table:r1:c1", "a_table:r4:c1", "b_table:r4:c1"], q)

    def test_prompt_interface_has_no_reference_fields(self):
        messages = build_plan_messages("甲超市营业收入是多少？", None, self.context)
        fields = json.loads(messages[1]["content"])
        self.assertEqual(set(fields), {"question", "as_of", "evidence"})
        self.assertNotIn("reference_answer", json.dumps(fields))

    def test_cross_year_difference_uses_new_minus_old(self):
        corpus = fixture(doc="old", year="2023", revenue="500") + fixture(doc="new", year="2024", revenue="300")
        context, by_id = context_for(corpus)
        scope = {"companies": ["甲超市"], "periods": ["2023FY", "2024FY"]}
        q = "甲超市2023年度到2024年度营业收入变化，按后一期减前一期计算"
        p, _ = execute({"op": "difference", "inputs": ["new_table:r1:c1", "old_table:r1:c1"]}, q, None, scope, context, by_id)
        self.assertEqual(p["answer"], "-200.00元")
        with self.assertRaises(ValueError):
            execute({"op": "difference", "inputs": ["old_table:r1:c1", "new_table:r1:c1"]}, q, None, scope, context, by_id)

    def test_continuation_requires_a_presented_header_and_unit(self):
        corpus = fixture()
        table, page = corpus
        header = {**table, "chunk_id": "a_header", "cells": [table["cells"][0]], "text": "指标 | 2024年 | 2023年"}
        table = {**table, "pages": [2], "cells": table["cells"][1:], "text": "营业收入 | 100 | 999"}
        context, by_id = context_for([header, table, page])
        self.assertTrue(context["cell_catalog"])
        sections_without_header = [s for s in context["sections"] if s["chunk_id"] != "a_header"]
        self.assertEqual(cell_catalog(sections_without_header, by_id), [])

    def test_cell_omitted_from_catalog_cannot_be_called_even_if_raw_table_is_present(self):
        context = copy.deepcopy(self.context)
        context["cell_catalog"] = [c for c in context["cell_catalog"] if c["cell_id"] != "a_table:r1:c1"]
        with self.assertRaises(ValueError):
            self.run_plan("difference", ["a_table:r1:c1", "b_table:r1:c1"],
                          "2024年度甲超市的营业收入减去乙超市是多少元？", context=context)


if __name__ == "__main__":
    unittest.main()
