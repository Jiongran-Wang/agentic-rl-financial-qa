"""Conservative answer-content scoring, separate from strict output formatting.

Scoring may read references. Nothing in this module is passed to inference.
Unsupported paraphrases remain false with an explicit reason, rather than
accepting any occurrence of the expected number in an arbitrary answer.
"""
from decimal import Decimal, ROUND_HALF_UP
import re
import unicodedata

VERSION = "retail-content-v2"
QUANTITY = re.compile(r"([+-]?\d+(?:\.\d+)?)(亿元|万元|千元|元/股|元|%)")


def clean(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text)).replace(",", "").replace("−", "-")


def quantity(match):
    number, unit = match.groups()
    scale = {"亿元": 100000000, "万元": 10000, "千元": 1000}.get(unit, 1)
    dimension = "元" if unit in {"亿元", "万元", "千元"} else unit
    return ((Decimal(number) * scale).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP), dimension)


def canonical_citation(value):
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1].strip()
    return value


def content_score(question, prediction):
    if prediction is None:
        return {"answer_content_correct": False, "content_reason": "invalid_response"}
    def result(correct, reason):
        return {"answer_content_correct": bool(correct), "content_reason": reason}
    if not question["answerable"]:
        return result(prediction["abstain"], "abstention_decision_only")
    if prediction["abstain"]:
        return result(False, "abstained_on_answerable")
    actual = clean(prediction["answer"]).rstrip("。.")
    expected = clean(question["answer"]).rstrip("。.")
    if re.search(r"不是|不为|不等于|错误|无法|不能|不确定|可能|或者|大约|约为|至少|至多|[≈<>~]", actual):
        return result(False, "negation_uncertainty_or_bound")
    kind = question["question_type"]
    if kind == "comparison":
        # This is company-selection accuracy, not verification of an extra claim.
        if actual == expected:
            return result(True, "company_selection")
        match = re.fullmatch(r"([^;:]+)[;:]([+-]?\d+(?:\.\d+)?(?:亿元|万元|千元|元)?)", actual)
        return result(bool(match and match[1] == expected), "company_selection_with_optional_amount")
    gold = list(QUANTITY.finditer(expected))
    predicted = list(QUANTITY.finditer(actual))
    if len(gold) != 1 or len(predicted) != 1:
        return result(False, "requires_one_explicit_quantity_with_unit")
    number = predicted[0]
    if actual[number.end():]:
        return result(False, "unsupported_answer_suffix")
    prefix = actual[:number.start()]
    if kind == "dependent_lookup":
        company = expected[:gold[0].start()].rstrip(";:")
        if prefix.rstrip(";:") != company:
            return result(False, "wrong_or_unsupported_company")
    elif prefix:
        # Allow a company-only prefix, or a declarative subject appearing in the
        # question. Requiring a whole subject prevents acceptance of a wrong
        # metric/year merely because the expected number occurs in the output.
        subject = re.sub(r"(?:是|为|等于)$", "", prefix).rstrip(";:")
        qtext = clean(question["question"])
        if not subject or subject not in qtext:
            return result(False, "subject_not_in_question")
    return result(quantity(number) == quantity(gold[0]), "quantity_sign_unit_and_two_decimals")


def score_v2(question, prediction, presented_ids):
    result = content_score(question, prediction)
    citations = prediction["citations"] if prediction else []
    canonical = list(dict.fromkeys(canonical_citation(c) for c in citations))
    allowed = set(presented_ids)
    result.update({"canonical_citations": canonical,
                   "canonical_invalid_citations": [c for c in canonical if c not in allowed],
                   "canonical_citation_validity": sum(c in allowed for c in canonical) / len(canonical) if canonical else None})
    return result
