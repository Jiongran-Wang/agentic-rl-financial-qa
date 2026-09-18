"""Normalize a missing slots wrapper only when all question-derived keys match.

Never rename, drop, reorder or infer operands. Original v3 code remains frozen.
"""
import json
import re

from retail_planner_v3 import parse_plan

VERSION = "retail-slots-wrapper-compat-v1"


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key: " + key)
        result[key] = value
    return result


def parse_compatible_plan(raw, expected_slots):
    text = re.sub(r"^<think>.*?</think>\s*", "", raw.strip(), flags=re.S)
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    obj = json.loads(text, object_pairs_hook=unique_object)
    normalization = "none"
    expected = set(expected_slots)
    if isinstance(obj, dict) and expected and set(obj) == expected:
        obj = {"slots": obj}
        normalization = "added_slots_wrapper"
    plan = parse_plan(json.dumps(obj, ensure_ascii=False))
    if "slots" in plan and set(plan["slots"]) != expected:
        raise ValueError("Plan must fill exactly the question-derived slots")
    return plan, normalization
