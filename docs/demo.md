# Recorded tool-use example

This is a replay of a saved iterative-GRPO response sequence from evaluation
repeat 1. Tool outputs are recomputed from the bundled corpus; no new LLM
inference occurs. The policy searches and reads two source reports, calculates
the asset difference, and supplies the resulting proof in its final answer.

```bash
python3 scripts/unpack_inputs.py
python3 scripts/demo.py
```

Expected transcript:

```text
RECORDED POLICY REPLAY — no live model inference

Question: 按两家公司2024年年度报告本期列，计算合百集团的总资产减去宁波中百同一指标的差额；资产指标取2024年末数。用元表示，保留两位小数和正负号。

1. SEARCH
{"query": "合百集团2024年度 主要会计数据 总资产", "top_k": 8, "type": "search"}

2. READ
{"chunk_ids": ["hfbh_2024fy_p0007_t02", "hfbh_2024fy_p0007_c00", "hfbh_2024fy_p0008_c00", "hfbh_2024fy_p0218_c00"], "type": "read"}

3. SEARCH
{"query": "宁波中百2024年度 主要会计数据 总资产", "top_k": 8, "type": "search"}

4. READ
{"chunk_ids": ["nbzb_2024fy_p0005_t04", "nbzb_2024fy_p0005_c00", "nbzb_2024fy_p0006_c00", "nbzb_2024fy_p0012_t01"], "type": "read"}

5. CALCULATE
{"expressions": [{"args": [{"cell_id": "hfbh_2024fy_p0007_t02:r9:c1"}, {"cell_id": "nbzb_2024fy_p0005_t04:r7:c1"}], "op": "subtract"}], "type": "calculate"}
Source-backed calculation: {"calc_5_0": {"expression": {"args": [{"cell_id": "hfbh_2024fy_p0007_t02:r9:c1"}, {"cell_id": "nbzb_2024fy_p0005_t04:r7:c1"}], "op": "subtract"}, "value": "12304705336.12", "unit": "元", "citations": ["hfbh_2024fy_p0007_t02", "nbzb_2024fy_p0005_c00", "nbzb_2024fy_p0005_t04"]}}

6. FINAL
{"action": "answer", "answer": "12304705336.12元", "calculation_ids": ["calc_5_0"], "citations": ["hfbh_2024fy_p0007_t02", "nbzb_2024fy_p0005_c00", "nbzb_2024fy_p0005_t04"], "type": "final"}

Answer: 12304705336.12元
Verification: verified_numeric_equivalence
```
