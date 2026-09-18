# Data and source provenance

The bundled JSON inputs derive from public Chinese retail-company annual
reports. The final evaluation contains 64 questions about four 2024 reports,
with 1,032 retrieval chunks. Training, development and final evaluation use
distinct question sets; the four final-evaluation companies were absent from
project training and development.

## Final evaluation sources

| Company | Report | Publication date | Source |
|---|---|---|---|
| 合百集团 | 2024FY | 2025-04-25 | [Annual report](https://static.cninfo.com.cn/finalpage/2025-04-25/1223267674.PDF) |
| 宁波中百 | 2024FY | 2025-04-15 | [Annual report](https://dataclouds.cninfo.com.cn/shgonggao/2025/2025-04-15/39359fc5192b11f0b549fa163e957f7a.pdf) |
| 友好集团 | 2024FY | 2025-04-25 | [Annual report](https://static.cninfo.com.cn/finalpage/2025-04-25/1223283725.PDF) |
| 南宁百货 | 2024FY | 2025-03-28 | [Annual report](https://file.finance.sina.com.cn/211.154.219.97%3A9494/MRGG/CNSESH_STOCK/2025/2025-3/2025-03-28/10819307.PDF) |

The 20 current-year reference figures come from the main accounting-indicator
pages: 合百集团 page 7, 宁波中百 page 5, 友好集团 page 5 and 南宁百货 page 6.
The original review visually checked these pages and used separate Decimal
arithmetic to verify all derived answers. Review was by the same project
assistant, not an independent human panel.

## Bundle layout after unpacking

- `data/retail_grpo_iter_v1/`: the 26 training tasks, training-only proofs,
  original corpus, 27 development tasks and their scorer inputs.
- `results/retail_reward_v2_sft_v1/`: 204 frozen SFT action samples and their
  validation contract. This path is retained for compatibility with the trainer.
- `data/retail_grpo_heldout_v2/`: public questions, corpus and the fixed
  three-policy collection contract.
- `data/retail_policy_eval_v2/`: source catalog, reference figures and private
  scorer inputs. “Private” means withheld from policy collection during the
  experiment; these evaluation labels are now included for public replay.
- Earlier source catalogs document the training/development company separation.

The bundle includes parsed report excerpts and tabular data, not raw PDFs.
Source documents and model weights retain their respective rights and terms;
this publication does not assign a new license to third-party content.
The repository does not include the vendored LLaMA-Factory or verl projects.

`artifacts/input-manifest.json` pins every unpacked file. Historical manifests
retain their original status strings and hashes, including entries for raw
PDFs and rendered pages that are not distributed here. Use the public replay
commands for the scope supported by this release; the original archival
scoring command also expects those omitted files and original logs.

## Evaluation boundaries

Normal questions use a June 30, 2025 publication cutoff; four unavailable-scope
questions use February 28, 2025, before the included reports were published.
The six task families are lookup, ratio, difference, comparison, comparison
followed by lookup, and unavailable-at-cutoff. Questions are templated and
share four reports, so question count must not be mistaken for independent
company-level coverage. Base-model pretraining exposure is unknown.

The test is now public and inspected. Do not tune on it and describe the same
questions as fresh held-out evidence in a later experiment.
