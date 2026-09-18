# Held-out results and limitations

## Primary comparison

| Policy | Repeat 1 | Repeat 2 | Per-question average |
|---|---:|---:|---:|
| SFT | 30/64 | 28/64 | 45.3125% |
| Original GRPO | 35/64 | 34/64 | 53.90625% |
| Iterative GRPO | 43/64 | 42/64 | 66.40625% |

Iterative GRPO improves by **21.09375 percentage points over SFT** and
**12.5 points over original GRPO**. Its paired wins/losses against SFT are
13/0 in repeat 1 and 15/1 in repeat 2. Against original GRPO they are 8/0 and
9/1. The only per-repeat regression is a prompt-budget failure on a comparison.

There are 64 unique questions from four new project companies. All 384
episodes are eligible under the frozen scoring rules, with no missing attempts
or infrastructure errors. The 128 attempts per policy are repeated measurements
on those same questions. No significance claim is made.

## Efficiency

| Metric, both repeats | SFT | Original GRPO | Iterative GRPO | Change versus SFT |
|---|---:|---:|---:|---:|
| Invalid actions | 341 | 266 | 153 | −55.13% |
| Repeated actions | 412 | 343 | 195 | −52.67% |
| Tool calls | 951 | 887 | 743 | −21.87% |
| Model requests | 1,035 | 978 | 846 | −18.26% |
| Processed tokens | 8,821,482 | 8,791,191 | 7,804,391 | −11.53% |

Tokens include prompt and completion tokens for all attempts, including
failures. These are inference counts, not measured financial savings or
end-to-end speedup. Collection took about 4.36 hours, excluding server startup;
iterative training plus development evaluation took about 10.09 hours on one
A100. No-final failures fall from 45 to 22 across both repeats.

## Task families

| Family | Unique questions | SFT | Original GRPO | Iterative GRPO |
|---|---:|---:|---:|---:|
| Lookup | 20 | 65.00% | 75.00% | 82.50% |
| Ratio | 12 | 45.83% | 58.33% | 75.00% |
| Difference | 10 | 20.00% | 25.00% | 45.00% |
| Larger-company comparison | 10 | 45.00% | 55.00% | 75.00% |
| Compare then lookup | 8 | 0.00% | 6.25% | 12.50% |
| Unavailable at cutoff | 4 | 100.00% | 100.00% | 100.00% |

## Remaining failures

- Compare-then-lookup remains difficult. Policies sometimes select the right
  company but return revenue when assets were requested, or return the
  comparison metric instead of the requested follow-up metric.
- Sign and unit/precision errors remain. A negative profit must be subtracted
  with its sign, and ratio answers must carry the requested percent unit.
- Two attempts give a correct company and revenue value but cite one extra
  valid calculation. Exact proof-multiset matching rejects them: iterative
  repeat 1 and original GRPO repeat 2 on `policy_eval_330f4adb1d9a0316`.
  The frozen scores were not relaxed after inspecting this behavior.

## Evidence and scope

The original audit replayed all 384 episodes, checked 2,859 model responses
against server settings and adapter assignments, verified 37 released payload
hashes and reinspected all 20 source figures. Independent Decimal arithmetic
recomputed all 64 reference answers. No labels or grades changed. The public
action export supports deterministic tool/grade replay; raw server logs and
model weights are not included.

The evidence comes from four reports, templated questions and two fixed company
pairs, with same-author source review. Base-model pretraining exposure is
unknown. Only one training seed was used; two inference repeats are not two
training seeds. Greedy trajectories can vary across restarts. Original GRPO
used two updates on 104 episodes, while iterative GRPO used six updates on
312 episodes; training compute and refreshed rollouts are confounded.

The 64-question test is now published and inspected. Further tuning requires a
new reserved test to support new held-out claims. An earlier, separate
32-question/two-company experiment scored 42.19% → 45.31%; do not pool that
benchmark with this one.

Machine-readable evidence: [summary](../reports/heldout-v2.json),
[episode grades](../reports/episode-scores.jsonl),
[original-code and artifact hashes](../reports/provenance.json).
