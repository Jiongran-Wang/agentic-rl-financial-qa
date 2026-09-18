# Method and engineering decisions

## A learned tool policy

Each episode presents a question and a publication cutoff. The model emits one
JSON action at a time: search, read, calculate, or final. Search discovers
chunks; read exposes source content and a cell catalog. Calculations reference
those cells rather than model-invented financial values. A final answer links
the calculations and their supporting citations. Errors consume the action
budget and are returned as observations the policy can use to recover.

The environment enforces a maximum of 12 actions, including a final-only
boundary. Inference uses 30,720 prompt tokens and 2,048 output tokens within a
32,768-token context. Tool observations are capped at 18,000 characters.

## Supervised initialization

The Qwen3-4B base is pinned to revision
`1cfa9a7208912126459214e8b04321603b3df60c`.
LoRA uses rank 16, alpha 32 and zero dropout over seven attention/MLP projection
types. The 26 training questions yield 44 demonstration trajectories and 204
action samples. Two SFT epochs perform 102 optimizer updates. Observation and
prompt tokens are masked out; model actions receive the loss.

The SFT epoch-2 checkpoint is the initialization and fixed reference for the
three-round experiment. It is separate from the much larger historical RAG
experiments in the original development workspace.

## Iterative GRPO

Each round samples four trajectories per training question at temperature 0.7:
26 × 4 = 104 episodes. Binary proof rewards are normalized within each group.
The trainer performs two full-batch updates, then uses the resulting actor to
collect the next round. Three rounds produce 312 fresh episodes and six updates.

The update uses the clipped current/old policy ratio, a sampled K3 penalty
against the fixed SFT reference, and a separate detached correction
`min(exp(old_teacher_logp - behavior_sampled_logp), 2)` multiplying both terms.
This correction addresses an observed sampling/teacher-forcing probability
mismatch. It is a token correction at sampled contexts, not an exact
trajectory-distribution correction or an unbiased KL estimator.

Parameters: AdamW, learning rate 1e-6, zero weight decay, gradient norm limit 1,
PPO clip 0.2 and KL coefficient 0.02. Optimizer state resets each round. Checks
require complete binary groups, at least four mixed-reward groups, finite
gradients, bounded probability replay differences and sampled corrected K3
below 0.1. Frozen base parameters and exact adapter reloads are checked.

## Evidence-based reward

The verifier checks the expected final action, company, period, unit, rounded
numeric value and source-cell operation. Required calculation proofs and
citations must match. Correct text alone is insufficient. Unavailable-at-cutoff
questions require the appropriate scoped search and abstention behavior.

This makes rewards executable and reviewable, but the exact proof rule can be
too restrictive: extra valid calculations can cause a correct answer to fail.
The published result keeps that frozen rule and discloses the affected cases.

## Selection and evaluation

SFT and all three round checkpoints are evaluated on 27 existing development
questions. Selection requires a strict increase in success; ties use fewer
invalid actions, fewer tokens, then the earlier checkpoint. Round 3 is selected
before the reserved evaluation. Its development result is 18/27 versus 13/27.

The final evaluation has 64 questions, three fixed policies and two complete
repeats. Each repeat starts a new vLLM V0 server. All arms use bf16, greedy
decoding, seed 73 and identical budgets; their order rotates across questions.
The primary metric averages both repeat scores for each question, then averages
the 64 questions. Missing or unscorable attempts block aggregate publication.
No checkpoint or repeat is selected from the held-out results.
