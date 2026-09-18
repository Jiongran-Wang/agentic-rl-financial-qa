# Reproduction scope

## Available without a GPU

1. `python3 scripts/unpack_inputs.py` verifies the archive and every member
   hash before unpacking the 25 experiment input files. Existing changed files
   are never overwritten.
2. `python3 scripts/demo.py` replays a recorded successful policy episode
   through the actual search/read/calculation environment and proof verifier.
3. `python3 scripts/audit_public_results.py` replays all 384 published action
   sequences, checks each full-episode hash and recomputes proof grades and
   the predeclared three-policy summary.

The public replay is deliberately narrower than the original local audit.
It does not regenerate model responses, inspect server processes or remeasure
token counts. Prompt/output truncation outcomes use the retained classification
from the original audit, since full prompts and server responses are omitted.
The complete original archive is identified by SHA256 in
`reports/provenance.json`; a hash is an integrity commitment, not independent
proof that the original experiment ran.

## CPU model and contract tests

Python 3.12 was used for the publication test run. Install
`requirements-test.txt` in an isolated environment, then run:

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
```

These tests instantiate tiny random Qwen3 models locally. They exercise real
LoRA gradients and save/reload, but they are not a rerun of the 4B experiment.
The standard-library protocol tests can also be run individually without the
model packages. The GitHub Actions workflow uses that lightweight subset.

## Historical GPU experiment

The original experiment used a CUDA A100 with a pinned base revision,
PyTorch 2.6.0, Transformers 4.51.3, PEFT 0.15.2 and Accelerate 1.6.0.
The evaluation runtime used vLLM 0.8.5.post1 V0. Keep training and serving in
separate environments; the original workspace used a LoRA dependency overlay.

The published trainers are unchanged copies of the historical implementations.
The bundled inputs include the 204 SFT action samples, 26 RL training questions,
27 development questions, frozen experiment contracts and the 64-question
evaluation data. Source paths and hashes are listed in the input manifest.

SFT entry point, after obtaining the pinned base model and writing a manifest
with `repo_id`, `revision` and its absolute `snapshot_path`:

```bash
python scripts/train_retail_sft_pilot_v1.py \
  --model /path/to/pinned-qwen3-4b \
  --model-manifest /path/to/model-manifest.json \
  --samples results/retail_reward_v2_sft_v1/action_samples.train.jsonl \
  --release results/retail_reward_v2_sft_v1/validation.json \
  --output results/new-sft-run
```

The three-round driver is `scripts/run_retail_grpo_iter_v1.py`. Its historical
contract pins the original SFT adapter bytes and expects the epoch-2 adapter at
`results/retail_sft_pilot_v1/job_3159735/epoch_2`. The collector similarly pins
all three original adapters. **Those checkpoints are not distributed here.**
Therefore the published release is not a one-command, end-to-end reproduction
of new model generation or the exact trained weights.

A new SFT run may produce different checkpoint hashes. To run a new experiment,
create a new versioned contract and explicitly bind the new initialization and
output paths; do not disable hash checks or relabel the result as the historical
run. Use new reserved data if changing or tuning the policy. The original
Slurm account, host paths, model caches and job scripts are intentionally not
part of the portable release.

## What is excluded

Model weights, full raw server logs, machine credentials, cluster account
settings, raw PDFs, caches, vendored LLaMA-Factory/verl trees and unrelated
historical RAG experiments. The retained legacy helper modules are imported
dependencies of the retail environment, not a claim that those frameworks
trained the final policy.
