#!/usr/bin/env python3
"""Bounded three-round driver; each GPU phase has a fresh process."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.retail.grpo_iter_v1 import select_candidate
from training.retail.policy_compare_v1 import sha


def run(a):
    common = ['--model', str(a.model), '--model-manifest', str(a.model_manifest), '--data', str(a.data)]
    initial = ROOT/'results/retail_sft_pilot_v1/job_3159735/epoch_2'
    adapter, candidates = initial, []
    for r in range(4):
        if r:
            training = a.output/f'round_{r}'
            command = [sys.executable, str(ROOT/'scripts/train_retail_grpo_iter_round_v1.py'), *common,
                '--adapter', str(adapter), '--anchor', str(initial), '--round', str(r), '--output', str(training)]
            if r > 1:
                command += ['--parent-summary', str(a.output/f'round_{r-1}/summary.json')]
            subprocess.run(command, check=True, cwd=ROOT)
            adapter = training/'adapter_final'
        development = a.output/f'development_{r}'
        subprocess.run([sys.executable, str(ROOT/'scripts/eval_retail_grpo_iter_v1.py'), *common,
            '--adapter', str(adapter), '--round', str(r), '--output', str(development)], check=True, cwd=ROOT)
        candidate = json.loads((development/'summary.json').read_text())
        candidate['summary_sha256'] = sha(development/'summary.json')
        candidates.append(candidate)
    selected = select_candidate(candidates)
    decision = dict(status='three_rounds_complete_development_selected',
        release_sha256=sha(a.data/'release.json'), selected=selected, candidates=candidates,
        optimizer_updates=6, new_training_episodes=312, development_episodes=108,
        reserved_policy_calls=0, selection_rule='Strict success improvement over baseline; then success, invalid actions, tokens, earlier round',
        promotion='Development candidate only; new reserved evaluation and trajectory audit still required')
    (a.output/'selection.json').write_text(json.dumps(decision, indent=2)+'\n')
    print(json.dumps(decision), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--model-manifest', type=Path, required=True)
    p.add_argument('--data', type=Path, default=ROOT/'data/retail_grpo_iter_v1')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    try:
        run(a)
    except Exception as exc:
        (a.output/'failure.json').write_text(json.dumps(dict(status='stopped_no_selection', error_type=type(exc).__name__, error=str(exc)))+'\n')
        raise
