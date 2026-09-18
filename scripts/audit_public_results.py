#!/usr/bin/env python3
"""Replay released actions and recompute strict proof grades without model inference."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.retail.environment import RetailEpisode, digest
from training.retail.grpo_heldout_v2 import load_public
from training.retail.rewards_v2 import NumericVerifierV2
from scripts.score_retail_grpo_heldout_v2 import aggregate_repeats


def run():
    archive = ROOT / 'artifacts/evaluation-actions.jsonl.gz'
    provenance = json.loads((ROOT / 'reports/provenance.json').read_text())
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == provenance['compact_actions_sha256']
    for name, expected in provenance['source_files_unmodified_sha256'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    corpus, tasks, release = load_public(ROOT / 'data/retail_grpo_heldout_v2')
    by_id = {t['id']: t for t in tasks}
    private = ROOT / 'data/retail_policy_eval_v2/private'
    refs = {r['id']: r for r in map(json.loads, (private / 'references.jsonl').read_text().splitlines())}
    verifier = NumericVerifierV2(corpus, json.loads((private / 'equivalent_cells.json').read_text()))
    rows = list(map(json.loads, gzip.decompress(archive.read_bytes()).decode().splitlines()))
    assert len(rows) == 384
    grades = {1: [], 2: []}
    for index, row in enumerate(rows, 1):
        assert row['task'] == by_id[row['id']]
        env = RetailEpisode(corpus, row['task'], 12, 18000)
        for action in row['actions']:
            env.step(action)
        assert digest(env.record()) == row['episode_sha256'], (row['repeat'], row['arm'], row['id'])
        assert env.prediction == row['prediction']
        if row['budget']:
            # Prompt length/output truncation attestations require the original
            # tokenizer/server logs; this export does not pretend to rerun them.
            assert row['outcome']['outcome'] == 'policy_budget_failure'
            assert row['budget']['kind'] in {'prompt_token_limit', 'output_token_limit'}
            grade = dict(score=0.0, eligible=True, reason=row['budget']['kind'])
        else:
            grade = verifier.score(env, refs[row['id']])
        assert all(grade[k] == row['grade'][k] for k in ['score', 'eligible', 'reason'])
        grades[row['repeat']].append(dict(id=row['id'], arm=row['arm'], **grade))
        if index % 32 == 0:
            print(f'Replayed {index}/384 episodes', flush=True)
    _, summary = aggregate_repeats(list(by_id), grades)
    expected = json.loads((ROOT / 'reports/heldout-v2.json').read_text())['primary']
    # JSON stores repeat-index dictionary keys as strings.
    assert json.loads(json.dumps(summary)) == expected
    print(json.dumps(dict(status='public_action_replay_and_grading_passed',
        episodes=384, unique_questions=64, success_rates=summary['success_rates'],
        delta_percentage_points=summary['deltas_percentage_points'],
        scope='Action/environment replay and strict proof grading; not new model generation or independent server-log verification.'), indent=2))


if __name__ == '__main__':
    run()
