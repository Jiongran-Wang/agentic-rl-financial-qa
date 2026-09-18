#!/usr/bin/env python3
"""Replay one real, successful held-out trajectory. No model or API key needed."""
import gzip
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.retail.environment import RetailEpisode, digest
from training.retail.rewards_v2 import NumericVerifierV2


def main():
    task_id = 'policy_eval_0d77c3ec12965094'
    rows = map(json.loads, gzip.decompress((ROOT / 'artifacts/evaluation-actions.jsonl.gz').read_bytes()).decode().splitlines())
    row = next(r for r in rows if r['repeat'] == 1 and r['arm'] == 'iterative' and r['id'] == task_id)
    corpus = json.loads((ROOT / 'data/retail_grpo_heldout_v2/corpus.json').read_text())
    env = RetailEpisode(corpus, row['task'], 12, 18000)
    print('RECORDED POLICY REPLAY — no live model inference\n')
    print('Question:', row['task']['question'])
    for i, raw in enumerate(row['actions'], 1):
        observation = env.step(raw)
        action = json.loads(raw)
        print(f'\n{i}. {action["type"].upper()}')
        print(json.dumps(action, ensure_ascii=False))
        if 'calculations' in observation:
            print('Source-backed calculation:', json.dumps(observation['calculations'], ensure_ascii=False))
        elif not observation['ok']:
            print('Tool error:', observation.get('error'))
    assert digest(env.record()) == row['episode_sha256']
    private = ROOT / 'data/retail_policy_eval_v2/private'
    ref = next(r for r in map(json.loads, (private / 'references.jsonl').read_text().splitlines()) if r['id'] == task_id)
    grade = NumericVerifierV2(corpus, json.loads((private / 'equivalent_cells.json').read_text())).score(env, ref)
    assert grade['score'] == 1
    print('\nAnswer:', env.prediction['answer'])
    print('Verification:', grade['reason'])


if __name__ == '__main__':
    main()
