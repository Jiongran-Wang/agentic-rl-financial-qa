"""Strict source commitments and deterministic environment replay; no policy calls."""
import json
import math
from training.retail.environment import RetailEpisode
from training.retail.rollout import messages_for
from training.retail.grpo_update_v1 import seed_for, action_row
from training.retail.policy_compare_v1 import sha


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def load_batch(source, diagnostic, release, tasks, corpus, verifier, refs):
    for label, directory in [('source', source), ('diagnostic', diagnostic)]:
        for name, expected in release[label + '_hashes'].items():
            if sha(directory / name) != expected:
                raise ValueError('Changed committed ' + label + ' input: ' + name)
    if any((source / name).exists() for name in ['adapter_final', 'summary.json']):
        raise ValueError('Source must be the failed, untouched run')
    progress = read_rows(source / 'progress.jsonl')
    if any(x['stage'] in {'backward', 'optimizer_update_passed', 'complete'} for x in progress):
        raise ValueError('Source batch already used for optimization')
    rows = read_rows(source / 'rollouts.jsonl')
    schedule = [(t['id'], i) for i in range(4) for t in tasks]
    if [(r['id'], r['sample_index']) for r in rows] != schedule:
        raise ValueError('Incomplete or reordered source schedule')
    by_id = {t['id']: t for t in tasks}
    for row in rows:
        env = RetailEpisode(corpus, by_id[row['id']], 12, 18000)
        if row['budget'] is not None:
            raise ValueError('Unexpected budget outcome in committed batch')
        for turn, call in enumerate(row['calls']):
            r = call['response']
            if env.done or call['messages'] != messages_for(env) or call['seed'] != seed_for(row['id'], row['sample_index'], env.steps):
                raise ValueError('Source policy history mismatch')
            ids = r['generated_ids']
            if r['kind'] != 'stop' or ids[-1] not in r['eos_ids'] or any(x in r['eos_ids'] for x in ids[:-1]):
                raise ValueError('Unexpected source stop')
            if len(ids) != len(r['sampled_logps']) or not all(math.isfinite(x) and x <= 1e-5 for x in r['sampled_logps']):
                raise ValueError('Invalid source probabilities')
            action_row(r['prompt_ids'], ids)
            if env.step(r['raw']) != row['episode']['trace'][turn]['observation']:
                raise ValueError('Tool replay changed')
        if not env.done or env.record() != row['episode'] or verifier.score(env, refs[row['id']]) != row['grade']:
            raise ValueError('Source episode/reward replay changed')
    native = read_rows(diagnostic / 'native_scan.jsonl')
    flat = [(r, turn, c) for r in rows for turn, c in enumerate(r['calls'])]
    if len(flat) != 662 or len(native) != len(flat):
        raise ValueError('Incomplete diagnostic vectors')
    for index, (d, (r, turn, c)) in enumerate(zip(native, flat)):
        validate_native(d, index, r, turn, c['response'])
    return rows, native


def validate_native(d, index, row, turn, response):
    expected = (index, row['id'], row['sample_index'], turn,
                len(response['prompt_ids']), len(response['generated_ids']))
    actual = tuple(d[k] for k in ['index', 'id', 'sample_index', 'turn', 'prompt_tokens', 'output_tokens'])
    if actual != expected or len(d['teacher_logps']) != expected[-1]:
        raise ValueError('Misaligned diagnostic vector')
    if not all(math.isfinite(x) and x <= 1e-5 for x in d['teacher_logps']):
        raise ValueError('Invalid diagnostic probabilities')
