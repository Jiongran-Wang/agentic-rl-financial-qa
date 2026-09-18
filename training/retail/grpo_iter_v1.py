"""Contracts for three fresh-rollout rounds and development-only selection."""
import hashlib
from pathlib import Path
import json
from .environment import RetailEpisode, digest
from .rollout import messages_for
from .policy_compare_v1 import sha

VERSION = 'retail-grpo-iter-v1'
ROUNDS = 3


def seed_for(round_index, task_id, sample_index, step):
    if not 1 <= round_index <= ROUNDS or not 0 <= sample_index < 4 or not 0 <= step < 12:
        raise ValueError('Invalid round, sample or step')
    key = f'{VERSION}|173|{round_index}|{task_id}|{sample_index}|{step}'
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], 'big') % 2**31


def collect_episode(corpus, task, sample_index, sampler, verifier, reference, round_index):
    env = RetailEpisode(corpus, task, 12, 18000)
    calls, budget = [], None
    while not env.done:
        messages = messages_for(env)
        seed = seed_for(round_index, task['id'], sample_index, env.steps)
        response = sampler(messages, seed)
        calls.append(dict(messages=messages, seed=seed, response=response))
        if response['kind'] != 'stop':
            budget = response['kind']
            if budget not in {'prompt_token_limit', 'output_token_limit'}:
                raise ValueError('Unscorable generation failure')
            break
        env.step(response['raw'])
    grade = dict(score=0.0, eligible=True, reason=budget) if budget else verifier.score(env, reference)
    if not grade['eligible'] or type(grade['score']) not in (int, float) or grade['score'] not in (0, 1):
        raise ValueError('Unscorable episode')
    return dict(id=task['id'], sample_index=sample_index, round=round_index,
                calls=calls, episode=env.record(), budget=budget, grade=grade)


def verify_parent(adapter, anchor, round_index, release, parent_summary=None):
    if not 1 <= round_index <= ROUNDS:
        raise ValueError('Round outside frozen schedule')
    for name, expected in release['adapter_hashes'].items():
        if sha(anchor / name) != expected:
            raise ValueError('Fixed SFT reference changed')
    if round_index == 1:
        expected = release['adapter_hashes']
    else:
        if parent_summary is None:
            raise ValueError('Missing parent audit')
        parent = json.loads(Path(parent_summary).read_text())
        if (parent['status'] != 'fresh_round_updates_and_reload_passed'
                or parent['round'] != round_index - 1
                or parent['release_sha256'] != release['_sha256']
                or parent['optimizer_updates'] != 2):
            raise ValueError('Wrong parent round')
        if Path(adapter).resolve() != (Path(parent_summary).parent / 'adapter_final').resolve():
            raise ValueError('Parent path mismatch')
        expected = parent['checkpoint_files']
    for name, h in expected.items():
        if sha(adapter / name) != h:
            raise ValueError('Parent checkpoint changed: ' + name)


def restore_adapter(model, state):
    """Swap one adapter in place before optimizer creation; verify every tensor."""
    import torch
    from peft import set_peft_model_state_dict, get_peft_model_state_dict
    set_peft_model_state_dict(model, state)
    actual = get_peft_model_state_dict(model)
    if set(actual) != set(state) or any(not torch.equal(v.detach().cpu(), state[k].cpu()) for k, v in actual.items()):
        raise ValueError('Adapter restore did not reproduce exact tensors')


def select_candidate(candidates):
    if len(candidates) != 4 or sorted(c['round'] for c in candidates) != [0, 1, 2, 3]:
        raise ValueError('Baseline and all three complete rounds required')
    for c in candidates:
        if c['status'] != 'development_complete' or c['tasks'] != 27 or c['unscorable'] != 0:
            raise ValueError('Incomplete development comparison')
        for field in ('successes', 'invalid_actions', 'tokens'):
            if type(c[field]) is not int or c[field] < 0:
                raise ValueError('Invalid development count')
        if c['successes'] > 27:
            raise ValueError('Success count exceeds denominator')
    # Require a strict primary-metric improvement to promote beyond baseline.
    baseline = next(c for c in candidates if c['round'] == 0)
    eligible = [c for c in candidates if c['successes'] > baseline['successes']]
    return min(eligible, key=lambda c: (-c['successes'], c['invalid_actions'], c['tokens'], c['round'])) if eligible else baseline
