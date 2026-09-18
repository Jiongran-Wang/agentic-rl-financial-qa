"""Framework-independent policy rollout and replay-validated SFT conversion."""
from copy import deepcopy
import json
from .environment import RetailEpisode, SYSTEM, canonical, replay
from .rewards import score_numeric, require_trainable


def messages_for(env):
    messages = [{'role':'system', 'content':SYSTEM}, {'role':'user', 'content':canonical(env.initial())}]
    # Initial observation must describe the initial budget, not the current one.
    initial = dict(task=env.task, environment=env.record()['environment'], steps_remaining=env.max_steps, final_only=False, done=False)
    messages[1]['content'] = canonical(initial)
    for t in env.trace:
        a = t['action']
        messages.append({'role':'assistant', 'content':a if isinstance(a, str) else canonical(a)})
        messages.append({'role':'user', 'content':canonical(t['observation'])})
    return messages


def run_policy(corpus, task, request, max_steps=10, observation_chars=18000, prompt_guard=None):
    env = RetailEpisode(corpus, task, max_steps, observation_chars)
    calls = []
    while not env.done:
        messages = messages_for(env)
        # Tokenization belongs to the caller's model adapter. Never silently
        # truncate tool history or remove a source to make a prompt fit.
        try:
            if prompt_guard:
                prompt_guard(messages)
            response = request(deepcopy(messages))
        except Exception as exc:
            return dict(episode=env.record(), calls=calls, status='infrastructure_error', error=f'{type(exc).__name__}: {exc}')
        calls.append(dict(messages=messages, response=deepcopy(response)))
        if not isinstance(response, dict) or not isinstance(response.get('raw'), str) or response.get('finish_reason') != 'stop':
            return dict(episode=env.record(), calls=calls, status='infrastructure_error', error='Missing or truncated model response')
        env.step(response['raw'])
    return dict(episode=env.record(), calls=calls, status='complete', error=None)


def to_sft(corpus, rollout, reference):
    require_trainable(reference)
    if rollout['status'] != 'complete':
        raise ValueError('Cannot train on an incomplete infrastructure-failed rollout')
    env = replay(corpus, rollout['episode'])
    reward = score_numeric(env, reference)
    if reward['score'] != 1:
        raise ValueError('SFT requires verified successful trajectory')
    # Preserve actual observations and repairs; only assistant messages are
    # targets. The dataset adapter must mask system/user observation tokens.
    messages = messages_for(env)[:-1]  # Exclude terminal environment echo.
    return dict(id=env.task['id'], messages=messages, reward=reward,
                corpus_sha256=env.corpus_sha256, target_roles=['assistant'],
                provenance=rollout.get('provenance', 'model_rollout'))
