"""Frozen, public-only SFT/GRPO reserved comparison; no scorer imports."""
from copy import deepcopy
import json
from pathlib import Path
from .environment import RetailEpisode, digest
from .rollout import messages_for
from .policy_compare_v1 import sha, validate_response, metrics
from .budget_outcomes_v2 import classify, PROMPT_LIMIT, OUTPUT_LIMIT

VERSION = 'retail-grpo-heldout-v2'
ARMS = ('sft', 'grpo', 'iterative')
MODEL_NAMES = {arm: 'retail-grpo-heldout-v2-' + arm for arm in ARMS}
SEED = 73


def load_public(data):
    data = Path(data)
    release = json.loads((data / 'release.json').read_text())
    for name, h in release['public_hashes'].items():
        if sha(data / name) != h: raise ValueError('Changed public input: ' + name)
    tasks = [json.loads(l) for l in (data / 'public.tasks.jsonl').read_text().splitlines()]
    if len(tasks) != 64 or len({t['id'] for t in tasks}) != 64 or any(set(t) != {'id','question','as_of'} for t in tasks):
        raise ValueError('Exactly 64 public reserved tasks required')
    corpus = json.loads((data / 'corpus.json').read_text())
    if {c['doc_id'] for c in corpus} != set(release['corpus_documents']):
        raise ValueError('Wrong reserved corpus documents')
    return corpus, tasks, release


def schedule(tasks):
    return [(arm, task) for i, task in enumerate(tasks) for arm in (ARMS[i % 3:] + ARMS[:i % 3])]


def sampling_payload(arm, messages, seed=SEED):
    if arm not in ARMS or seed != SEED: raise ValueError('Unknown arm or seed')
    return dict(model=MODEL_NAMES[arm], messages=messages, temperature=0.0,
        top_p=1.0, top_k=-1, seed=SEED, n=1, max_tokens=OUTPUT_LIMIT,
        repetition_penalty=1.0, presence_penalty=0.0, frequency_penalty=0.0,
        chat_template_kwargs={'enable_thinking':False})


def arm_metrics(rows, expected):
    safe_rows = deepcopy(rows)
    invalid_accounting_calls = 0
    for row in safe_rows:
        for call in row['calls']:
            usage = call['response'].get('usage')
            if not isinstance(usage, dict): usage = {}
            valid = all(type(usage.get(k)) is int and usage[k] >= 0 for k in ('prompt_tokens','completion_tokens','total_tokens'))
            valid = valid and usage['total_tokens'] == usage['prompt_tokens'] + usage['completion_tokens']
            if not valid:
                invalid_accounting_calls += 1
                usage = dict(prompt_tokens=0, completion_tokens=0, total_tokens=0)
            call['response']['usage'] = usage
    result = metrics(safe_rows, expected)
    result['calls_with_unusable_token_accounting'] = invalid_accounting_calls
    result['legacy_noncomplete_count'] = result.pop('infrastructure_errors')
    result['infrastructure_errors'] = sum(r['outcome']['outcome']=='infrastructure_failure' for r in rows)
    result['policy_budget_failures'] = sum(r['outcome']['outcome']=='policy_budget_failure' for r in rows)
    for field in ('prompt_tokens','completion_tokens'):
        result[field] = sum((c['response'].get('usage') or {}).get(field,0) for r in safe_rows for c in r['calls'])
    return result


def run_episode(corpus, task, arm, count_tokens, request):
    if arm not in ARMS: raise ValueError('Unknown evaluation arm')
    env=RetailEpisode(corpus,task,12,18000);calls=[]
    def record(status,error=None,budget=None):
        row=dict(id=task['id'],arm=arm, split='reserved',episode=env.record(),calls=calls,
                 status=status,error=error,budget=budget)
        row['outcome']=classify(row,MODEL_NAMES[arm])
        return row
    while not env.done:
        messages=messages_for(env)
        try:
            n=count_tokens(messages)
            if type(n) is not int or n<0:raise ValueError('Invalid prompt token count')
        except Exception as exc:
            return record('infrastructure_error',type(exc).__name__+': '+str(exc))
        if n>PROMPT_LIMIT:
            return record('budget_exhausted','Prompt budget exceeded',dict(kind='prompt_token_limit',
                          prompt_tokens=n,limit=PROMPT_LIMIT,messages_sha256=digest(messages)))
        seed=SEED
        try:
            response=request(deepcopy(messages),seed)
        except Exception as exc:
            return record('infrastructure_error',type(exc).__name__+': '+str(exc))
        if not isinstance(response,dict):
            return record('infrastructure_error','Non-object response')
        response['prompt_tokens_local']=n
        calls.append(dict(messages=messages,response=deepcopy(response),seed=seed,
                          request_sha256=digest(sampling_payload(arm,messages,seed))))
        try:
            validate_response(response,MODEL_NAMES[arm],n)
            if not isinstance(response.get('raw'),str):raise ValueError('Missing response text')
        except (KeyError,TypeError,ValueError) as exc:
            return record('infrastructure_error',str(exc))
        if response.get('finish_reason')=='length':
            return record('budget_exhausted','Output token limit',dict(kind='output_token_limit',limit=OUTPUT_LIMIT))
        if response.get('finish_reason')!='stop':
            return record('infrastructure_error','Unknown or missing finish reason')
        env.step(response['raw'])
    return record('complete')

