"""Public-only comparison contract; no private reference paths or scoring imports."""
import hashlib
import json
from pathlib import Path

ARMS = ('base', 'epoch_1', 'epoch_2')
SPLITS = ('train_diagnostic', 'development')
MODEL_NAMES = {a: 'retail-compare-' + a for a in ARMS}
REVISION = '1cfa9a7208912126459214e8b04321603b3df60c'
CHECKPOINT_HASHES = {
    'epoch_1': '18cd6882203db2cd8b363d2cd3b63f394ac5cd1215e821674f255719e320b04c',
    'epoch_2': '03bbc8defc8dd2463349554eec1641390351b840dc66a0d4b819da9e97dc5f1b',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_public(data):
    result = {}
    for split in SPLITS:
        directory = Path(data) / split
        tasks = [json.loads(l) for l in (directory / 'public.tasks.jsonl').read_text().splitlines() if l.strip()]
        if not tasks or any(set(t) != {'id', 'question', 'as_of'} for t in tasks):
            raise ValueError('Missing tasks or non-public fields')
        if len({t['id'] for t in tasks}) != len(tasks):
            raise ValueError('Duplicate task IDs')
        result[split] = (json.loads((directory / 'corpus.json').read_text()), tasks)
    if {t['id'] for t in result[SPLITS[0]][1]} & {t['id'] for t in result[SPLITS[1]][1]}:
        raise ValueError('Training/development task overlap')
    return result


def schedule(public):
    """Rotate model order by task; each arm sees every task exactly once."""
    jobs = []
    for split in SPLITS:
        for i, task in enumerate(public[split][1]):
            arms = ARMS[i % 3:] + ARMS[:i % 3]
            jobs.extend((split, arm, task) for arm in arms)
    return jobs


def validate_response(response, model_name, local_prompt_tokens):
    if response.get('served_model') != model_name:
        raise ValueError('Response came from an unexpected model arm')
    usage = response.get('usage') or {}
    if any(type(usage.get(k)) is not int or usage[k] < 0
           for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')):
        raise ValueError('Missing token accounting')
    if usage['prompt_tokens'] != local_prompt_tokens:
        raise ValueError('Server tokenizer/template differs from local prompt')
    if usage['total_tokens'] != usage['prompt_tokens'] + usage['completion_tokens']:
        raise ValueError('Inconsistent token accounting')
    if usage['completion_tokens'] > 2048:
        raise ValueError('Output budget exceeded')


def metrics(rows, expected):
    traces = [t for r in rows for t in r['episode']['trace']]
    calls = [c for r in rows for c in r['calls']]
    repeats = 0
    tool_calls = 0
    for row in rows:
        seen = set()
        for t in row['episode']['trace']:
            action = t['action']
            if isinstance(action, str):
                try:
                    action = json.loads(action)
                except ValueError:
                    pass
            key = json.dumps(action, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
            repeats += key in seen
            seen.add(key)
            tool_calls += isinstance(action, dict) and action.get('type') in {'search', 'read', 'calculate'}
    return dict(expected=expected, attempted=len(rows), missing=expected-len(rows),
                infrastructure_errors=sum(r['status'] != 'complete' for r in rows),
                finals=sum(bool(r['episode']['prediction']) for r in rows),
                actions=len(traces), invalid_actions=sum(not t['observation']['ok'] for t in traces),
                repeated_actions=repeats, tool_calls=tool_calls, model_calls=len(calls),
                tokens=sum((c['response'].get('usage') or {}).get('total_tokens', 0) for c in calls))
