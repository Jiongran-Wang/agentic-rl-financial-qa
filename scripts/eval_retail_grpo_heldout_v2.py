#!/usr/bin/env python3
"""Matched greedy SFT/GRPO reserved rollouts; private scores remain local."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from training.retail.environment import RetailEpisode,digest
from training.retail.rollout import messages_for
from training.retail.policy_compare_v1 import REVISION,sha
from training.retail.grpo_heldout_v2 import VERSION,ARMS,MODEL_NAMES,load_public,schedule,sampling_payload,run_episode,arm_metrics
from retail_pilot import write_json


def request_greedy(base_url, arm, messages, seed):
    payload=sampling_payload(arm,messages,seed)
    request=Request(base_url.rstrip('/')+'/chat/completions',data=json.dumps(payload).encode(),
                    headers={'Content-Type':'application/json'})
    started=time.perf_counter()
    try:
        with urlopen(request,timeout=180) as response:data=json.load(response)
    except HTTPError as exc:
        raise RuntimeError(f'Model endpoint returned HTTP {exc.code}') from None
    if len(data['choices']) != 1:raise ValueError('Expected exactly one completion')
    choice=data['choices'][0]
    return dict(raw=choice['message']['content'],finish_reason=choice.get('finish_reason'),
                served_model=data.get('model'),usage=data.get('usage'),seconds=time.perf_counter()-started)


def preflight(a):
    if os.environ.get('VLLM_USE_V1')!='0':raise ValueError('Pinned V0 engine required')
    versions={n:importlib.metadata.version(n) for n in ('torch','transformers','vllm')}
    for n,v in dict(torch='2.6.0',transformers='4.51.3',vllm='0.8.5.post1').items():
        if versions[n].split('+')[0]!=v:raise ValueError('Version mismatch: '+n)
    manifest=json.loads((ROOT/'model-manifest.json').read_text())
    if manifest['repo_id']!='Qwen/Qwen3-4B' or manifest['revision']!=REVISION or a.model.resolve()!=Path(manifest['snapshot_path']).resolve():
        raise ValueError('Original pinned base required')
    corpus,tasks,release=load_public(a.data)
    for name,h in release['tokenizer_hashes'].items():
        if sha(a.model/name)!=h:raise ValueError('Changed tokenizer: '+name)
    adapters={}
    for arm in ARMS:
        spec=release['adapters'][arm];directory=ROOT/spec['path']
        if sha(directory/'adapter_model.safetensors')!=spec['weight_sha256']:raise ValueError('Wrong adapter weights: '+arm)
        if sha(directory/'adapter_config.json')!=spec['config_sha256']:raise ValueError('Wrong adapter config: '+arm)
        adapters[arm]=dict(path=str(directory.resolve()),weight_sha256=spec['weight_sha256'],config_sha256=spec['config_sha256'])
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(a.model),local_files_only=True)
    def count(messages):
        return len(tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False))
    initial=[]
    for task in tasks:
        n=count(messages_for(RetailEpisode(corpus,task,12,18000)))
        if n>30720:raise ValueError('Initial prompt exceeds context limit')
        initial.append(dict(id=task['id'],prompt_tokens=n))
    contract=dict(version=VERSION,versions=versions,base_revision=REVISION,base_path=str(a.model.resolve()),
        adapters=adapters,models=MODEL_NAMES,protocol=release['protocol'],serving=release['serving'],expected_episodes=192,repeat_index=a.repeat_index,
        release_sha256=sha(a.data/'release.json'),task_sha256=sha(a.data/'public.tasks.jsonl'),
        corpus_sha256=digest(corpus),initial_budgets=initial,private_labels_loaded=False,
        planned_reserved_question_attempts=192,weight_updates=0,fresh_test=True)
    return corpus,tasks,count,contract


def run(a):
    if a.output.exists():raise FileExistsError(a.output)
    corpus,tasks,count,contract=preflight(a)
    a.output.mkdir(parents=True,exist_ok=False)
    write_json(a.output/'contract.json',contract)
    if a.preflight_only:return
    with urlopen(a.base_url.rstrip('/')+'/models',timeout=30) as response:catalog=json.load(response)
    models={m['id']:m for m in catalog['data']}
    for arm in ARMS:
        registered=models.get(MODEL_NAMES[arm],{}).get('root')
        if not isinstance(registered,str) or Path(registered).resolve()!=Path(contract['adapters'][arm]['path']).resolve():
            raise ValueError('Expected adapter is not registered: '+arm)
    write_json(a.output/'server_models.json',catalog)
    jobs=schedule(tasks)
    write_json(a.output/'schedule.json',[dict(arm=arm,id=task['id']) for arm,task in jobs])
    rows=[]
    with (a.output/'rollouts.jsonl').open('x') as f:
        for arm,task in jobs:
            started=time.perf_counter()
            row=run_episode(corpus,task,arm,count,lambda messages,seed:request_greedy(a.base_url,arm,messages,seed))
            row.update(seconds=time.perf_counter()-started,provenance='predeclared_reserved_comparison',repeat_index=a.repeat_index)
            rows.append(row);f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
            print(f"{len(rows)}/{len(jobs)} {arm} {task['id']} {row['outcome']['outcome']} steps={len(row['episode']['trace'])}",flush=True)
    groups={arm:arm_metrics([r for r in rows if r['arm']==arm],len(tasks)) for arm in ARMS}
    write_json(a.output/'summary.json',dict(status='collection_complete_private_scoring_pending',
        groups=groups,attempted=len(rows),expected=len(jobs),weight_updates=0,repeat_index=a.repeat_index,reserved_question_attempts=len(rows),reserved_model_calls=sum(len(r['calls']) for r in rows)))
    if any(g['infrastructure_errors'] for g in groups.values()):raise SystemExit(2)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'data/retail_grpo_heldout_v2')
    p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--base-url',default='http://127.0.0.1:8000/v1');p.add_argument('--preflight-only',action='store_true')
    p.add_argument('--repeat-index',type=int,choices=(1,2),required=True)
    run(p.parse_args())
