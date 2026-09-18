#!/usr/bin/env python3
"""Frozen private-proof scoring and paired comparison after exact history replay."""
import argparse
import json
from pathlib import Path
import sys
import tarfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from training.retail.environment import RetailEpisode,digest
from training.retail.rollout import messages_for
from training.retail.policy_compare_v1 import REVISION,sha,validate_response
from training.retail.grpo_heldout_v2 import ARMS,MODEL_NAMES,SEED,load_public,schedule,sampling_payload,arm_metrics
from training.retail.budget_outcomes_v2 import classify
from training.retail.rewards_v2 import NumericVerifierV2
from retail_pilot import write_json,write_jsonl


def verify_history(corpus,task,row):
    arm=row['arm']
    if arm not in ARMS or row['split']!='reserved' or row['episode']['task']!=task:
        raise ValueError('Changed task or arm')
    env=RetailEpisode(corpus,task,12,18000);trace=row['episode']['trace']
    for i,call in enumerate(row['calls']):
        if env.done:raise ValueError('Call after terminal episode')
        messages=messages_for(env)
        if call['messages']!=messages or call['seed']!=SEED or call['request_sha256']!=digest(sampling_payload(arm,messages,SEED)):
            raise ValueError('History, seed or request contract mismatch')
        response=call['response']
        if i<len(trace):
            validate_response(response,MODEL_NAMES[arm],response['prompt_tokens_local'])
            if response.get('finish_reason')!='stop' or trace[i]['action']!=response['raw']:
                raise ValueError('Executed action differs from complete response')
            if env.step(response['raw'])!=trace[i]['observation']:raise ValueError('Tool observation mismatch')
        elif row['status']=='complete' or i!=len(row['calls'])-1:
            raise ValueError('Unexplained extra response')
    if len(trace)>len(row['calls']) or env.record()!=row['episode']:raise ValueError('Episode replay mismatch')
    budget=row.get('budget') or {}
    if budget.get('kind')=='prompt_token_limit' and budget.get('messages_sha256')!=digest(messages_for(env)):
        raise ValueError('Budget history mismatch')
    outcome=classify(row,MODEL_NAMES[arm])
    if outcome!=row['outcome']:raise ValueError('Outcome classification mismatch')
    return env,outcome


COMPARISONS = (('grpo','sft'), ('iterative','sft'), ('iterative','grpo'))


def paired_summary(task_ids,grades):
    if not task_ids or len(set(task_ids)) != len(task_ids):raise ValueError('Empty or duplicate task IDs')
    by={}
    for g in grades:
        key=(g['arm'],g['id'])
        if key in by or g['arm'] not in ARMS or g['id'] not in task_ids:raise ValueError('Unknown or duplicate grade')
        if type(g['eligible']) is not bool:raise ValueError('Invalid eligibility flag')
        if g['eligible'] and (type(g['score']) not in (int,float) or g['score'] not in (0,1)):
            raise ValueError('Eligible grade must have binary score')
        by[key]=g
    pairs=[]
    for task_id in task_ids:
        entries={a:by.get((a,task_id)) for a in ARMS}
        eligible=all(g and g['eligible'] for g in entries.values())
        pairs.append(dict(id=task_id,eligible=bool(eligible),scores={a:g['score'] if g else None for a,g in entries.items()}))
    complete=all(p['eligible'] for p in pairs)
    comparisons={}
    for target,base in COMPARISONS:
        counts=dict(both_success=0,both_fail=0,target_win=0,baseline_win=0,missing_or_unscorable=0)
        for p in pairs:
            if not p['eligible']:kind='missing_or_unscorable'
            else:
                a,b=p['scores'][target],p['scores'][base]
                kind='both_success' if a==b==1 else 'both_fail' if a==b==0 else 'target_win' if a>b else 'baseline_win'
            counts[kind]+=1
        comparisons[target+'_minus_'+base]=dict(counts=counts,
            delta_percentage_points=100*(counts['target_win']-counts['baseline_win'])/len(task_ids) if complete else None)
    result=dict(status='paired_heldout_scored_pending_source_review' if complete else 'blocked_missing_or_unscorable',
        expected_pairs=len(task_ids),eligible_pairs=sum(p['eligible'] for p in pairs),comparison_complete=complete,
        success_rates={a:sum(p['scores'][a] for p in pairs)/len(task_ids) for a in ARMS} if complete else None,
        comparisons=comparisons)
    return pairs,result


def score_repeat(run,output,data=ROOT/'data/retail_grpo_heldout_v2'):
    if output.exists():raise FileExistsError(output)
    corpus,task_list,release=load_public(data);tasks={t['id']:t for t in task_list}
    for field in ['private_hashes','reward_source_hashes']:
        for name,h in release[field].items():
            if sha(ROOT/name)!=h:raise ValueError('Changed frozen scoring input: '+name)
    private=ROOT/'data/retail_policy_eval_v2/private'
    refs={r['id']:r for r in map(json.loads,(private/'references.jsonl').read_text().splitlines())}
    if set(refs)!=set(tasks):raise ValueError('Private/public task mismatch')
    for task_id, ref in refs.items():
        if ref['task_sha256']!=digest(tasks[task_id]) or ref['corpus_sha256']!=digest(corpus) or ref.get('source_check')!='passed':
            raise ValueError('Reference provenance mismatch')
    verifier=NumericVerifierV2(corpus,json.loads((private/'equivalent_cells.json').read_text()))
    contract=json.loads((run/'contract.json').read_text())
    if contract['release_sha256']!=sha(data/'release.json') or contract['protocol']!=release['protocol']:
        raise ValueError('Collector protocol differs from frozen release')
    if (contract.get('version')!='retail-grpo-heldout-v2' or contract.get('models')!=MODEL_NAMES
        or contract.get('base_revision')!=REVISION or contract.get('serving')!=release['serving']
        or contract.get('task_sha256')!=sha(data/'public.tasks.jsonl') or contract.get('corpus_sha256')!=digest(corpus)
        or contract.get('private_labels_loaded') is not False or contract.get('weight_updates')!=0):
        raise ValueError('Collector identity or data commitment mismatch')
    for name,version in dict(torch='2.6.0',transformers='4.51.3',vllm='0.8.5.post1').items():
        if contract.get('versions',{}).get(name,'').split('+')[0]!=version:raise ValueError('Wrong collector runtime')
    repeat_index=int(run.name.removeprefix('repeat_'))
    if repeat_index not in (1,2) or contract['repeat_index']!=repeat_index or contract['expected_episodes']!=192:
        raise ValueError('Wrong repeat contract')
    for arm in ARMS:
        for key in ['weight_sha256','config_sha256']:
            if contract['adapters'][arm][key]!=release['adapters'][arm][key]:raise ValueError('Wrong evaluated adapter')
    manifest_name='RETAIL_GRPO_HELDOUT_V2_MANIFEST.sha256'
    with tarfile.open(ROOT/'dist/retail-grpo-heldout-v2.tar.gz') as archive:
        if archive.extractfile('agenticRAG-retail/'+manifest_name).read()!=(run.parent/manifest_name).read_bytes():
            raise ValueError('Collector manifest differs from released package')
    for line in (run.parent/manifest_name).read_text().splitlines():
        h,name=line.split(maxsplit=1)
        if sha(ROOT/name)!=h:raise ValueError('Changed collector source: '+name)
    rows=[json.loads(l) for l in (run/'rollouts.jsonl').read_text().splitlines() if l.strip()]
    expected=[(a,t['id']) for a,t in schedule(task_list)]
    if [(r['arm'],r['id']) for r in rows]!=expected[:len(rows)]:raise ValueError('Missing middle, reordered, extra or duplicate attempt')
    grades=[]
    for row in rows:
        if row.get('repeat_index')!=repeat_index:raise ValueError('Wrong row repeat index')
        env,outcome=verify_history(corpus,tasks[row['id']],row)
        if outcome['outcome']=='complete':grade=verifier.score(env,refs[row['id']])
        else:grade=dict(score=outcome['fixed_reward'],eligible=outcome['eligible'],reason=outcome['reason'])
        grades.append(dict(id=row['id'],arm=row['arm'],operation=refs[row['id']]['operation'],outcome=outcome['outcome'],**grade))
        if len(grades)%10==0:print(f'Replayed {len(grades)}/{len(rows)}',flush=True)
    pairs,paired=paired_summary(list(tasks),grades)
    groups={}
    for arm in ARMS:
        selected=[r for r in rows if r['arm']==arm];gs=[g for g in grades if g['arm']==arm]
        groups[arm]=arm_metrics(selected,len(tasks))
        groups[arm].update(source_proof_successes=sum(g['score']==1 for g in gs),unscorable=sum(not g['eligible'] for g in gs),
            by_operation={op:dict(expected=sum(r['operation']==op for r in refs.values()),successes=sum(g['score']==1 and g['operation']==op for g in gs)) for op in sorted({r['operation'] for r in refs.values()})})
    costs={t+'_minus_'+b:{k:groups[t][k]-groups[b][k] for k in ['invalid_actions','repeated_actions','tool_calls','tokens','prompt_tokens','completion_tokens']} for t,b in COMPARISONS}
    result=dict(status=paired['status'],groups=groups,paired=paired,cost_deltas=costs,
        cost_comparison_complete=paired['comparison_complete'],rollouts_sha256=sha(run/'rollouts.jsonl'),
        fresh_test=True,reserved_question_attempts=len(rows),reserved_model_calls=sum(len(r['calls']) for r in rows),weight_updates=0,automatic_promotion=False,
        note=release['heldout_status'])
    output.mkdir(parents=True);write_jsonl(output/'grades.jsonl',grades);write_jsonl(output/'pairs.jsonl',pairs);write_json(output/'summary.json',result)
    return result


def aggregate_repeats(task_ids,repeat_grades):
    """Three arms, two repeats; one paired unit per unique question."""
    if set(repeat_grades)-{1,2}:raise ValueError('Unexpected repeat')
    per_repeat={};indexed={}
    for repeat in (1,2):
        grades=repeat_grades.get(repeat,[])
        _,per_repeat[repeat]=paired_summary(task_ids,grades)
        indexed[repeat]={(g['arm'],g['id']):g for g in grades}
    pairs=[]
    for task_id in task_ids:
        entries={a:[indexed[r].get((a,task_id)) for r in (1,2)] for a in ARMS}
        eligible=all(g is not None and g['eligible'] for gs in entries.values() for g in gs)
        values={a:[g['score'] if g else None for g in gs] for a,gs in entries.items()}
        means={a:sum(v)/2 for a,v in values.items()} if eligible else None
        pairs.append(dict(id=task_id,repeat_scores=values,eligible=eligible,mean_success=means,
            deltas={t+'_minus_'+b:means[t]-means[b] for t,b in COMPARISONS} if eligible else None))
    complete=all(p['eligible'] for p in pairs)
    rates={a:sum(p['mean_success'][a] for p in pairs)/len(pairs) for a in ARMS} if complete else None
    result=dict(status='heldout_scored_pending_source_review' if complete else 'blocked_missing_or_unscorable',
        unique_questions=len(task_ids),repeats=2,expected_episodes=len(task_ids)*6,
        received_grades=sum(len(g) for g in repeat_grades.values()),eligible_questions=sum(p['eligible'] for p in pairs),
        comparison_complete=complete,success_rates=rates,
        deltas_percentage_points={t+'_minus_'+b:100*(rates[t]-rates[b]) for t,b in COMPARISONS} if complete else None,
        per_repeat=per_repeat,unit='question; two repeats averaged before comparison',best_repeat_selection=False)
    return pairs,result


def score(run,output,data=ROOT/'data/retail_grpo_heldout_v2'):
    if output.exists():raise FileExistsError(output)
    _,tasks,release=load_public(data)
    reserved=ROOT/release['reserved_manifest_path']
    if sha(reserved)!=release['reserved_commitment_sha256']:raise ValueError('Reserved freeze changed')
    for name,h in json.loads(reserved.read_text())['file_hashes'].items():
        if sha(reserved.parent/name)!=h:raise ValueError('Changed original reserved artifact: '+name)
    expected_dirs={'repeat_1','repeat_2'}
    if {p.name for p in run.glob('repeat_*')}-expected_dirs:raise ValueError('Unexpected repeat directory')
    output.mkdir(parents=True)
    repeat_grades={};repeat_summaries={}
    for r in (1,2):
        source=run/f'repeat_{r}'
        if not source.exists():continue
        if not (source/'rollouts.jsonl').exists():continue
        result=score_repeat(source,output/f'repeat_{r}',data)
        repeat_summaries[r]=result
        repeat_grades[r]=[json.loads(l) for l in (output/f'repeat_{r}/grades.jsonl').read_text().splitlines()]
    pairs,primary=aggregate_repeats([t['id'] for t in tasks],repeat_grades)
    refs=[json.loads(l) for l in (ROOT/'data/retail_policy_eval_v2/private/references.jsonl').read_text().splitlines()]
    families={}
    for operation in sorted({r['operation'] for r in refs}):
        ids=[r['id'] for r in refs if r['operation']==operation]
        subset={repeat:[g for g in grades if g['id'] in ids] for repeat,grades in repeat_grades.items()}
        _,families[operation]=aggregate_repeats(ids,subset)
    total_costs={arm:{k:sum(s['groups'][arm][k] for s in repeat_summaries.values())
                for k in ['invalid_actions','repeated_actions','tool_calls','model_calls','tokens','prompt_tokens','completion_tokens']}
                for arm in ARMS}
    result=dict(version='retail-grpo-heldout-v2',status=primary['status'],primary=primary,by_operation=families,
        per_repeat=repeat_summaries,total_costs=total_costs,cost_comparison_complete=primary['comparison_complete'],
        release_sha256=sha(data/'release.json'),fresh_test=True,weight_updates=0,automatic_promotion=False,
        reserved_question_attempts=primary['received_grades'],
        reserved_model_calls=sum(s['reserved_model_calls'] for s in repeat_summaries.values()),
        limitations=release['heldout_status'])
    write_jsonl(output/'pairs.jsonl',pairs);write_json(output/'summary.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(score(a.run,a.output),indent=2))
