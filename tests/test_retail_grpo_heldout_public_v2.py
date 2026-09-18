from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from argparse import Namespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from training.retail.grpo_heldout_v2 import ARMS,MODEL_NAMES,SEED,run_episode,schedule,sampling_payload,load_public,arm_metrics
import eval_retail_grpo_heldout_v2 as collector

TASK=dict(id='q',question='可用证据？',as_of=None)
FINAL='{"type":"final","action":"insufficient_evidence","answer":"证据不足","citations":[],"calculation_ids":[]}'

def response(arm,raw=FINAL,finish='stop',tokens=2,prompt=5):
    return dict(raw=raw,finish_reason=finish,served_model=MODEL_NAMES[arm],usage=dict(prompt_tokens=prompt,completion_tokens=tokens,total_tokens=prompt+tokens))


class ComparisonPublicTests(unittest.TestCase):
    def test_each_task_each_arm_once_and_alternating_order(self):
        tasks=[dict(TASK,id=str(i)) for i in range(64)]
        jobs=schedule(tasks)
        self.assertEqual(len(jobs),192)
        self.assertEqual(len({(a,t['id']) for a,t in jobs}),192)
        self.assertEqual([a for a,t in jobs[:6]],['sft','grpo','iterative','grpo','iterative','sft'])

    def test_identical_greedy_request_except_model_and_exact_http_payload(self):
        a=sampling_payload('sft',[],73);b=sampling_payload('grpo',[],73)
        self.assertNotEqual(a.pop('model'),b.pop('model'));self.assertEqual(a,b)
        self.assertEqual((a['temperature'],a['top_p'],a['top_k'],a['max_tokens'],a['seed']),(0.,1.,-1,2048,73))
        output=dict(model=MODEL_NAMES['grpo'],choices=[dict(message={'content':FINAL},finish_reason='stop')],usage=dict(prompt_tokens=5,completion_tokens=2,total_tokens=7))
        with patch.object(collector,'urlopen',return_value=io.BytesIO(json.dumps(output).encode())) as call:
            result=collector.request_greedy('http://unused/v1','grpo',[],73)
        self.assertEqual(json.loads(call.call_args[0][0].data),sampling_payload('grpo',[],73))
        self.assertEqual(result['served_model'],MODEL_NAMES['grpo'])

    def test_length_is_zero_reward_never_executes_partial_output(self):
        row=run_episode([],TASK,'grpo',lambda m:5,lambda m,s:response('grpo','{','length',2048))
        self.assertEqual(row['outcome']['fixed_reward'],0)
        self.assertEqual(row['episode']['trace'],[])
        self.assertEqual(len(row['calls']),1)

    def test_prompt_guard_skips_request_and_boundary_is_allowed(self):
        def forbidden(*args):raise AssertionError('Should not call model')
        row=run_episode([],TASK,'sft',lambda m:30721,forbidden)
        self.assertEqual(row['outcome']['reason'],'prompt_token_limit');self.assertEqual(row['calls'],[])
        row=run_episode([],TASK,'sft',lambda m:30720,lambda m,s:response('sft',prompt=30720))
        self.assertEqual(row['status'],'complete')

    def test_timeout_wrong_arm_and_bad_usage_are_unscorable(self):
        def timeout(*args):raise TimeoutError('timeout')
        wrong=response('sft');bad=response('grpo');bad['usage']['total_tokens']='seven'
        for request in [timeout,lambda m,s:wrong,lambda m,s:bad,lambda m,s:response('grpo','x','length',7)]:
            row=run_episode([],TASK,'grpo',lambda m:5,request)
            self.assertFalse(row['outcome']['eligible']);self.assertIsNone(row['outcome']['fixed_reward'])
            result=arm_metrics([row],1)
            self.assertEqual(result['infrastructure_errors'],1)

    def test_action_budget_failure_is_terminal_with_all_calls_retained(self):
        row=run_episode([],TASK,'sft',lambda m:5,lambda m,s:response('sft','{}'))
        self.assertEqual(row['status'],'complete');self.assertEqual(len(row['calls']),12)
        self.assertIsNone(row['episode']['prediction'])

    def test_public_inputs_are_hash_pinned_and_answer_fields_rejected(self):
        corpus,tasks,release=load_public(ROOT/'data/retail_grpo_heldout_v2')
        self.assertEqual(len(tasks),64)
        with tempfile.TemporaryDirectory() as td:
            data=Path(td)
            for n in ['corpus.json','public.tasks.jsonl','release.json']:
                (data/n).write_bytes((ROOT/'data/retail_grpo_heldout_v2'/n).read_bytes())
            tasks[0]['answer']='private';p=data/'public.tasks.jsonl';p.write_text('\n'.join(map(json.dumps,tasks)))
            with self.assertRaises(ValueError):load_public(data)
            from training.retail.policy_compare_v1 import sha
            release['public_hashes']['public.tasks.jsonl']=sha(p);(data/'release.json').write_text(json.dumps(release))
            with self.assertRaises(ValueError):load_public(data)

    def test_collector_keeps_both_arms_and_budget_failures_without_service_error(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);args=Namespace(repeat_index=1,output=root/'run',preflight_only=False,base_url='http://unused')
            contract=dict(adapters={arm:dict(path=str(root/arm)) for arm in ARMS})
            catalog=dict(data=[dict(id=MODEL_NAMES[a],root=str(root/a)) for a in ARMS])
            with patch.object(collector,'preflight',return_value=([],[TASK],lambda m:5,contract)), \
                 patch.object(collector,'urlopen',return_value=io.BytesIO(json.dumps(catalog).encode())), \
                 patch.object(collector,'request_greedy',side_effect=lambda url,arm,m,s:response(arm,'{','length',2048)):
                collector.run(args)
            summary=json.loads((args.output/'summary.json').read_text())
            self.assertEqual(summary['attempted'],3)
            self.assertTrue(all(g['policy_budget_failures']==1 and g['infrastructure_errors']==0 for g in summary['groups'].values()))

    def test_server_registration_swap_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);args=Namespace(repeat_index=1,output=root/'run',preflight_only=False,base_url='http://unused')
            contract=dict(adapters={arm:dict(path=str(root/arm)) for arm in ARMS})
            catalog=dict(data=[dict(id=MODEL_NAMES[a],root=str(root/'wrong')) for a in ARMS])
            with patch.object(collector,'preflight',return_value=([],[TASK],lambda m:5,contract)), \
                 patch.object(collector,'urlopen',return_value=io.BytesIO(json.dumps(catalog).encode())):
                with self.assertRaises(ValueError):collector.run(args)

if __name__=='__main__':unittest.main()
