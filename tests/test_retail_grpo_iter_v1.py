import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.retail.grpo_iter_v1 import seed_for, verify_parent, select_candidate, collect_episode
from training.retail.policy_compare_v1 import sha


class IterativeContracts(unittest.TestCase):
    def test_driver_orders_fresh_rounds_and_stops_on_failure(self):
        from scripts.run_retail_grpo_iter_v1 import run
        import subprocess
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); data=root/'data'; data.mkdir(); (data/'release.json').write_text('{}')
            a=SimpleNamespace(model=root/'model',model_manifest=root/'model.json',data=data,output=root/'result')
            a.output.mkdir()
            calls=[]
            def fake(command,**kwargs):
                r=int(command[command.index('--round')+1]);out=Path(command[command.index('--output')+1]);out.mkdir()
                kind='train' if 'train_retail' in command[1] else 'eval'
                calls.append((kind,r))
                if kind=='eval':
                    (out/'summary.json').write_text(json.dumps(dict(round=r,status='development_complete',tasks=27,
                        unscorable=0,successes=10+r,invalid_actions=10,tokens=1000)))
                elif r>1:
                    self.assertIn(str(a.output/f'round_{r-1}/adapter_final'),command)
                    self.assertIn(str(a.output/f'round_{r-1}/summary.json'),command)
            with patch('scripts.run_retail_grpo_iter_v1.subprocess.run',side_effect=fake): run(a)
            self.assertEqual(calls,[('eval',0),('train',1),('eval',1),('train',2),('eval',2),('train',3),('eval',3)])
            self.assertEqual(json.loads((a.output/'selection.json').read_text())['selected']['round'],3)
            a.output=root/'failed';a.output.mkdir()
            with patch('scripts.run_retail_grpo_iter_v1.subprocess.run',side_effect=subprocess.CalledProcessError(1,'failed')):
                with self.assertRaises(subprocess.CalledProcessError): run(a)
            self.assertFalse((a.output/'selection.json').exists())

    def test_fresh_seed_schedule_and_invalid_rounds(self):
        seeds = [seed_for(r, f't{t}', i, s) for r in range(1,4) for t in range(26) for i in range(4) for s in range(12)]
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertEqual(seed_for(2,'a',0,0), seed_for(2,'a',0,0))
        for r,i,s in [(0,0,0),(4,0,0),(1,4,0),(1,0,12)]:
            with self.assertRaises(ValueError): seed_for(r,'a',i,s)

    def test_selection_requires_all_rounds_and_strict_primary_gain(self):
        rows = [dict(round=i,status='development_complete',tasks=27,unscorable=0,
                     successes=10,invalid_actions=10-i,tokens=1000-i) for i in range(4)]
        self.assertEqual(select_candidate(rows)['round'],0)
        rows[1]['successes']=11; rows[2]['successes']=11
        self.assertEqual(select_candidate(rows)['round'],2)
        rows[1]['invalid_actions']=rows[2]['invalid_actions']
        rows[1]['tokens']=rows[2]['tokens']
        self.assertEqual(select_candidate(rows)['round'],1)
        for change in ('missing','failed','unscorable','duplicate'):
            bad=copy.deepcopy(rows)
            if change=='missing': bad.pop()
            elif change=='failed': bad[2]['status']='failed'
            elif change=='unscorable': bad[2]['unscorable']=1
            else: bad[2]['round']=1
            with self.assertRaises(ValueError): select_candidate(bad)

    def test_parent_and_anchor_identity_reject_stale_checkpoints(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); anchor=root/'sft'; adapter=root/'round_1/adapter_final'
            anchor.mkdir(); adapter.mkdir(parents=True)
            for folder,content in [(anchor,b'initial'),(adapter,b'updated')]:
                (folder/'adapter_model.safetensors').write_bytes(content)
                (folder/'adapter_config.json').write_text('{}')
            hashes=lambda folder:{p.name:sha(p) for p in folder.iterdir()}
            release=dict(adapter_hashes=hashes(anchor),_sha256='release')
            verify_parent(anchor,anchor,1,release)
            parent=adapter.parent/'summary.json'
            record=dict(status='fresh_round_updates_and_reload_passed',round=1,
                        release_sha256='release',optimizer_updates=2,checkpoint_files=hashes(adapter))
            parent.write_text(json.dumps(record))
            verify_parent(adapter,anchor,2,release,parent)
            with self.assertRaises(ValueError): verify_parent(adapter,anchor,3,release,parent)
            (adapter/'adapter_model.safetensors').write_bytes(b'tampered')
            with self.assertRaises(ValueError): verify_parent(adapter,anchor,2,release,parent)
            (anchor/'adapter_model.safetensors').write_bytes(b'wrong reference')
            with self.assertRaises(ValueError): verify_parent(anchor,anchor,1,release)

    def test_budget_is_zero_but_service_error_aborts_and_labels_are_hidden(self):
        task=dict(id='x',question='收入？',as_of='2025-06-30')
        class Verifier:
            def score(self,*a): raise AssertionError('Budget must not invoke scorer')
        secret='PRIVATE_ANSWER_SENTINEL'
        def sampler(messages,seed):
            self.assertNotIn(secret,json.dumps(messages))
            return dict(kind='output_token_limit',prompt_ids=[2],generated_ids=[3],raw='partial')
        result=collect_episode([],task,0,sampler,Verifier(),dict(answer=secret),2)
        self.assertEqual(result['grade']['score'],0)
        self.assertTrue(result['grade']['eligible'])
        self.assertEqual(result['calls'][0]['response']['generated_ids'],[3])
        def failure(*a): raise RuntimeError('service failure')
        with self.assertRaises(RuntimeError): collect_episode([],task,0,failure,Verifier(),{},2)


if __name__=='__main__': unittest.main()
