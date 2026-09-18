import copy
import json
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.score_retail_grpo_heldout_v2 import paired_summary,aggregate_repeats,verify_history
from training.retail.grpo_heldout_v2 import ARMS,MODEL_NAMES,run_episode,sampling_payload


def grades(values):
    return [dict(id=i,arm=a,eligible=True,score=values[a][j]) for j,i in enumerate(['q1','q2']) for a in ARMS]


class ThreeArmScoringTests(unittest.TestCase):
    def test_all_three_pairwise_comparisons(self):
        rows=grades(dict(sft=[1,0],grpo=[0,1],iterative=[1,1]))
        _,s=paired_summary(['q1','q2'],rows)
        self.assertEqual(s['success_rates'],dict(sft=.5,grpo=.5,iterative=1.))
        self.assertEqual(s['comparisons']['iterative_minus_sft']['delta_percentage_points'],50.)
        self.assertEqual(s['comparisons']['iterative_minus_grpo']['counts']['target_win'],1)
        self.assertEqual(s['comparisons']['grpo_minus_sft']['delta_percentage_points'],0.)

    def test_repeat_average_and_unique_denominator(self):
        one=grades(dict(sft=[1,0],grpo=[1,0],iterative=[1,1]))
        two=grades(dict(sft=[0,1],grpo=[1,0],iterative=[1,0]))
        pairs,s=aggregate_repeats(['q1','q2'],{1:one,2:two})
        self.assertEqual(s['unique_questions'],2)
        self.assertEqual(s['expected_episodes'],12)
        self.assertEqual(s['success_rates']['iterative'],.75)
        self.assertEqual(s['deltas_percentage_points']['iterative_minus_sft'],25.)
        self.assertEqual(pairs[0]['mean_success']['sft'],.5)
        self.assertFalse(s['best_repeat_selection'])

    def test_missing_or_unscorable_arm_blocks_all_aggregate_rates(self):
        full=grades(dict(sft=[1,0],grpo=[1,0],iterative=[1,1]))
        for kind in ('missing','failure','repeat'):
            bad=copy.deepcopy(full)
            if kind=='missing':bad.pop()
            if kind=='failure':bad[-1].update(eligible=False,score=None)
            inputs={1:full,2:bad} if kind!='repeat' else {1:full}
            _,s=aggregate_repeats(['q1','q2'],inputs)
            self.assertFalse(s['comparison_complete'])
            self.assertIsNone(s['success_rates'])
            self.assertIsNone(s['deltas_percentage_points'])

    def test_duplicate_unknown_nonbinary_and_extra_repeat_rejected(self):
        rows=grades(dict(sft=[1,0],grpo=[1,0],iterative=[1,1]))
        for field,value in [('arm','unknown'),('id','q3'),('score',.5),('eligible','yes')]:
            bad=copy.deepcopy(rows);bad[0][field]=value
            with self.assertRaises(ValueError):paired_summary(['q1','q2'],bad)
        with self.assertRaises(ValueError):paired_summary(['q1','q2'],rows+[rows[0]])
        with self.assertRaises(ValueError):aggregate_repeats(['q1','q2'],{3:rows})

    def test_iterative_history_and_request_identity_tampering(self):
        task=dict(id='q',question='问题',as_of=None)
        raw=json.dumps(dict(type='final',action='insufficient_evidence',answer='证据不足',citations=[],calculation_ids=[]))
        def response(m,s):
            return dict(raw=raw,finish_reason='stop',served_model=MODEL_NAMES['iterative'],usage=dict(prompt_tokens=5,completion_tokens=2,total_tokens=7))
        row=run_episode([],task,'iterative',lambda m:5,response)
        _,out=verify_history([],task,row)
        self.assertTrue(out['eligible'])
        bad=copy.deepcopy(row);bad['calls'][0]['seed']=74
        with self.assertRaises(ValueError):verify_history([],task,bad)
        bad=copy.deepcopy(row);bad['calls'][0]['response']['served_model']=MODEL_NAMES['sft']
        with self.assertRaises(ValueError):verify_history([],task,bad)
        payloads=[sampling_payload(a,[]) for a in ARMS]
        for p in payloads:p.pop('model')
        self.assertEqual(payloads[0],payloads[1]);self.assertEqual(payloads[1],payloads[2])


if __name__=='__main__':unittest.main()
