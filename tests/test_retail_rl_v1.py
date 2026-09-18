import copy
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'retrieval')]
from test_retail_v2 import fixture
from training.retail.environment import RetailEpisode, digest, replay
from training.retail.rewards import score_numeric, score_semantic, RewardUnavailable
from training.retail.rollout import run_policy, messages_for, to_sft

TASK=dict(id='test_numeric',question='甲超市2024年度营业收入是多少？',as_of=None)
LEAF={'cell_id':'a_table:r1:c1'}


def final(answer='100.00元',citations=None,ids=None,action='answer'):
    return dict(type='final',action=action,answer=answer,citations=citations or [],calculation_ids=ids or [])


def public_read(env, query=TASK['question']):
    found=env.step(dict(type='search',query=query,top_k=4))
    ids=[r['chunk_id'] for r in found['results']]
    return env.step(dict(type='read',chunk_ids=ids))


def complete(env, leaf=None, answer='100.00元'):
    public_read(env)
    result=env.step(dict(type='calculate',expressions=[leaf or LEAF]))['calculations']
    refs=sorted(set().union(*(set(x['citations']) for x in result.values())))
    env.step(final(answer,refs,list(result)))
    return env


def reference(env):
    return dict(kind='numeric_proof_v1',source_check='passed',task_sha256=digest(env.task),
                corpus_sha256=env.corpus_sha256,expected_action='answer',answer_aliases=['100.00元'],
                proof_variants=[[LEAF]],training_approved=False)


class EnvironmentTests(unittest.TestCase):
    def setUp(self): self.corpus=fixture()

    def env(self,**kw): return RetailEpisode(self.corpus,TASK,**kw)

    def test_rich_qa_rows_cannot_become_policy_tasks(self):
        with self.assertRaises(ValueError): RetailEpisode(self.corpus,dict(**TASK,answer='secret'))
        env=self.env()
        self.assertFalse(env.trace)
        self.assertFalse(env.presented)
        self.assertEqual(set(env.initial()['task']),{'id','question','as_of'})

    def test_cutoff_cannot_be_overridden_and_future_ids_cannot_be_read(self):
        env=RetailEpisode(self.corpus,dict(TASK,as_of='2025-03-31'))
        self.assertFalse(env.step(dict(type='search',query='甲超市',top_k=2,as_of='2099-01-01'))['ok'])
        self.assertEqual(env.step(dict(type='search',query='甲超市',top_k=2))['results'],[])
        self.assertFalse(env.step(dict(type='read',chunk_ids=['a_table']))['ok'])

    def test_search_preview_does_not_authorize_citation_or_calculation(self):
        env=self.env();env.step(dict(type='search',query='甲超市2024年度营业收入',top_k=3))
        self.assertFalse(env.step(final(citations=['a_table']))['ok'])
        self.assertFalse(env.step(dict(type='calculate',expressions=[LEAF]))['ok'])

    def test_raw_table_and_header_observation_enable_calculation(self):
        env=self.env();obs=public_read(env)
        self.assertIn('a_table:r1:c1',{c['cell_id'] for c in obs['cell_catalog']})
        calc=env.step(dict(type='calculate',expressions=[LEAF]))['calculations']
        self.assertEqual(next(iter(calc.values()))['value'],'100')
        self.assertEqual(set(next(iter(calc.values()))['citations']),{'a_table','a_page'})

    def test_entire_observation_payload_is_bounded(self):
        env=self.env(observation_chars=1000);obs=public_read(env)
        payload={k:obs[k] for k in ('context','cell_catalog')}
        self.assertLessEqual(len(json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(',',':'))),1000)
        self.assertEqual(set(env.calculator.cells),{c['cell_id'] for c in obs['cell_catalog']})

    def test_fake_cells_and_financial_constants_rejected(self):
        env=self.env();public_read(env)
        for e in [{'cell_id':'a_table:r1:c2'},{'constant':'1234'}, {'quote':{'chunk_id':'a_table'}}]:
            self.assertFalse(env.step(dict(type='calculate',expressions=[e]))['ok'])

    def test_failed_batch_does_not_commit_successful_prefix(self):
        env=self.env();public_read(env)
        self.assertFalse(env.step(dict(type='calculate',expressions=[LEAF,{'cell_id':'fake'}]))['ok'])
        self.assertFalse(env.calculations)

    def test_final_requires_calculation_header_and_unit_citations(self):
        env=self.env();public_read(env)
        ids=list(env.step(dict(type='calculate',expressions=[LEAF]))['calculations'])
        self.assertFalse(env.step(final(citations=['a_table'],ids=ids))['ok'])
        self.assertTrue(env.step(final(citations=['a_table','a_page'],ids=ids))['ok'])

    def test_invalid_actions_consume_budget_and_cannot_loop_forever(self):
        env=self.env(max_steps=3)
        for _ in range(3):obs=env.step({'type':'search','query':'x','top_k':True})
        self.assertTrue(obs['done']);self.assertEqual(env.error,'step_budget_exhausted')
        with self.assertRaises(RuntimeError):env.step({'type':'search'})

    def test_duplicate_json_keys_rejected_and_repair_possible(self):
        env=self.env()
        self.assertFalse(env.step('{"type":"search","type":"read"}')['ok'])
        self.assertTrue(env.step(dict(type='search',query='甲超市',top_k=2))['ok'])

    def test_successful_episode_replays_and_tampering_fails(self):
        env=complete(self.env());record=env.record()
        self.assertEqual(replay(self.corpus,record).record(),record)
        record['trace'][0]['observation']['results'][0]['company']='伪造'
        with self.assertRaises(ValueError):replay(self.corpus,record)

    def test_episode_state_is_isolated(self):
        first=self.env();second=self.env();public_read(first)
        self.assertFalse(second.presented)
        self.assertFalse(second.step(dict(type='calculate',expressions=[LEAF]))['ok'])


class RewardTests(unittest.TestCase):
    def test_correct_answer_requires_correct_proof(self):
        env=complete(RetailEpisode(fixture(),TASK))
        self.assertEqual(score_numeric(env,reference(env))['score'],1)
        # A numerically identical different field is not proof of revenue.
        env=complete(RetailEpisode(fixture(cash='100'),TASK),{'cell_id':'a_table:r4:c1'})
        self.assertEqual(score_numeric(env,reference(env))['score'],0)

    def test_wrong_year_source_cannot_earn_reward_for_same_number(self):
        corpus=fixture()+fixture(doc='old',year='2023',revenue='100')
        env=RetailEpisode(corpus,TASK)
        public_read(env,'甲超市2023年度营业收入')
        calcs=env.step(dict(type='calculate',expressions=[{'cell_id':'old_table:r1:c1'}]))['calculations']
        env.step(final(citations=['old_table','old_page'],ids=list(calcs)))
        self.assertEqual(score_numeric(env,reference(env))['score'],0)

    def test_extra_unverified_claim_or_incorrect_amount_earns_zero(self):
        for answer in ['101.00元','100.00元；盈利翻倍','乙超市100.00元']:
            env=complete(RetailEpisode(fixture(),TASK),answer=answer)
            self.assertEqual(score_numeric(env,reference(env))['score'],0)

    def test_guessed_answer_without_calculation_has_no_success_bonus(self):
        env=RetailEpisode(fixture(),TASK);public_read(env)
        env.step(final(citations=['a_table','a_page']))
        self.assertEqual(score_numeric(env,reference(env))['score'],0)

    def test_reference_mismatch_and_semantic_reward_are_unavailable(self):
        env=complete(RetailEpisode(fixture(),TASK));r=reference(env)
        r['task_sha256']='different'
        with self.assertRaises(RewardUnavailable):score_numeric(env,r)
        with self.assertRaises(RewardUnavailable):score_semantic(env,r)

    def test_zero_reward_and_unavailable_reward_remain_distinct(self):
        env=RetailEpisode(fixture(),TASK,max_steps=2)
        env.step({'type':'invalid'});env.step({'type':'invalid'})
        self.assertEqual(score_numeric(env,reference(env))['score'],0)
        r=reference(env);r['source_check']='failed'
        with self.assertRaises(RewardUnavailable):score_numeric(env,r)

    def test_abstention_reward_requires_verified_scope_and_search(self):
        task=dict(TASK,as_of='2025-03-31');env=RetailEpisode(fixture(),task)
        env.step(dict(type='search',query='甲超市2024年度营业收入',top_k=3))
        env.step(final(answer='证据不足。',action='insufficient_evidence'))
        r=reference(env);r.update(expected_action='insufficient_evidence',answer_aliases=['证据不足。'],proof_variants=[[]])
        self.assertEqual(score_numeric(env,r)['score'],1)

    def test_irrelevant_search_does_not_justify_abstention(self):
        env=RetailEpisode(fixture(),dict(TASK,as_of='2025-03-31'))
        env.step(dict(type='search',query='无关关键词',top_k=3))
        env.step(final(answer='证据不足。',action='insufficient_evidence'))
        r=reference(env);r.update(expected_action='insufficient_evidence',answer_aliases=['证据不足。'],proof_variants=[[]])
        self.assertEqual(score_numeric(env,r)['score'],0)


class RolloutTests(unittest.TestCase):
    def test_transport_and_truncation_never_become_training_failures(self):
        def fail(_): raise TimeoutError('test')
        for fn in [fail,lambda _:{'raw':'{}','finish_reason':'length'}]:
            r=run_policy(fixture(),TASK,fn)
            self.assertEqual(r['status'],'infrastructure_error')
            self.assertFalse(r['episode']['done'])

    def test_repair_history_and_assistant_only_targets_are_preserved(self):
        env=RetailEpisode(fixture(),TASK)
        env.step({'type':'read','chunk_ids':['fake']});complete(env)
        r=reference(env);rollout=dict(status='complete',episode=env.record(),provenance='scripted_test')
        with self.assertRaises(RewardUnavailable):to_sft(fixture(),rollout,r)
        r['training_approved']=True
        exported=to_sft(fixture(),rollout,r)
        self.assertEqual(exported['target_roles'],['assistant'])
        self.assertIn('fake',exported['messages'][2]['content'])
        self.assertEqual(exported['messages'][-1]['role'],'assistant')
        self.assertNotIn('answer_aliases',json.dumps(exported['messages']))
        self.assertEqual(json.loads(exported['messages'][1]['content'])['steps_remaining'],env.max_steps)

    def test_model_request_receives_no_private_reference(self):
        calls=[]
        def request(messages):
            calls.append(messages)
            return dict(raw=json.dumps(final(action='insufficient_evidence',answer='证据不足')),finish_reason='stop')
        r=run_policy(fixture(),TASK,request)
        self.assertEqual(r['status'],'complete')
        self.assertNotIn('proof_variants',json.dumps(calls))


if __name__=='__main__':unittest.main()
