import copy
import json
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from test_retail_rl_v1 import TASK,LEAF,final,public_read,complete,reference
from test_retail_v2 import fixture
from training.retail.environment import RetailEpisode,digest
from training.retail.rewards import RewardUnavailable
from training.retail.rewards_v2 import NumericVerifierV2,answer_matches
from training.retail.sft_data import action_sample,tokenize_sample,pad_batch


def verifier(corpus):return NumericVerifierV2(corpus,dict(corpus_sha256=digest(corpus),pairs=[]))
def ref(env,op='lookup'):
    r=reference(env);r['operation']=op;return r


class RewardV2Tests(unittest.TestCase):
    def test_correct_proof_with_relevant_company_prefix(self):
        corpus=fixture();env=complete(RetailEpisode(corpus,TASK),answer='甲超市；100.00元')
        self.assertEqual(verifier(corpus).score(env,ref(env))['score'],1)

    def test_thousands_grouping_and_exact_unit_conversion(self):
        for answer in ['1,234.00元','0.1234万元','甲超市；1234.00元']:
            self.assertTrue(answer_matches(answer,'1234.00元','lookup',TASK['question'],{'甲超市'}))
        for answer in ['12,34.00元','1,23,4.00元','0.12万元','约1234.00元','1234.00元；利润翻倍','乙超市；1234.00元','1234.00元/股']:
            self.assertFalse(answer_matches(answer,'1234.00元','lookup',TASK['question'],{'甲超市'}),answer)

    def test_dependent_lookup_requires_correct_winner_and_value(self):
        q='甲超市与乙超市比较后查询资产'
        self.assertTrue(answer_matches('甲超市；1,234.00元','甲超市；1234.00元','select_then_lookup',q,{'甲超市','乙超市'}))
        for answer in ['乙超市；1234.00元','1234.00元','甲超市；1235.00元']:
            self.assertFalse(answer_matches(answer,'甲超市；1234.00元','select_then_lookup',q,{'甲超市','乙超市'}))

    def test_divide_is_percent_equivalent_only_with_correct_display(self):
        corpus=fixture();env=RetailEpisode(corpus,TASK);public_read(env)
        expr={'op':'divide','args':[{'cell_id':'a_table:r4:c1'},LEAF]}
        cs=env.step(dict(type='calculate',expressions=[expr]))['calculations']
        env.step(final(answer='12.00%',citations=['a_table','a_page'],ids=list(cs)))
        r=ref(env,'ratio');r.update(answer_aliases=['12.00%'],proof_variants=[[dict(expr,op='percent')]])
        self.assertEqual(verifier(corpus).score(env,r)['score'],1)
        env.prediction['answer']='0.12%';self.assertEqual(verifier(corpus).score(env,r)['score'],0)

    def test_wrong_operation_cannot_hide_behind_correct_answer(self):
        corpus=fixture();env=complete(RetailEpisode(corpus,TASK),{'op':'subtract','args':[LEAF,{'cell_id':'a_table:r4:c1'}]})
        self.assertEqual(verifier(corpus).score(env,ref(env))['score'],0)

    def test_same_number_wrong_field_or_period_still_rejected(self):
        corpus=fixture(cash='100')+fixture(doc='old',year='2023')
        env=complete(RetailEpisode(corpus,TASK),{'cell_id':'a_table:r4:c1'})
        self.assertEqual(verifier(corpus).score(env,ref(env))['score'],0)
        env=RetailEpisode(corpus,TASK);public_read(env,'甲超市2023年度营业收入')
        cs=env.step(dict(type='calculate',expressions=[{'cell_id':'old_table:r1:c1'}]))['calculations']
        env.step(final(citations=['old_table','old_page'],ids=list(cs)))
        self.assertEqual(verifier(corpus).score(env,ref(env))['score'],0)

    def test_unknown_equivalent_location_is_not_auto_approved(self):
        corpus=fixture()+fixture(doc='alternate');env=RetailEpisode(corpus,TASK)
        public_read(env);cs=env.step(dict(type='calculate',expressions=[{'cell_id':'alternate_table:r1:c1'}]))['calculations']
        env.step(final(citations=['alternate_table','alternate_page'],ids=list(cs)))
        self.assertEqual(verifier(corpus).score(env,ref(env))['score'],0)

    def certificate(self,corpus,a,b):
        v=verifier(corpus);ids=set()
        for i in (a,b):
            c=v.catalog[i];ids.update([c['chunk_id'],c['header']['chunk_id'],c['unit_source']['chunk_id']])
        return dict(corpus_sha256=digest(corpus),pairs=[dict(canonical_cell=a,alternative_cell=b,review_status='source_reviewed',scope_review='Test fixture same report scope',source_hashes={i:digest(v.by_id[i]) for i in ids})])

    def test_certified_source_equivalence_requires_all_metadata_and_hashes(self):
        corpus=fixture()+fixture(doc='alternate')
        cert=self.certificate(corpus,'a_table:r1:c1','alternate_table:r1:c1')
        v=NumericVerifierV2(corpus,cert);env=RetailEpisode(corpus,TASK);public_read(env)
        cs=env.step(dict(type='calculate',expressions=[{'cell_id':'alternate_table:r1:c1'}]))['calculations']
        env.step(final(citations=['alternate_table','alternate_page'],ids=list(cs)))
        self.assertEqual(v.score(env,ref(env))['score'],1)
        cert['pairs'][0]['source_hashes']['a_table']='tampered'
        with self.assertRaises(RewardUnavailable):NumericVerifierV2(corpus,cert)

    def test_bad_equivalence_certificate_cannot_approve_wrong_year(self):
        corpus=fixture()+fixture(doc='old',year='2023')
        cert=self.certificate(corpus,'a_table:r1:c1','old_table:r1:c1')
        with self.assertRaises(RewardUnavailable):NumericVerifierV2(corpus,cert)

    def test_missing_proof_remains_zero(self):
        corpus=fixture();env=RetailEpisode(corpus,TASK);public_read(env);env.step(final(citations=['a_table']))
        self.assertEqual(verifier(corpus).score(env,ref(env))['score'],0)


class CharacterTokenizer:
    def apply_chat_template(self,messages,tokenize,add_generation_prompt,enable_thinking):
        text=''.join('<'+m['role']+'>'+m['content']+'<end>' for m in messages)
        if add_generation_prompt:text+='<assistant>'
        return [ord(c) for c in text]


class TargetMaskTests(unittest.TestCase):
    def test_failed_actions_and_tool_observations_have_zero_target_loss(self):
        env=RetailEpisode(fixture(),TASK);env.step({'type':'read','chunk_ids':['BAD_ACTION']})
        sample=action_sample(env,dict(type='search',query=TASK['question'],top_k=6),'mask','test')
        row=tokenize_sample(CharacterTokenizer(),sample)
        self.assertTrue(all(x==-100 for x in row['labels'][:row['prefix_tokens']]))
        learned=''.join(chr(x) for x in row['labels'] if x!=-100)
        self.assertNotIn('BAD_ACTION',learned);self.assertTrue(learned.startswith(sample['target']))

    def test_padding_preserves_masks(self):
        rows=[dict(input_ids=[1,2,3],attention_mask=[1,1,1],labels=[-100,2,3]),dict(input_ids=[4,5],attention_mask=[1,1],labels=[-100,5])]
        b=pad_batch(rows,0);self.assertEqual(b['labels'][1],[-100,5,-100]);self.assertEqual(b['attention_mask'][1],[1,1,0])

    def test_no_silent_truncation(self):
        env=RetailEpisode(fixture(),TASK);sample=action_sample(env,{'type':'search'},'test','test')
        with self.assertRaises(ValueError):tokenize_sample(CharacterTokenizer(),sample,max_length=20)

    def test_prefix_changing_template_rejected(self):
        class Bad(CharacterTokenizer):
            def apply_chat_template(self,*a,**kw):
                result=super().apply_chat_template(*a,**kw)
                return [99]+result if kw['add_generation_prompt'] else result
        env=RetailEpisode(fixture(),TASK);sample=action_sample(env,{'type':'search'},'test','test')
        with self.assertRaises(ValueError):tokenize_sample(Bad(),sample)


if __name__=='__main__':unittest.main()
