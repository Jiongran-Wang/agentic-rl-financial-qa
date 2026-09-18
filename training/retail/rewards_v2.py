"""Bounded numeric equivalence; v1 scores and policy environment remain unchanged."""
from decimal import Decimal, localcontext
import re
import unicodedata
from .environment import canonical, digest
from .rewards import RewardUnavailable, score_numeric
from retail_evidence_v10 import cell_catalog
from retail_challenge_engine import SourceCalculator

VERSION='numeric-proof-v2'
NUMBER=r'[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?'
QUANTITY=re.compile('('+NUMBER+r')(亿元|万元|千元|元/股|元|%)')


def clean(text):
    return unicodedata.normalize('NFKC',text).strip().rstrip('。')


def quantity(text):
    m=QUANTITY.fullmatch(text)
    if not m:return None
    value=Decimal(m[1].replace(',',''));unit=m[2]
    with localcontext() as ctx:
        ctx.prec=60
        if unit in {'亿元','万元','千元'}:
            value*=Decimal({'亿元':100000000,'万元':10000,'千元':1000}[unit]);unit='元'
    return value,unit


def answer_matches(actual,expected,operation,question,companies):
    actual,expected=clean(actual),clean(expected)
    if operation=='larger':return actual==expected
    if operation=='unavailable_as_of':return actual==expected
    prefix=None
    if ';' in actual:
        parts=actual.split(';')
        if len(parts)!=2:return False
        prefix,actual=parts
    if operation=='select_then_lookup':
        parts=expected.split(';')
        if len(parts)!=2 or prefix!=parts[0]:return False
        expected=parts[1]
    elif prefix is not None and (prefix not in companies or prefix not in question):
        return False
    a,b=quantity(actual),quantity(expected)
    # No fuzzy substring matching, negation, new claims or precision tolerance.
    return a is not None and b is not None and a==b


def signature(c):
    return (c['company'],c['period'],c['label'],c['unit'],Decimal(c['value']))


class NumericVerifierV2:
    def __init__(self,corpus,certificates):
        self.corpus=corpus;self.corpus_sha256=digest(corpus)
        self.by_id={c['chunk_id']:c for c in corpus}
        self.catalog={c['cell_id']:c for c in cell_catalog([{'chunk_id':c['chunk_id']} for c in corpus],self.by_id)}
        self.companies={c['company'] for c in corpus}
        if certificates['corpus_sha256']!=self.corpus_sha256:
            raise RewardUnavailable('Cell certificates belong to another corpus')
        self.aliases={}
        for pair in certificates['pairs']:
            a,b=pair['canonical_cell'],pair['alternative_cell']
            if a==b or a not in self.catalog or b not in self.catalog or b in self.aliases:
                raise RewardUnavailable('Invalid or duplicate cell certificate')
            if pair.get('review_status')!='source_reviewed' or not pair.get('scope_review'):
                raise RewardUnavailable('Equivalent source scope has not been reviewed')
            if signature(self.catalog[a])!=signature(self.catalog[b]):
                raise RewardUnavailable('Equivalent cells differ in company, period, metric, unit or value')
            # Pin every involved raw chunk, including table header/unit sources.
            needed=set()
            for i in (a,b):
                c=self.catalog[i];needed.update([c['chunk_id'],c['header']['chunk_id'],c['unit_source']['chunk_id']])
            if set(pair['source_hashes'])!=needed or any(digest(self.by_id[i])!=h for i,h in pair['source_hashes'].items()):
                raise RewardUnavailable('Cell certificate raw sources changed')
            self.aliases[b]=a
        if any(a in self.aliases for a in self.aliases.values()):
            raise RewardUnavailable('Chained source aliases are not supported')
        self.reference_calculator=SourceCalculator(self.by_id)
        self.reference_calculator.cells=self.catalog

    def form(self,e,root=True):
        if set(e)=={'cell_id'}:return {'cell_id':self.aliases.get(e['cell_id'],e['cell_id'])}
        if set(e)=={'constant'}:return e
        if set(e)!={'op','args'}:raise ValueError('Unsupported proof node')
        op=e['op']
        if root and op=='to_yi':return self.form(e['args'][0],root=True)
        if root and op=='percent':op='divide'
        args=[self.form(a,root=False) for a in e['args']]
        if op in {'add','multiply','argmax'}:args.sort(key=canonical)
        return {'op':op,'args':args}

    @staticmethod
    def value(result):
        v,u=Decimal(result['value']),result['unit']
        with localcontext() as ctx:
            ctx.prec=60
            if u=='ratio':return v*100,'%'
            if u=='亿元':return v*100000000,'元'
        return v,u

    def proof_key(self,expression,result):
        v,u=self.value(result)
        # Decimal arithmetic may differ only in final precision when percent
        # is calculated directly vs divide then scale. Compare to 28 decimal
        # places; displayed-answer equality remains exact.
        with localcontext() as ctx:
            ctx.prec=60
            v=v.quantize(Decimal('1e-28'))
        return canonical(self.form(expression)),v,u

    def score(self,env,reference):
        if reference.get('kind')!='numeric_proof_v1' or reference.get('source_check')!='passed':
            raise RewardUnavailable('Reference lacks source consistency check')
        if reference['task_sha256']!=digest(env.task) or reference['corpus_sha256']!=self.corpus_sha256 or env.corpus_sha256!=self.corpus_sha256:
            raise RewardUnavailable('Task or corpus mismatch')
        if not env.done:raise RewardUnavailable('Episode is not terminal')
        def result(ok,reason):return dict(score=float(ok),eligible=True,reason=reason,reward_version=VERSION)
        p=env.prediction
        if not p:return result(False,'no_final')
        if p['action']!=reference['expected_action']:return result(False,'wrong_action')
        if not any(answer_matches(p['answer'],g,reference['operation'],env.task['question'],self.companies) for g in reference['answer_aliases']):
            return result(False,'wrong_or_unsupported_answer')
        if p['action']!='answer':
            r=score_numeric(env,reference);return result(bool(r['score']),r['reason'])
        linked=[env.calculations[i] for i in p['calculation_ids']]
        if not linked:return result(False,'missing_proof')
        if set(p['citations'])!=set().union(*(set(c['citations']) for c in linked)):
            return result(False,'incomplete_or_extraneous_proof_citations')
        actual=sorted(self.proof_key(c['expression'],c) for c in linked)
        try:
            expected=[sorted(self.proof_key(e,self.reference_calculator.evaluate(e)) for e in variant) for variant in reference['proof_variants']]
        except (ValueError,KeyError,ArithmeticError) as exc:
            raise RewardUnavailable('Reference proof cannot be recomputed') from exc
        if actual not in expected:return result(False,'wrong_source_operation_or_result')
        return result(True,'verified_numeric_equivalence')
