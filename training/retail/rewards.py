"""Private numeric proof rewards. Unsupported semantic scoring is unavailable, never zero."""
from .environment import canonical, digest
from retail_scoped import query_scope


class RewardUnavailable(RuntimeError):
    """Abort/mark an unscorable rollout; do not teach the policy that infrastructure failure is failure."""


def expression_key(expr):
    """Permit commutative add/multiply order; retain order for all other operations."""
    if isinstance(expr, dict) and set(expr) == {'op', 'args'}:
        args = [expression_key(e) for e in expr['args']]
        if expr['op'] in {'add', 'multiply'}:
            args.sort()
        return canonical({'op': expr['op'], 'args': args})
    return canonical(expr)


def score_numeric(env, reference):
    if reference.get('kind') != 'numeric_proof_v1' or reference.get('source_check') != 'passed':
        raise RewardUnavailable('No source-checked numeric reference; semantic reward is not validated')
    if reference.get('task_sha256') != digest(env.task) or reference.get('corpus_sha256') != env.corpus_sha256:
        raise RewardUnavailable('Reward reference does not belong to this public task/corpus')
    if not env.done:
        raise RewardUnavailable('Reward requires a terminal episode')
    metrics = dict(steps=env.steps, tool_errors=sum(not t['observation']['ok'] for t in env.trace),
                   searches=sum(t['observation']['ok'] and 'results' in t['observation'] for t in env.trace),
                   presented_chunks=len(env.presented), reward_version='numeric_proof_v1')
    def result(ok, reason):
        return dict(score=float(ok), eligible=True, reason=reason, **metrics)
    p = env.prediction
    if p is None:
        return result(False, 'no_final')
    if p['action'] != reference['expected_action']:
        return result(False, 'wrong_action')
    # Exact short answers are deliberate in this first numeric pilot. Free-form
    # narrative text cannot sneak an unverified assertion beside a correct number.
    norm = lambda x: x.strip().rstrip('。')
    if norm(p['answer']) not in {norm(x) for x in reference['answer_aliases']}:
        return result(False, 'answer_not_in_verified_numeric_aliases')
    if p['action'] != 'answer':
        if p['citations'] or p['calculation_ids']:
            return result(False, 'unsupported_claim_in_abstention')
        requested=query_scope(env.task['question'],env.corpus)
        matching=False
        for t in env.trace:
            a=t['action']
            if isinstance(a,str):
                from retail_challenge_engine import parse_object
                try:a=parse_object(a)
                except (ValueError,TypeError):continue
            if t['observation']['ok'] and a.get('type')=='search':
                scope=query_scope(a['query'],env.corpus)
                matching |= (bool(requested['companies']) and bool(requested['periods'])
                             and set(scope['companies'])==set(requested['companies'])
                             and set(scope['periods'])==set(requested['periods']))
        return result(matching, 'verified_unavailable_scope_requires_matching_search')
    linked = [env.calculations[i] for i in p['calculation_ids']]
    actual = sorted(expression_key(c['expression']) for c in linked)
    allowed = [sorted(expression_key(e) for e in variant) for variant in reference['proof_variants']]
    if not linked or actual not in allowed:
        return result(False, 'wrong_or_missing_source_calculation_proof')
    needed = set().union(*(set(c['citations']) for c in linked))
    if set(p['citations']) != needed:
        return result(False, 'citations_do_not_match_complete_proof')
    return result(True, 'verified_answer_and_source_calculation')


def score_semantic(*args, **kwargs):
    raise RewardUnavailable('Narrative reward unavailable: the current reviewer has documented false acceptances')


def require_trainable(reference):
    if reference.get('training_approved') is not True:
        raise RewardUnavailable('Draft/source-consistency audit is not training approval')
