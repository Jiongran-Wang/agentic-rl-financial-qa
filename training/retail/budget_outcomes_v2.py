"""Separate failed policies under fixed budgets from unscorable service failures."""
from .policy_compare_v1 import validate_response

VERSION = 'retail-budget-outcomes-v2'
PROMPT_LIMIT = 30720
OUTPUT_LIMIT = 2048
LEGACY_GUARD_ERROR = 'ValueError: Prompt budget exceeded; no history truncation'


def classify(row, model_name, trusted_legacy_guard=False):
    """Legacy guard evidence is accepted only by the hash-pinned migration tool."""
    def outcome(kind, reason):
        return dict(outcome_version=VERSION, outcome=kind, reason=reason,
                    eligible=kind != 'infrastructure_failure',
                    fixed_reward=0.0 if kind == 'policy_budget_failure' else None)
    status = row.get('status')
    episode = row.get('episode', {})
    trace = episode.get('trace', [])
    calls = row.get('calls', [])
    if status == 'complete':
        if not episode.get('done') or len(trace) != len(calls):
            return outcome('infrastructure_failure', 'inconsistent_complete_record')
        return outcome('complete', 'terminal_episode_requires_proof_scoring')
    if status not in {'infrastructure_error', 'budget_exhausted'}:
        return outcome('infrastructure_failure', 'unknown_status')
    if episode.get('prediction') or episode.get('done'):
        return outcome('infrastructure_failure', 'inconsistent_failed_record')
    # A length stop is a policy failure only with a complete, validated response
    # from the intended arm, at the exact output cap, not a timeout or HTTP error.
    if len(calls) == len(trace) + 1:
        response = calls[-1].get('response', {})
        try:
            validate_response(response, model_name, response['prompt_tokens_local'])
        except (KeyError, TypeError, ValueError):
            return outcome('infrastructure_failure', 'invalid_response_contract')
        if (isinstance(response.get('raw'), str) and response['raw'] and
                response.get('finish_reason') == 'length' and
                response['usage']['completion_tokens'] == OUTPUT_LIMIT):
            return outcome('policy_budget_failure', 'output_token_limit')
    if len(calls) == len(trace):
        budget = row.get('budget') or {}
        n = budget.get('prompt_tokens')
        if (status == 'budget_exhausted' and budget.get('kind') == 'prompt_token_limit'
                and type(n) is int and n > PROMPT_LIMIT and budget.get('limit') == PROMPT_LIMIT
                and isinstance(budget.get('messages_sha256'), str)):
            return outcome('policy_budget_failure', 'prompt_token_limit')
        if trusted_legacy_guard and row.get('error') == LEGACY_GUARD_ERROR:
            return outcome('policy_budget_failure', 'prompt_token_limit_legacy_verified_guard')
    return outcome('infrastructure_failure', 'transport_server_or_unverified_failure')
