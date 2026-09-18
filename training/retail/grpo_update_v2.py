"""Versioned, bounded behavior correction for the untouched 3161778 batch."""
import math
from .grpo_update_v1 import objective as uncorrected_objective

VERSION = 'retail-grpo-update-v2'
CORRECTION_CAP = 2.0
REPLAY_MAX_ABS = 1e-4
REPLAY_MEAN_ABS = 1e-5


def check_replay(actual, expected):
    if len(actual) != len(expected) or not actual:
        raise ValueError('Teacher replay lengths differ')
    if not all(math.isfinite(x) for x in list(actual) + list(expected)):
        raise ValueError('Nonfinite teacher replay')
    delta = [abs(x-y) for x, y in zip(actual, expected)]
    if max(delta) > REPLAY_MAX_ABS or sum(delta)/len(delta) > REPLAY_MEAN_ABS:
        raise ValueError('Starting teacher policy differs from diagnostic reference')
    return max(delta)


def correction(old_logps, behavior_logps):
    import torch
    if old_logps.ndim != 1 or old_logps.numel() == 0 or old_logps.shape != behavior_logps.shape:
        raise ValueError('Behavior/teacher probability arrays differ')
    if not torch.isfinite(old_logps).all() or not torch.isfinite(behavior_logps).all():
        raise ValueError('Nonfinite behavior/teacher probabilities')
    raw = (old_logps.detach() - behavior_logps.detach()).exp()
    if not torch.isfinite(raw).all() or (raw <= 0).any():
        raise ValueError('Invalid importance weights')
    return raw.clamp(max=CORRECTION_CAP), raw


def objective(logps, old_logps, ref_logps, behavior_logps, advantage, beta=.02, clip=.2):
    """Multiply the complete v1 token surrogate + K3 penalty by detached old/behavior.

    PPO ratio remains current/old. This is a sampled-context token correction,
    not an exact trajectory-distribution correction or an unbiased KL estimator.
    """
    import torch
    weights, raw = correction(old_logps, behavior_logps)
    losses, stats = uncorrected_objective(logps, old_logps, ref_logps, advantage, beta, clip)
    losses = losses * weights
    if not torch.isfinite(losses).all():
        raise ValueError('Nonfinite corrected loss')
    stats.update(uncorrected_kl=stats['kl'], kl=stats['kl'] * weights,
                 correction=weights, correction_raw=raw, correction_capped=raw > CORRECTION_CAP)
    return losses, stats
