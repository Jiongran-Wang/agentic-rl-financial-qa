"""Frozen order, accumulation and learning-rate policy for the small SFT pilot."""
import math
import random

EPOCHS=2
ACCUMULATION=4
SEED=73
PEAK_LR=5e-5
WARMUP_UPDATES=6


def epoch_groups(size,epoch):
    if size<1 or epoch not in range(1,EPOCHS+1):raise ValueError('Invalid epoch or dataset size')
    order=list(range(size));random.Random(SEED+epoch).shuffle(order)
    return [order[i:i+ACCUMULATION] for i in range(0,size,ACCUMULATION)]


def learning_rate(update,total):
    if not 1<=update<=total or total<=WARMUP_UPDATES:raise ValueError('Invalid update budget')
    if update<=WARMUP_UPDATES:return PEAK_LR*update/WARMUP_UPDATES
    progress=(update-WARMUP_UPDATES)/(total-WARMUP_UPDATES)
    return PEAK_LR*(0.1+0.9*0.5*(1+math.cos(math.pi*progress)))
