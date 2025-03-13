# dpo_utils.py

"""
Contains helper functions for implementing DPO on sonnet generation.
"""

import torch
import torch.nn.functional as F
import numpy as np
import random

def dpo_loss(policy_model, ref_model, batch, beta=0.1):
    '''
    DPO loss according to https://arxiv.org/abs/2305.18290
    '''
    with torch.no_grad():
        ref_logprob_winner = ref_model.compute_logprob_sum(batch['winner_ids'], batch['winner_mask'])
        ref_logprob_loser  = ref_model.compute_logprob_sum(batch['loser_ids'], batch['loser_mask'])
    policy_logprob_winner = policy_model.compute_logprob_sum(batch['winner_ids'], batch['winner_mask'])
    policy_logprob_loser  = policy_model.compute_logprob_sum(batch['loser_ids'], batch['loser_mask'])
    difference = (policy_logprob_winner - ref_logprob_winner) - (policy_logprob_loser - ref_logprob_loser)
    logits = beta * difference
    loss = -F.logsigmoid(logits).mean()
    return loss

def seed_everything(seed=11711):
    '''
    Seed function for reproducibility
    '''
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)