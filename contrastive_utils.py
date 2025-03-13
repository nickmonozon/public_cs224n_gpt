# contrastive_utils.py

'''
Contains helper functions for mining hard negatives and 
computing contrastive loss for paraphrase training examples.
'''

import torch
import torch.nn.functional as F

try:
    from sentence_transformers import SentenceTransformer, util
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    pass


def supervised_contrastive_loss_for_paraphrases(emb_s1: torch.Tensor, emb_s2: torch.Tensor, neg_s1: torch.Tensor, 
                                                neg_s2: torch.Tensor,  labels: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    # Only compute contrastive loss for paraphrase (label=1) samples.
    pos_mask = (labels == 1)
    if pos_mask.sum() == 0:
        # If no positive samples, no contrastive loss
        return torch.tensor(0.0, device=emb_s1.device, dtype=emb_s1.dtype)

    # Filter embeddings
    emb_s1 = emb_s1[pos_mask]
    emb_s2 = emb_s2[pos_mask]
    neg_s1 = neg_s1[pos_mask]
    neg_s2 = neg_s2[pos_mask]

    # Compute cosine similarities for the paraphrase pair
    pos_sim = F.cosine_similarity(emb_s1, emb_s2, dim=1) / temperature

    # Compute cosine similarities for negatives
    # neg_s1, neg_s2 has shape [pos_batch_size, k, hidden_dim]
    neg_sim_s1 = F.cosine_similarity(emb_s1.unsqueeze(1), neg_s1, dim=2) / temperature
    neg_sim_s2 = F.cosine_similarity(emb_s2.unsqueeze(1), neg_s2, dim=2) / temperature

    # Exponentiate all similarities
    pos_sim_exp = pos_sim.exp()  # [pos_batch_size]
    neg_s1_exp = neg_sim_s1.exp().sum(dim=1)  # [pos_batch_size]
    neg_s2_exp = neg_sim_s2.exp().sum(dim=1)  # [pos_batch_size]

    # Denom = exp(sim(pos)) + sum(exp(sim(neg_s1))) + sum(exp(sim(neg_s2)))
    denom = pos_sim_exp + neg_s1_exp + neg_s2_exp

    # Contrastive loss for each example
    loss = -torch.log(pos_sim_exp / denom)

    return loss.mean()


def mine_hard_negatives(train_sentences, pairs, k=4, baseline_paraphrases=None, model_name='all-MiniLM-L6-v2'):
    """
    Mines top-k hard negatives for each sentence index in the dataset using pretrained SBERT.
    Hard negatives are non-paraphrase candidates that are closest in embedding space to the anchor sentences.
    """
    if not HAS_SENTENCE_TRANSFORMERS:
        raise ImportError("Need to install 'sentence_transformers'")
    
    sbert = SentenceTransformer(model_name)
    sentence_embeddings = sbert.encode(train_sentences, convert_to_tensor=True)

    from collections import defaultdict
    
    # Identify true positives for each sentence (from ground truth labels)
    positives_for = defaultdict(set)
    for (i, j, lbl) in pairs:
        if lbl == 1:
            positives_for[i].add(j)
            positives_for[j].add(i)

    # If baseline_paraphrases is None, treat it as empty dict-of-sets
    if baseline_paraphrases is None:
        baseline_paraphrases = defaultdict(set)

    hard_negatives_for = defaultdict(list)

    # For each sentence i, rank all others by similarity and pick top-k
    for i in range(len(train_sentences)):
        sim = util.cos_sim(sentence_embeddings[i], sentence_embeddings).squeeze(0)  # [N]
        sorted_idx = torch.argsort(sim, descending=True)

        found = 0
        for idx in sorted_idx:
            if idx == i:
                continue
            # Skip ground truth positives
            if idx in positives_for[i]:
                continue
            # Skip baseline model paraphrase predictions
            if idx in baseline_paraphrases[i]:
                continue
            # This is a valid negative
            hard_negatives_for[i].append(idx.item())
            found += 1
            if found >= k:
                break

    return hard_negatives_for
