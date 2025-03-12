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
    """
    Arguments:
        emb_s1:     [batch_size, hidden_dim] embedding of the first sentence
        emb_s2:     [batch_size, hidden_dim] embedding of the second sentence
        neg_s1:     [batch_size, k, hidden_dim] hard negatives for emb_s1
        neg_s2:     [batch_size, k, hidden_dim] hard negatives for emb_s2
        labels:     [batch_size] with binary labels 
        temperature: Scalar temperature to scale the cosine similarities.
        
    Returns:
        A scalar tensor representing the average contrastive loss over all paraphrase examples.
        If there are no positive (label=1) examples in the batch, returns 0.
    """

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
    # neg_s1, neg_s2 has shape [pos_batch_size, k, hidden_dim].
    neg_sim_s1 = F.cosine_similarity(emb_s1.unsqueeze(1), neg_s1, dim=2) / temperature
    neg_sim_s2 = F.cosine_similarity(emb_s2.unsqueeze(1), neg_s2, dim=2) / temperature

    # Exponentiate all similarities
    pos_sim_exp = pos_sim.exp()  # [pos_batch_size]
    neg_s1_exp = neg_sim_s1.exp().sum(dim=1)  # [pos_batch_size]
    neg_s2_exp = neg_sim_s2.exp().sum(dim=1)  # [pos_batch_size]

    # Denominator = exp(sim(pos)) + sum(exp(sim(neg_s1))) + sum(exp(sim(neg_s2)))
    denom = pos_sim_exp + neg_s1_exp + neg_s2_exp

    # Contrastive loss for each example
    loss = -torch.log(pos_sim_exp / denom)

    return loss.mean()


def combined_loss(logits: torch.Tensor, labels: torch.Tensor, emb_s1: torch.Tensor, emb_s2: torch.Tensor, neg_s1: torch.Tensor, 
                  neg_s2: torch.Tensor, paraphrase_labels: torch.Tensor, lambda_: float = 1.0, temperature: float = 0.1) -> torch.Tensor:
    
    # Cross-entropy loss term
    ce_loss = F.cross_entropy(logits, labels)

    # Supervised contrastive loss (only for paraphrases)
    contrastive = supervised_contrastive_loss_for_paraphrases(emb_s1, emb_s2, neg_s1, neg_s2, paraphrase_labels, temperature=temperature)

    return ce_loss + lambda_ * contrastive


def mine_hard_negatives(train_sentences, pairs, k=4, baseline_paraphrases=None, model_name='all-MiniLM-L6-v2'):
    """
    Mines top-k hard negatives for each sentence index in the dataset using pretrained SBERT.
    Hard negatives are defined as non-paraphrase candidates that are closest in embedding space to the anchor sentences.

    Arguments:
        train_sentences:       List of all sentences in the dataset (indexed by i).
        pairs:                 List of (i, j, label) where label in {0, 1}.
        k:                     Number of hard negatives per sentence index.
        baseline_paraphrases:  Optional dictionary mapping each sentence index i to a set of indices that the baseline model predicts as paraphrases of i.

    Returns:
        A dictionary mapping each sentence index i -> [list of top-k hardest negative indices]
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
