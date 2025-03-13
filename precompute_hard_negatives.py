# precompute_hard_negatives.py

'''
Precompute the hard negatives for training samples to avoid recomputing them at each training iteration.

Example usage: python precompute_hard_negatives.py --output_file "hard_negatives_training.pkl" --k 4 --subset_size 1000
'''

import argparse
import pickle
from datasets import load_paraphrase_data
from contrastive_utils import mine_hard_negatives

def precompute_hard_negatives_full(train_file, output_file, k=4, subset_size=None):
    
    # Load training data
    data = load_paraphrase_data(train_file, split='train')
    print(f"Loaded {len(data)} training examples from {train_file}")

    # If subset (for parallel implementation)
    if subset_size is not None:
        data = data[:subset_size]
        print(f"Using subset of {len(data)} training examples")

    # Data should be list of (s1, s2, label, id)
    # Flatten sentences into a single list
    all_sentences = []
    index_map = {}  # (row_id, 's1') -> global_idx, ...

    for ex_idx, (s1, s2, label, sent_id) in enumerate(data):
        i_s1 = len(all_sentences)
        all_sentences.append(s1)
        index_map[(ex_idx, 's1')] = i_s1
        
        i_s2 = len(all_sentences)
        all_sentences.append(s2)
        index_map[(ex_idx, 's2')] = i_s2

    # Build pairs on individual sentence indices
    pairs = []
    for ex_idx, (s1, s2, label, sent_id) in enumerate(data):
        i_s1 = index_map[(ex_idx, 's1')]
        i_s2 = index_map[(ex_idx, 's2')]
        pairs.append((i_s1, i_s2, label))

    # Mine hard negatives
    print(f"Computing hard negatives among {len(all_sentences)} sentences, with {len(pairs)} pairs of interest...")

    hard_negatives = mine_hard_negatives(
        train_sentences=all_sentences,
        pairs=pairs,
        k=k
    )

    # Save the hard negatives mapping to a pickle
    with open(output_file, 'wb') as f:
        pickle.dump(hard_negatives, f)
    print(f"Hard negatives mapping saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="data/quora-train.csv",
                        help="Path to the training CSV file.")
    parser.add_argument("--output_file", type=str, default="hard_negatives_full.pkl",
                        help="File to save the hard negatives mapping.")
    parser.add_argument("--k", type=int, default=5,
                        help="Number of hard negatives to retrieve for each example.")
    parser.add_argument("--subset_size", type=int, default=1000,
                        help="Number of training examples to use. Note that full dataset computation requirements parallelization.")
    args = parser.parse_args()
    
    precompute_hard_negatives_full(args.train_file, args.output_file, args.k, subset_size=args.subset_size)