#!/usr/bin/env python3
"""
Dataset classes for Quora paraphrase detection + optional Shakespeare sonnets dataset.
"""
import csv
import re
import pickle
import torch

from torch.utils.data import Dataset
from transformers import GPT2Tokenizer


def preprocess_string(s):
    return ' '.join(s.lower()
                    .replace('.', ' .')
                    .replace('?', ' ?')
                    .replace(',', ' ,')
                    .replace('\'', ' \'')
                    .split())


class ParaphraseDetectionDataset(Dataset):
    """
    Loads Quora paraphrase data and (optionally) a dictionary of precomputed hard negatives:
      hard_negatives[i] = [list of negative example indices for anchor i].
    """
    def __init__(self, dataset, args, hard_negatives_file=None):
        """
        dataset: List of tuples (s1, s2, label, sent_id)
        args: Argparse or similar config
        hard_negatives_file: path to pickle with { i: [neg_idx1, neg_idx2, ...], ... }
        """
        self.dataset = dataset
        self.args = args

        # GPT-2 tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # If a hard-negatives file is provided, load it
        if hard_negatives_file is not None:
            with open(hard_negatives_file, 'rb') as f:
                self.hard_negatives = pickle.load(f)
            print(f"Loaded hard negatives mapping from {hard_negatives_file}")
        else:
            self.hard_negatives = None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Returns a tuple: (anchor_sample, hard_neg_samples)
          anchor_sample = (s1, s2, label, sent_id)
          hard_neg_samples = [ (s1_neg, s2_neg, label_neg, sent_id_neg), ... ]
        """
        anchor_sample = self.dataset[idx]

        if self.hard_negatives is not None:
            neg_indices = self.hard_negatives.get(idx, [])
            hard_neg_samples = [self.dataset[i] for i in neg_indices]
        else:
            hard_neg_samples = None

        return anchor_sample, hard_neg_samples

    def collate_fn(self, batch):
        """
        batch: list of ( (s1, s2, label, sent_id), [ (s1_neg, s2_neg, label_neg, sent_id_neg), ... ] )
        """
        anchor_samples = [x[0] for x in batch]  # list of anchor tuples
        all_hard_negs  = [x[1] for x in batch]  # list of lists

        # Tokenize the anchor samples
        sent1 = [s[0] for s in anchor_samples]
        sent2 = [s[1] for s in anchor_samples]
        labels = [s[2] for s in anchor_samples]
        sent_ids = [s[3] for s in anchor_samples]

        # Build a simple GPT-2 prompt
        cloze_style_sents = [
            f'Question 1: "{s1}"\nQuestion 2: "{s2}"\nAre these questions asking the same thing?\n'
            for (s1, s2) in zip(sent1, sent2)
        ]
        encoding = self.tokenizer(
            cloze_style_sents, return_tensors='pt',
            padding=True, truncation=True
        )

        token_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        labels = torch.tensor(labels, dtype=torch.long)

        # NOTE: We do *not* tokenize the hard negatives here by default, because
        #       you might want to handle them differently (e.g., do a separate
        #       forward pass, contrastive loss, etc. in the training loop).
        #       But if you prefer, you can tokenize them right here.

        batch_out = {
            'token_ids': token_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'sent_ids': sent_ids,
            'hard_negatives': all_hard_negs  # raw text samples for each anchor
        }
        return batch_out


class ParaphraseDetectionTestDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def collate_fn(self, all_data):
        sent1 = [x[0] for x in all_data]
        sent2 = [x[1] for x in all_data]
        sent_ids = [x[2] for x in all_data]

        cloze_style_sents = [
            f'Is "{s1}" a paraphrase of "{s2}"? Answer "yes" or "no": '
            for (s1, s2) in zip(sent1, sent2)
        ]
        encoding = self.tokenizer(
            cloze_style_sents, return_tensors='pt',
            padding=True, truncation=True
        )

        token_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        batched_data = {
            'token_ids': token_ids,
            'attention_mask': attention_mask,
            'sent_ids': sent_ids
        }
        return batched_data


def load_paraphrase_data(paraphrase_filename, split='train'):
    """
    Returns a list of:
      train/dev: (sentence1, sentence2, is_duplicate_label, id)
      test: (sentence1, sentence2, id)
    """
    paraphrase_data = []
    if split == 'test':
        with open(paraphrase_filename, 'r', encoding="utf-8") as fp:
            for record in csv.DictReader(fp, delimiter='\t'):
                sent_id = record['id'].lower().strip()
                s1 = preprocess_string(record['sentence1'])
                s2 = preprocess_string(record['sentence2'])
                paraphrase_data.append((s1, s2, sent_id))
    else:
        with open(paraphrase_filename, 'r', encoding="utf-8") as fp:
            for record in csv.DictReader(fp, delimiter='\t'):
                try:
                    sent_id = record['id'].lower().strip()
                    s1 = preprocess_string(record['sentence1'])
                    s2 = preprocess_string(record['sentence2'])
                    lbl = int(float(record['is_duplicate']))
                    paraphrase_data.append((s1, s2, lbl, sent_id))
                except:
                    pass

    print(f"Loaded {len(paraphrase_data)} {split} examples from {paraphrase_filename}")
    return paraphrase_data


class SonnetsDataset(Dataset):
  def __init__(self, file_path):
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    self.tokenizer.pad_token = self.tokenizer.eos_token
    self.sonnets = self._load_sonnets(file_path)

  def _load_sonnets(self, file_path):
    """Reads the file and extracts individual sonnets."""
    with open(file_path, 'r', encoding='utf-8') as f:
      text = f.read()

    # Split sonnets based on numbering pattern (e.g., "\n\n1\n\n")
    sonnets = re.split(r'\n\s*\d+\s*\n', text)[1:]  # Remove header text

    # Strip leading/trailing spaces
    return [s.strip() for s in sonnets]

  def __len__(self):
    return len(self.sonnets)

  def __getitem__(self, idx):
    return (idx, self.sonnets[idx])

  def collate_fn(self, all_data):
    idx = [example[0] for example in all_data]
    sonnets = [example[1] for example in all_data]

    encoding = self.tokenizer(sonnets, return_tensors='pt', padding=True, truncation=True)
    token_ids = torch.LongTensor(encoding['input_ids'])
    attention_mask = torch.LongTensor(encoding['attention_mask'])

    batched_data = {
      'token_ids': token_ids,
      'attention_mask': attention_mask,
      'sent_ids': idx
    }

    return batched_data
  
class SonnetsPreferenceDataset(Dataset):
    """
    A dataset of (prompt, winning completion, losing completion) for DPO.
    The input file should have blocks separated by:
        ----- SONNET PAIR -----
    with fields labeled as follows:
        PROMPT:
        <prompt text>
        WINNER:
        <winner text>
        LOSER:
        <loser text>
    """
    def __init__(self, filepath, model_name='gpt2', max_length=256):
        super().__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length

        self.samples = []
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Split file by the marker
        blocks = re.split(r'-----\s*SONNET PAIR\s*-----', content)
        blocks = [block.strip() for block in blocks if block.strip()]

        for block in blocks:
            prompt_match = re.search(r'PROMPT:\s*(.*?)\s*WINNER:', block, re.DOTALL)
            winner_match = re.search(r'WINNER:\s*(.*?)\s*LOSER:', block, re.DOTALL)
            loser_match = re.search(r'LOSER:\s*(.*)', block, re.DOTALL)
            if prompt_match and winner_match and loser_match:
                prompt_str = prompt_match.group(1).strip()
                winner_str = winner_match.group(1).strip()
                loser_str = loser_match.group(1).strip()
                self.samples.append((prompt_str, winner_str, loser_str))
            else:
                continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        prompt_str, winner_str, loser_str = self.samples[idx]
        return prompt_str, winner_str, loser_str

    def collate_fn(self, batch):
        prompts = []
        winners = []
        losers = []
        for prompt_str, winner_str, loser_str in batch:
            prompts.append(prompt_str)
            winners.append(winner_str)
            losers.append(loser_str)

        winner_encodings = self.tokenizer(
            [p + "\n" + w for p, w in zip(prompts, winners)],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        loser_encodings = self.tokenizer(
            [p + "\n" + l for p, l in zip(prompts, losers)],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        batch_out = {
            'winner_ids': winner_encodings['input_ids'],
            'winner_mask': winner_encodings['attention_mask'],
            'loser_ids': loser_encodings['input_ids'],
            'loser_mask': loser_encodings['attention_mask']
        }
        return batch_out