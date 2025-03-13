#!/usr/bin/env python
# dpo_training.py

import sys
# for training on my computer
sys.stdout.reconfigure(encoding='utf-8')

import argparse
import random
import re
import torch
import torch.nn.functional as F
import numpy as np

from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel
)

from tqdm import tqdm
from optimizer import AdamW

from datasets import SonnetsPreferenceDataset
from dpo_utils import dpo_loss, seed_everything

##############################################################################
#### DPO MODEL
##############################################################################
class PolicyModel(nn.Module):
    """
    Compute log probabilities and generate text
    """
    def __init__(self, model_size='gpt2'):
        super().__init__()
        self.gpt = GPT2LMHeadModel.from_pretrained(model_size)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_size)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def forward(self, input_ids, attention_mask):
        outputs = self.gpt(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

    def compute_logprob_sum(self, input_ids, attention_mask):
        logits = self.forward(input_ids, attention_mask)  # [B, T, V]
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
        sum_log_probs = token_log_probs.sum(dim=1)
        return sum_log_probs

    def generate(self, input_ids, attention_mask=None, temperature=0.7, top_p=0.9, max_new_tokens=128):
        device = input_ids.device
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids).to(device)
        with torch.no_grad():
            generated_ids = self.gpt.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        return generated_ids

##############################################################################
### TRAINING LOOP
##############################################################################
def train_dpo(args):
    seed_everything(args.seed)
    device = torch.device('cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu')

    # Load frozen reference model
    ref_model = PolicyModel(model_size=args.model_size).to(device)
    for p in ref_model.parameters():
        p.requires_grad = False

    # Load trainable policy model
    policy_model = PolicyModel(model_size=args.model_size).to(device)

    dset = SonnetsPreferenceDataset(args.sonnets_pairwise_path, model_name=args.model_size)
    dloader = DataLoader(
        dset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=dset.collate_fn
    )

    optimizer = AdamW(policy_model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        policy_model.train()
        total_loss = 0.0
        for batch_data in tqdm(dloader, desc=f"DPO-epoch-{epoch+1}"):
            for k in batch_data:
                batch_data[k] = batch_data[k].to(device)
            loss = dpo_loss(policy_model, ref_model, batch_data, beta=args.beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dloader)
        print(f"[Epoch {epoch+1}] avg DPO loss: {avg_loss:.4f}")

    if args.save:
        torch.save(policy_model.state_dict(), args.save_path)
        print(f"Saved final DPO-tuned policy model at {args.save_path}")
    return policy_model

##############################################################################
### PARSE THE HELD OUT SONNETS
##############################################################################
def load_sonnets(dev_path):
    """
    Parses the held out file of sonnets.
    """
    with open(dev_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    sonnets = []
    idx = 0
    # Skip header lines until a line is found that is numeric.
    while idx < len(lines):
        if lines[idx].strip().isdigit():
            break
        idx += 1

    # Parse sonnet blocks
    while idx < len(lines):
        num_line = lines[idx].strip()
        if not num_line.isdigit():
            idx += 1
            continue
        sonnet_number = num_line
        idx += 1
        # Skip any blank lines
        while idx < len(lines) and lines[idx].strip() == "":
            idx += 1
        # Extract first 3 non-empty lines as the prompt.
        prompt_lines = []
        while idx < len(lines) and len(prompt_lines) < 3:
            line = lines[idx].rstrip("\n")
            if line.strip():
                prompt_lines.append(line)
            idx += 1
        prompt_text = "\n".join(prompt_lines)
        sonnets.append((sonnet_number, prompt_text))
    return sonnets

##############################################################################
### GENERATE USING HELD OUT SET
##############################################################################
def generate_predictions(policy_model, args, device):
    policy_model.eval()
    sonnet_prompts = load_sonnets(args.held_out_sonnet_path)
    
    with open(args.sonnet_out, "w", encoding="utf-8", errors="replace") as f_out:
        f_out.write("--Generated Sonnets--\n\n")
        for number, prompt in sonnet_prompts:
            encoding = policy_model.tokenizer(prompt, return_tensors="pt", truncation=True)
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)
            
            generated_ids = policy_model.generate(
                input_ids,
                attention_mask=attention_mask,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens
            )
            generated_text = policy_model.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            # Remove the prompt if it is repeated in the generated text
            if generated_text.startswith(prompt):
                completion = generated_text[len(prompt):].strip()
            else:
                completion = generated_text
            # Final output: original sonnet number, the first 3 lines, then the generated continuation.
            full_output = f"{number}\n{prompt}\n{completion}\n\n"
            f_out.write(full_output)
            print(full_output)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sonnets_pairwise_path", type=str, default="data/sonnet_pairs.txt",
                        help="Path to the pairwise training file (with PROMPT, WINNER, LOSER).")
    parser.add_argument("--held_out_sonnet_path", type=str, default="data/sonnets_held_out_dev.txt",
                        help="Path to the dev file containing 11 sonnets (full text).")
    parser.add_argument("--sonnet_out", type=str, default="predictions/generated_sonnets_dev.txt",
                        help="Path to the output file for generated sonnets.")
    parser.add_argument("--model_size", type=str, default='gpt2',
                        choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'],
                        help="Which GPT-2 variant to use.")
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.1,
                        help="Temperature factor in the DPO formula.")
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--save_path", type=str, default="dpo_policy_model.pt",
                        help="Path to save the trained policy model.")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="Maximum new tokens to generate in dev completions.")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature for generation.")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p (nucleus) sampling threshold for generation.")
    parser.add_argument("--save", type=bool, default=False,
                        help="If the model will be saved or not.")
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device('cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu')
    policy_model = train_dpo(args)
    if args.held_out_sonnet_path:
        print("Generating predictions on held out set...")
        generate_predictions(policy_model, args, device)

if __name__ == "__main__":
    main()
