# train_contrastive.py

import argparse
import random
import torch

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import (
    ParaphraseDetectionDataset,
    ParaphraseDetectionTestDataset,
    load_paraphrase_data
)
from evaluation import model_eval_paraphrase, model_test_paraphrase
from models.gpt2 import GPT2Model
from optimizer import AdamW

from contrastive_utils import supervised_contrastive_loss_for_paraphrases

TQDM_DISABLE = False

def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class ParaphraseGPT(nn.Module):
    """Your GPT-2 Model designed for paraphrase detection."""

    def __init__(self, args):
        super().__init__()
        self.gpt = GPT2Model.from_pretrained(
            model=args.model_size, 
            d=args.d, 
            l=args.l, 
            num_heads=args.num_heads
        )
        # For binary classification (yes/no)
        self.paraphrase_detection_head = nn.Linear(args.d, 2)

        for param in self.gpt.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        """
        We feed the combined prompt, e.g. "Is s1 a paraphrase of s2? yes/no"
        The GPT-2 last_token is used for classification.
        """
        outputs = self.gpt(input_ids=input_ids, attention_mask=attention_mask)
        last_token = outputs["last_token"]  # shape [batch_size, d]
        logits = self.paraphrase_detection_head(last_token)  # [batch_size, 2]
        return logits

def save_model(model, optimizer, args, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }
    torch.save(save_info, filepath)
    print(f"Model saved to {filepath}")

def train(args):
    """Train GPT-2 for paraphrase detection on the Quora dataset."""
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

    # Load training and dev data
    para_train_data_list = load_paraphrase_data(args.para_train)
    para_dev_data_list   = load_paraphrase_data(args.para_dev)

    # Build dataset and dataloaders
    para_train_data = ParaphraseDetectionDataset(
        para_train_data_list, args, hard_negatives_file=args.hard_negatives_file
    )
    para_dev_data = ParaphraseDetectionDataset(para_dev_data_list, args)

    para_train_dataloader = DataLoader(
        para_train_data,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=para_train_data.collate_fn
    )
    para_dev_dataloader = DataLoader(
        para_dev_data,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=para_dev_data.collate_fn
    )

    # Add model dimension parameters
    args = add_arguments(args)
    model = ParaphraseGPT(args).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.)
    best_dev_acc = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0

        for batch in tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            # Batch dict: "token_ids", "attention_mask", "labels", possibly "sent_ids", "hard_negatives"
            b_ids = batch["token_ids"].to(device)          # [batch_size, seq_len]
            b_mask = batch["attention_mask"].to(device)    # [batch_size, seq_len]
            labels = batch["labels"].flatten().to(device)  # [batch_size]

            # Hard negatives, if provided by your Dataset:
            hard_negatives = batch.get("hard_negatives", None)

            optimizer.zero_grad()
            logits = model(b_ids, b_mask)  # shape [batch_size, 2]

            # Standard cross-entropy loss
            ce_loss = F.cross_entropy(logits, labels, reduction='mean')

            # If you're collecting all anchor embeddings + negative embeddings, do so here
            # (Currently your code lumps s1/s2 into a single prompt, so "anchor_embeddings"
            # is just the embedding from that combined input.)
            # If you want each sample's embedding as an "anchor," you can do:
            anchor_embeddings = model.gpt(b_ids, b_mask)["last_token"]

            if hard_negatives:
                # Build negative embeddings
                # This is the original approach from your snippet:
                #   for each example in the batch, we have a list of (s1, s2, label, sent_id)
                #   then we embed them with GPT-2 (optionally no_grad if you want).
                hard_neg_embeddings = []
                hard_neg_labels = []

                for i, neg_samples in enumerate(hard_negatives):
                    if not neg_samples:
                        continue
                    # neg_samples is something like [(s1_text, s2_text, lbl, sent_id), ...]
                    neg_sent1 = [s[0] for s in neg_samples]
                    neg_sent2 = [s[1] for s in neg_samples]
                    neg_labels = [s[2] for s in neg_samples]

                    # Form a combined prompt for each negative
                    neg_prompts = [
                        f'Question 1: "{s1}"\nQuestion 2: "{s2}"\nAre these questions asking the same thing?\n'
                        for (s1, s2) in zip(neg_sent1, neg_sent2)
                    ]
                    # Tokenize
                    neg_enc = para_train_data.tokenizer(
                        neg_prompts,
                        return_tensors='pt',
                        padding=True,
                        truncation=True
                    )
                    neg_ids = neg_enc["input_ids"].to(device)
                    neg_mask = neg_enc["attention_mask"].to(device)

                    # Embed
                    with torch.no_grad():
                        neg_emb = model.gpt(neg_ids, neg_mask)["last_token"]  # shape [num_neg, d]
                    hard_neg_embeddings.append(neg_emb)
                    hard_neg_labels.extend(neg_labels)

                if hard_neg_embeddings:
                    hard_neg_embeddings = torch.cat(hard_neg_embeddings, dim=0)
                    hard_neg_labels = torch.tensor(hard_neg_labels, dtype=torch.long).to(device)

                    cont_loss = supervised_contrastive_loss(
                        anchor_embeddings,
                        hard_neg_embeddings,
                        labels,
                        hard_neg_labels,
                        temperature=0.07
                    )
                    loss = ce_loss + args.cont_weight * cont_loss
                else:
                    loss = ce_loss
            else:
                # No hard negatives? Then just CE
                loss = ce_loss

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss /= num_batches

        # Evaluate on dev
        dev_acc, dev_f1, *_ = model_eval_paraphrase(para_dev_dataloader, model, device)

        # Save if best
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, args.filepath)

        print(f"Epoch {epoch}: train loss={train_loss:.3f}, dev acc={dev_acc:.3f}")

@torch.no_grad()
def test(args):
    """Evaluate your model on the dev and test sets; save predictions to disk."""
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    saved = torch.load(args.filepath)

    model = ParaphraseGPT(saved['args'])
    model.load_state_dict(saved['model'])
    model = model.to(device)
    model.eval()
    print(f"Loaded model from {args.filepath}")

    para_dev_data_list = load_paraphrase_data(args.para_dev)
    para_test_data_list = load_paraphrase_data(args.para_test, split='test')

    para_dev_data = ParaphraseDetectionDataset(para_dev_data_list, args)
    para_test_data = ParaphraseDetectionTestDataset(para_test_data_list, args)

    para_dev_dataloader = DataLoader(
        para_dev_data, shuffle=False, batch_size=args.batch_size,
        collate_fn=para_dev_data.collate_fn
    )
    para_test_dataloader = DataLoader(
        para_test_data, shuffle=False, batch_size=args.batch_size,
        collate_fn=para_test_data.collate_fn
    )

    dev_para_acc, _, dev_para_y_pred, _, dev_para_sent_ids = model_eval_paraphrase(para_dev_dataloader, model, device)
    print(f"Dev paraphrase acc={dev_para_acc:.3f}")

    test_para_y_pred, test_para_sent_ids = model_test_paraphrase(para_test_dataloader, model, device)

    # To ensure that Gradescope leaderboard autograder processes labels correctly
    label_mapping = {0: 3505, 1: 8919}

    with open(args.para_dev_out, "w", encoding="utf-8") as f:
        f.write("id,Predicted_Is_Paraphrase\n")
        for i, yhat in zip(dev_para_sent_ids, dev_para_y_pred):
            mapped_yhat = label_mapping[yhat]
            f.write(f"{i},{mapped_yhat}\n")

    with open(args.para_test_out, "w", encoding="utf-8") as f:
        f.write("id,Predicted_Is_Paraphrase\n")
        for i, yhat in zip(test_para_sent_ids, test_para_y_pred):
            mapped_yhat = label_mapping[yhat]
            f.write(f"{i},{mapped_yhat}\n")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")
    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")
    parser.add_argument("--hard_negatives_file", type=str, default=None, help="Path to the hard negatives file")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--model_size", type=str,
                        choices=["gpt2","gpt2-medium","gpt2-large"],
                        default="gpt2")
    parser.add_argument("--cont_weight", type=float, default=0.25)
    args = parser.parse_args()
    return args

def add_arguments(args):
    """Sets GPT-2 dimension params based on model size."""
    if args.model_size == "gpt2":
        args.d = 768
        args.l = 12
        args.num_heads = 12
    elif args.model_size == "gpt2-medium":
        args.d = 1024
        args.l = 24
        args.num_heads = 16
    elif args.model_size == "gpt2-large":
        args.d = 1280
        args.l = 36
        args.num_heads = 20
    else:
        raise Exception(f"{args.model_size} not supported")
    return args

if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.epochs}-{args.lr}-paraphrase.pt'
    seed_everything(args.seed)
    train(args)
    test(args)
