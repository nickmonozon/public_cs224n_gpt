# generate_less_preferred.py

'''
Use pre-trained, non-fine-tuned GPT2 to compute less preferred pairs for each sonnet in the training set.
'''

import argparse
import re
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm

def load_sonnets(filepath):
    """
    Load full sonnets from a file.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read().strip()
    blocks = re.split(r'\n\s*\n', content)
    blocks = [block.strip() for block in blocks if block.strip()]
    if blocks:
        blocks = blocks[1:]
    sonnets = []
    for block in blocks:
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        if len(lines) >= 4:
            sonnets.append(lines)
    return sonnets

def split_prompt_winner(sonnet_lines, prompt_line_count=3):
    """
    Split the sonnet into two parts:
      - Prompt: the first `prompt_line_count` lines (joined with newline)
      - Winner: the remaining lines (joined with newline)
    """
    prompt = "\n".join(sonnet_lines[:prompt_line_count])
    winner = "\n".join(sonnet_lines[prompt_line_count:])
    return prompt, winner

def generate_completion_fixed_lines(model, tokenizer, prompt, loser_lines=11,
                                    max_length=300, temperature=1.0, top_p=0.9):
    """
    Generate a completion for the given prompt and then ensure that the output is exactly
    `loser_lines` long (newline-delimited)
    """
    encodings = tokenizer(prompt, return_tensors='pt')
    input_ids = encodings['input_ids'].to(model.device)
    attention_mask = encodings['attention_mask'].to(model.device)
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Remove the prompt
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):].strip()
    # Split text
    lines = generated_text.split('\n')
    # Clean up empty strings that may result from split
    lines = [line for line in lines if line.strip() != '']
    if len(lines) < loser_lines:
        # Pad with empty lines if not enough lines were generated
        lines += [''] * (loser_lines - len(lines))
    else:
        lines = lines[:loser_lines]
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sonnet_file", type=str, default="data/sonnets.txt",
                        help="Path to the file containing full sonnets.")
    parser.add_argument("--output_file", type=str, default="data/sonnet_pairs.txt",
                        help="Path to the output file where the generated triplets will be saved.")
    parser.add_argument("--loser_model_name", type=str, default="gpt2",
                        help="Name or path of the model used to generate the less-preferred (loser) completions.")
    parser.add_argument("--prompt_lines", type=int, default=3,
                        help="Number of lines from each sonnet to use as the prompt.")
    parser.add_argument("--max_length", type=int, default=300,
                        help="Maximum token length for generating the loser completion.")
    parser.add_argument("--temperature", type=float, default=1.5,
                        help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Cumulative probability threshold for nucleus (top-p) sampling.")
    parser.add_argument("--use_gpu", action="store_true",
                        help="Use GPU for generation if available.")
    args = parser.parse_args()

    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")

    tokenizer = GPT2Tokenizer.from_pretrained(args.loser_model_name)
    model = GPT2LMHeadModel.from_pretrained(args.loser_model_name).to(device)
    model.eval()

    sonnets = load_sonnets(args.sonnet_file)

    with open(args.output_file, "w", encoding="utf-8") as out_f:
        for sonnet in tqdm(sonnets, desc="Processing Sonnets"):
            prompt, winner = split_prompt_winner(sonnet, prompt_line_count=args.prompt_lines)
            loser = generate_completion_fixed_lines(model, tokenizer, prompt,
                                                     loser_lines=11,
                                                     max_length=args.max_length,
                                                     temperature=args.temperature,
                                                     top_p=args.top_p)
            out_f.write("----- SONNET PAIR -----\n")
            out_f.write("PROMPT:\n" + prompt + "\n")
            out_f.write("WINNER:\n" + winner + "\n")
            out_f.write("LOSER:\n" + loser + "\n\n")
    print(f"Finished generating sonnet pairs in {args.output_file}")

if __name__ == "__main__":
    main()
