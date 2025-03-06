import torch
import math
import sys
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def calculate_perplexity(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.eval()
    
    tokens = tokenizer.encode(text)
    token_count = len(tokens)
    tokens = [50256] + tokens
    byte_count = len(text.encode('utf-8'))
    tokens_per_byte = token_count / byte_count
    
    with torch.no_grad():
        tokens_tensor = torch.tensor([tokens])
        loss = model(tokens_tensor, labels=tokens_tensor).loss.item()
    
    perplexity = math.exp(loss)  # Perplexity = e^ALLT
    bits_per_byte = tokens_per_byte * loss * 1.44269504  # (N/L) * ALLT
    
    return {
        'perplexity': perplexity,
        'bits_per_byte': bits_per_byte,
        'tokens_per_byte': tokens_per_byte,
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_text_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    results = calculate_perplexity(file_path)
    if results:
        print(f"Perplexity: {results['perplexity']:.4f}")
        print(f"Bits per Byte: {results['bits_per_byte']:.4f}")
        print(f"Tokens per Byte: {results['tokens_per_byte']:.4f}")
