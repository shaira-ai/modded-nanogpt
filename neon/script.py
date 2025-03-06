import torch
import math
import sys
import transformers
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def ForCausalLMLoss(
    logits, labels, vocab_size: int, num_items_in_batch: int = None, ignore_index: int = -100, **kwargs
):
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()
    labels = labels.to(logits.device)
    # Shift so that tokens < n predict n
    labels = torch.nn.functional.pad(labels, (0, 1), value=ignore_index)
    shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    logits = logits.view(-1, vocab_size)
    #print(logits)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(logits.device)
    #print(shift_labels)
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    target_probabilities = []
    for idx,x in enumerate(shift_labels):
        if x == -100:
            continue
        target_probabilities.append(probabilities[idx][x].item())
    #print(target_probabilities)
    log_probs = torch.log(torch.tensor(target_probabilities))
    #print(log_probs)
    average_log_prob = log_probs.mean().item()
    sum_log_prob = log_probs.sum().item()
    #print(math.exp(-average_log_prob))
    #print(-sum_log_prob/(44 * math.log(2)))
    loss = transformers.loss.loss_utils.fixed_cross_entropy(logits, shift_labels, num_items_in_batch, ignore_index, **kwargs)
    my_loss = -average_log_prob
    assert(math.fabs((my_loss - loss)/loss) < 0.0001)
    return loss

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
        model._loss_function = ForCausalLMLoss
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
