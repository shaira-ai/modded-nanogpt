import torch
import math
import numpy as np
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import transformers

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

def find_valid_next_tokens(text, byte_position, tokenizer, max_length=None):
    """
    Find all tokens that could start at the given byte position.
    """
    remaining_text = text[byte_position:]
    valid_tokens = []
    
    if max_length is None:
        max_length = min(50, len(remaining_text))
    
    # Try all possible substring lengths up to max_length
    for length in range(1, min(len(remaining_text) + 1, max_length + 1)):
        substring = remaining_text[:length]
        tokens = tokenizer.encode(substring, add_special_tokens=False)
        
        if len(tokens) == 1:
            token_id = tokens[0]
            token_text = substring
            valid_tokens.append((token_id, token_text))
    
    return valid_tokens

def generate_canonical_token_positions(document, tokenizer):
    prefixes = [document[:i+1] for i in range(len(document))]
    all_token_data = []

    for prefix in prefixes:
        # Get tokens and their positions
        encoding = tokenizer(prefix, return_offsets_mapping=True, add_special_tokens=False)
        tokens = encoding.input_ids
        offsets = encoding.offset_mapping
        
        # Calculate byte position (END of token) for each token
        for token_idx, (token_id, offset) in enumerate(zip(tokens, offsets)):
            # Skip special tokens if any
            if token_id in tokenizer.all_special_ids:
                continue
                
            token_text = tokenizer.decode([token_id])
            start_pos, end_pos = offset
            # Only add if this is a valid offset
            if start_pos != end_pos:
                byte_position_end = end_pos
                all_token_data.append((token_id, token_text, token_idx, byte_position_end))
    
    unique_token_data = []
    seen = set()
    for token in all_token_data:
        key = (token[0], token[3])  # (token_id, byte_position_end)
        if key not in seen:
            seen.add(key)
            unique_token_data.append(token)
    
    # Sort by byte_position_end first, then by token length
    sorted_token_data = sorted(unique_token_data, 
                              key=lambda x: (x[3], len(x[1])))
    
    return sorted_token_data

def calculate_naive_apbpb(document, tokenizer, model):
    byte_count = len(document.encode('utf-8'))
    separator_token = "<|endoftext|>"

    token_data = generate_canonical_token_positions(document, tokenizer)
    
    df = pd.DataFrame([(token_id, token_text, token_pos, byte_pos_end) 
                      for token_id, token_text, token_pos, byte_pos_end in token_data],
                     columns=['token_id', 'token_text', 'token_position', 'byte_position_end'])
    
    #print("\nSorted by byte position (end) and token length:")
    #print(df.head(10))
    
    token_probs = {}
    
    for token_id, token_text, _, byte_pos_end in token_data:
        byte_pos_start = byte_pos_end - len(token_text)
        prefix = document[:byte_pos_start]
        
        if byte_pos_start == 0:
            input_text = separator_token
        else:
            input_text = separator_token + prefix
        
        inputs = tokenizer(input_text, return_tensors="pt")
        
        with torch.no_grad():
            model._loss_function = ForCausalLMLoss
            outputs = model(**inputs)
            logits = outputs.logits[0, -1]
        
        # Get token probability
        next_token_probs = torch.softmax(logits, dim=0)
        token_probs[(token_id, byte_pos_start)] = next_token_probs[token_id].item()
    
    prob_array = np.zeros(len(document) + 1)
    prob_array[0] = 1.0  # Start with 100% probability
    
    # Process each byte position
    for pos in range(len(document)):
        # Skip positions that can't be reached
        if prob_array[pos] <= 0:
            continue
        
        valid_tokens = find_valid_next_tokens(document, pos, tokenizer)
        
        for token_id, token_text in valid_tokens:
            if (token_id, pos) in token_probs:
                token_prob = token_probs[(token_id, pos)]
                new_pos = pos + len(token_text)
                prob_array[new_pos] += prob_array[pos] * token_prob
    
    final_prob = prob_array[len(document)]
    
    if final_prob > 0:
        apbpb = -math.log2(final_prob) / byte_count
    else:
        apbpb = float('inf')
    
    return apbpb, prob_array, token_data

def calculate_standard_bpb(document, tokenizer, model):
    byte_count = len(document.encode('utf-8'))
    tokens = tokenizer.encode(document, add_special_tokens=False)
    token_count = len(tokens)
    tokens = [50256] + tokens
    
    tokens_tensor = torch.tensor([tokens])
    model._loss_function = ForCausalLMLoss
    with torch.no_grad():
        outputs = model(tokens_tensor, labels=tokens_tensor)
        loss = outputs.loss.item()
    
    bpb = (token_count / byte_count) * loss * 1.44269504
    
    return bpb

def main():
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    
    try:
        import sys
        if len(sys.argv) > 1:
            with open(sys.argv[1], 'r', encoding='utf-8') as f:
                document = f.read()
        else:
            document = "The quick brown fox jumps over the lazy dog."
    except:
        document = "The quick brown fox jumps over the lazy dog."
    
    standard_bpb = calculate_standard_bpb(document, tokenizer, model)

    naive_apbpb, prob_array, token_data = calculate_naive_apbpb(document, tokenizer, model)
    
    print(f"Standard token-wise BPB: {standard_bpb:.6f}")
    print(f"Naive APBPB: {naive_apbpb:.6f}")
    
    print(f"APBPB < BPB? {'Yes' if naive_apbpb < standard_bpb else 'No'}")
    print(f"Final probability: {prob_array[-1]:.8e}")
    
    '''
    print("Showing probability at selected positions:")
    step = max(1, len(document) // 10)
    for i in range(0, len(document) + 1, step):
        pos_text = document[:i] if i > 0 else "<start>"
        print(f"Position {i:3d}: {prob_array[i]:.8e} - '{pos_text}'")
    '''

if __name__ == "__main__":
    main()
