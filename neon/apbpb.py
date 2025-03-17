import torch
import math
import numpy as np
import pandas as pd
import tiktoken
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import transformers
import mpmath  # For better precision with small numbers
from pre_segment import please_encode

def maybe_utf8_decode(data):
    try:
        return data.decode('utf-8')
    except UnicodeDecodeError:
        return None

def get_token_to_id_mapping(tokenizer):
    return dict(tokenizer._mergeable_ranks)

def prep_tokenizer(tokenizer):
    if not hasattr(tokenizer, "non_special_vocab_xd"):
        tokenizer.non_special_vocab_xd = get_token_to_id_mapping(tokenizer)
    if not hasattr(tokenizer, "max_token_length"):
        vocab_max_token_length = 0
        for k in tokenizer.non_special_vocab_xd:
            vocab_max_token_length = max(vocab_max_token_length, len(k))
        tokenizer.max_token_length = vocab_max_token_length

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
    if type(text) != bytes:
        text = text.encode("utf-8")
    remaining_text = text[byte_position:]
    valid_tokens = []


    if max_length is None:
        max_length = min(tokenizer.max_token_length, len(remaining_text))

    # Try all possible substring lengths up to max_length
    for length in range(1, min(len(remaining_text) + 1, max_length + 1)):
        substring = remaining_text[:length]
        tokens = please_encode(tokenizer, substring)

        if len(tokens) == 1:
            token_id = tokens[0]
            token_text = substring
            valid_tokens.append((token_id, token_text))

    # If this is the end of the string, also try tokens that are longer!
    if max_length == len(remaining_text):
        for k in tokenizer.non_special_vocab_xd:
            if len(k) > max_length and k.startswith(remaining_text):
                valid_tokens.append((tokenizer.non_special_vocab_xd[k], k))

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

def calculate_naive_apbpb(document, encoder, model, device):
    if type(document) != bytes:
        document = document.encode("utf-8")
    byte_count = len(document)
    separator_id = encoder.encode("<|endoftext|>", allowed_special={'<|endoftext|>'})[0]

    # Use mpmath for high precision calculations to avoid underflow
    mpmath.mp.dps = 1000  # Set precision to 100 decimal places

    position_tokens = {}
    for pos in range(len(document)):
        valid_tokens = find_valid_next_tokens(document, pos, encoder)
        position_tokens[pos] = valid_tokens

    prob_array = [mpmath.mpf(0)] * (len(document) + 1 + encoder.max_token_length)
    prob_array[0] = mpmath.mpf(1.0)  # Start with 100% probability

    # Get canonical token positions for analysis/debugging
    # canonical_token_data = generate_canonical_token_positions(document, encoder)

    # Print canonical token positions for debugging
    # print("\nCanonical token positions from prefixes:")
    # for token_id, token_text, token_pos, byte_pos_end in canonical_token_data[:10]:  # Show first 10
    #     print(f"Token: '{token_text}', Position: {token_pos}, End byte: {byte_pos_end}")

    # Calculate token probabilities for each position (one model call per position)
    token_probs = {}
    #string_pos_tuples = []
    for pos in range(len(document)):
        # Prepare input for the model
        input_text = document[:pos]
        inputs = please_encode(encoder, input_text)
        inputs = [separator_id] + inputs
        inputs = [separator_id] + please_encode(encoder, input_text)
        strings = [encoder.decode([x]) for x in inputs]
        inputs = {
            'input_ids': torch.tensor([inputs]).to(device),
            'attention_mask': torch.ones(1, len(inputs), dtype=torch.long).to(device),
        }
        new_tuples = [(pos, string) for pos, string in zip(range(len(strings)), strings)]
        #for x in new_tuples:
        #    if x not in string_pos_tuples:
        #        string_pos_tuples.append(x)

        # Get all token probabilities in one go
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1]
            next_token_probs = torch.softmax(logits, dim=0)

        # Store probabilities for all valid tokens at this position
        for token_id, token_text in position_tokens[pos]:
            token_prob = next_token_probs[token_id].item()
            token_probs[(token_id, pos)] = token_prob
    #for x in string_pos_tuples:
    #    print(x)

    # Process each byte position to update probability array
    for pos in range(len(document)):
        # Skip positions that can't be reached
        if prob_array[pos] <= 0:
            continue

        # Process all valid tokens from this position
        for token_id, token_text in position_tokens[pos]:
            token_prob = token_probs[(token_id, pos)]

            # Update probability array
            new_pos = pos + len(token_text)
            prob_array[new_pos] += prob_array[pos] * mpmath.mpf(token_prob)

            print(f"in position {pos} I can get \"{token_text}\" with probability {token_prob}")

    final_prob = sum(prob_array[len(document):])

    if final_prob > 0:
        apbpb = -mpmath.log(final_prob, 2) / byte_count
    else:
        apbpb = float('inf')

    # Convert to float for printing
    float_prob_array = [float(p) if p > 0 else 0.0 for p in prob_array]

    return float(apbpb), float_prob_array

def calculate_standard_bpb(document, encoder, model, device):
    if type(document) != bytes:
        print(f"document {document} wasn't bytes")
        document = document.encode()
    byte_count = len(document)
    print("byte count: "+str(byte_count))
    separator_id = encoder.encode("<|endoftext|>", allowed_special={'<|endoftext|>'})[0]

    # Add separator token at the beginning - this was missing before
    tokens = [separator_id] + please_encode(encoder, document)
    token_count = len(tokens) - 1  # Subtract 1 for the separator token

    tokens_tensor = torch.tensor([tokens]).to(device)
    model._loss_function = ForCausalLMLoss  # Set the loss function
    #print("standard bpb tokens: "+str(tokens_tensor))
    with torch.no_grad():
        outputs = model(tokens_tensor, labels=tokens_tensor)
        loss = outputs.loss.item()

    bpb = (token_count / byte_count) * loss * math.log2(math.e)

    return bpb

def main():
    # Use tiktoken for GPT-2 tokenization
    encoder = tiktoken.get_encoding("gpt2")
    prep_tokenizer(encoder)
    #print(dir(encoder) )
    #print(repr(encoder._pat_str))

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
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
    document = document.encode("utf-8")

    standard_bpb = calculate_standard_bpb(document, encoder, model, device)

    naive_apbpb, prob_array = calculate_naive_apbpb(document, encoder, model, device)

    print(f"Standard token-wise BPB: {standard_bpb:.6f}")
    print(f"Naive APBPB: {naive_apbpb:.6f}")

    print(f"APBPB < BPB? {'Yes' if naive_apbpb < standard_bpb else 'No'}")
    print(f"Final probability: {sum(prob_array[len(document):]):.8e}")

    # Print selected probabilities
    print("Showing probability at selected positions:")
    step = max(1, len(document) // 10)
    for i in range(0, len(document) + 1, step):
        pos_text = document[:i] if i > 0 else b"<start>"
        pos_text = pos_text.decode("utf-8", "backslashreplace")
        print(f"Position {i:3d}: {prob_array[i]:.8e} - '{pos_text}'")

if __name__ == "__main__":
    main()