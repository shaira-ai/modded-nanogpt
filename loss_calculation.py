import torch
import math
import numpy as np
import tiktoken
import argparse
import os
import uuid
import mpmath
import traceback
from pathlib import Path

# Import the custom model
from model import GPT, Hyperparameters, init_model, device, get_window_size_blocks

# Fallback implementation for pre_segment
def please_encode(tokenizer, text):
    if isinstance(text, bytes):
        text = text.decode('utf-8', errors='replace')
    return tokenizer.encode(text)

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

def find_valid_next_tokens(text, byte_position, tokenizer, max_length=None):
    """Find all valid tokens that can follow the text at the given byte position"""
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

    # If this is the end of the string, also try tokens that are longer
    if max_length == len(remaining_text):
        for k in tokenizer.non_special_vocab_xd:
            if len(k) > max_length and k.startswith(remaining_text):
                valid_tokens.append((tokenizer.non_special_vocab_xd[k], k))

    return valid_tokens

def calculate_standard_bpb(document, encoder, model, device):
    """Calculate standard token-wise bits per byte"""
    if type(document) != bytes:
        document = document.encode("utf-8")
    
    byte_count = len(document)
    print(f"Document byte count: {byte_count}")
    
    # Get the separator token ID
    separator_id = encoder.encode("<|endoftext|>", allowed_special={'<|endoftext|>'})[0]
    
    # Tokenize the document
    tokens = [separator_id] + please_encode(encoder, document)
    token_count = len(tokens) - 1  # Subtract 1 for the separator token
    print(f"Document contains {token_count} tokens")
    
    if token_count == 0:
        print("Warning: No tokens found in document")
        return float('inf')
    
    # Convert to tensors
    input_tensor = torch.tensor(tokens[:-1], dtype=torch.int32).to(device)
    target_tensor = torch.tensor(tokens[1:], dtype=torch.int64).to(device)
    
    # Get window blocks for attention
    window_blocks = get_window_size_blocks(1000)
    
    # Get logits from model
    try:
        with torch.no_grad():
            model.eval()
            # In inference mode, we use the logits to calculate loss
            logits = model(input_tensor, window_blocks)
            
            # Calculate cross-entropy loss manually
            loss = 0
            for i in range(len(input_tensor)):
                token_logits = logits[0, i]
                target = target_tensor[i]
                
                # Apply log softmax to get log probabilities
                log_probs = torch.nn.functional.log_softmax(token_logits, dim=0)
                
                # Get negative log likelihood for this token
                if target < log_probs.size(0):
                    loss -= log_probs[target].item()
                else:
                    print(f"Warning: Target token {target} out of range")
            
            # Average loss
            loss_value = loss / len(input_tensor)
        
        # Calculate bits per byte
        bpb = (token_count / byte_count) * loss_value * math.log2(math.e)
        return bpb
    
    except Exception as e:
        print(f"Error calculating standard BPB: {e}")
        traceback.print_exc()
        return float('inf')

def calculate_naive_apbpb(document, encoder, model, device):
    """Calculate all-paths bits per byte using the naive algorithm"""
    if type(document) != bytes:
        document = document.encode("utf-8")
    
    byte_count = len(document)
    
    # Get separator token
    separator_id = encoder.encode("<|endoftext|>", allowed_special={'<|endoftext|>'})[0]
    
    # Use mpmath for high precision
    mpmath.mp.dps = 1000
    
    # Find all valid tokens at each position
    position_tokens = {}
    total_valid_tokens = 0
    
    for pos in range(len(document)):
        valid_tokens = find_valid_next_tokens(document, pos, encoder)
        position_tokens[pos] = valid_tokens
        total_valid_tokens += len(valid_tokens)
        
        if pos % 10 == 0:
            print(f"Position {pos}: found {len(valid_tokens)} valid tokens")
    
    print(f"Total valid tokens found: {total_valid_tokens}")
    
    # Initialize probability array
    prob_array = [mpmath.mpf(0)] * (len(document) + 1 + encoder.max_token_length)
    prob_array[0] = mpmath.mpf(1.0)  # Start with 100% probability
    
    # Calculate token probabilities for each position
    token_probs = {}
    
    for pos in range(len(document)):
        if prob_array[pos] <= 0:
            continue  # Skip unreachable positions
            
        # Prepare input for the model
        input_text = document[:pos]
        inputs = [separator_id] + please_encode(encoder, input_text)
        inputs_tensor = torch.tensor(inputs, dtype=torch.int32).to(device)
        
        try:
            # Get token probabilities
            with torch.no_grad():
                model.eval()
                
                # For inference, just pass input_seq (model will create ModelOutput)
                outputs = model(inputs_tensor)
                
                # Extract logits for the last token
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits[0, -1]
                elif isinstance(outputs, torch.Tensor) and outputs.dim() >= 2:
                    logits = outputs[0, -1]
                else:
                    raise ValueError(f"Unexpected output format: {type(outputs)}")
                
                # Convert to probabilities
                next_token_probs = torch.nn.functional.softmax(logits, dim=0)
            
            # Store probabilities for valid tokens
            for token_id, token_text in position_tokens[pos]:
                if token_id < next_token_probs.size(0):
                    token_prob = next_token_probs[token_id].item()
                    token_probs[(token_id, pos)] = token_prob
                    
                    # Update probability array
                    new_pos = pos + len(token_text)
                    prob_array[new_pos] += prob_array[pos] * mpmath.mpf(token_prob)
                    
                    print(f"Position {pos}, token '{token_text.decode('utf-8', 'replace')}': prob={token_prob:.6f}")
                    
        except Exception as e:
            print(f"Error at position {pos}: {str(e)}")
            traceback.print_exc()
            # Continue with other positions
    
    # Calculate final probability and APBPB
    final_prob = sum(prob_array[len(document):])
    
    print(f"Final probability sum: {float(final_prob):.8e}")
    
    if final_prob > 0:
        apbpb = -mpmath.log(final_prob, 2) / byte_count
    else:
        apbpb = float('inf')
    
    # Convert to float array for easier handling
    float_prob_array = [float(p) if p > 0 else 0.0 for p in prob_array]
    
    return float(apbpb), float_prob_array

def load_model_from_checkpoint(checkpoint_path):
    """Load a GPT model from a checkpoint file"""
    print(f"Loading model from {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
            
            # Infer model dimensions
            if 'embed.weight' in state_dict:
                vocab_size, model_dim = state_dict['embed.weight'].shape
            else:
                print("Cannot determine model dimensions from state dict")
                return None
            
            # Count layers
            num_layers = 0
            while f'blocks.{num_layers}.lambdas' in state_dict:
                num_layers += 1
            
            # Determine number of heads
            num_heads = None
            for i in range(num_layers):
                key = f'blocks.{i}.attn.qkv_w'
                if key in state_dict:
                    _, hdim, _ = state_dict[key].shape
                    num_heads = hdim // 128  # Assuming head_dim=128
                    break
            
            # Find sequence length from rotary embeddings
            max_seq_len = None
            for i in range(num_layers):
                key = f'blocks.{i}.attn.rotary.cos'
                if key in state_dict:
                    max_seq_len = state_dict[key].shape[0]
                    break
            
            if max_seq_len is None:
                max_seq_len = 128  # Default value
            
            print(f"Inferred model config: dim={model_dim}, layers={num_layers}, heads={num_heads}, seq_len={max_seq_len}")
            
            # Create model with correct configuration
            config = Hyperparameters(
                vocab_size=vocab_size,
                num_layers=num_layers,
                num_heads=num_heads,
                model_dim=model_dim,
                max_seq_len=max_seq_len
            )
            
            model = init_model(config)
            
            # Load weights
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            
            print("Model loaded successfully")
            return model
        else:
            print("Checkpoint does not contain model state dict")
            return None
    
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description="Calculate bits per byte metrics")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--text", type=str, default="The quick brown fox jumps over the lazy dog.", 
                      help="Text to evaluate")
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument("--skip_standard", action="store_true", help="Skip standard BPB calculation")
    parser.add_argument("--skip_apbpb", action="store_true", help="Skip APBPB calculation")
    parser.add_argument("--debug", action="store_true", help="Enable additional debug output")
    args = parser.parse_args()
    
    # Initialize tokenizer
    encoder = tiktoken.get_encoding("gpt2")
    prep_tokenizer(encoder)
    
    print(f"Using device: {device}")
    print(f"Evaluating: {args.text[:30]}...")
    
    # Load model
    if args.checkpoint:
        model = load_model_from_checkpoint(args.checkpoint)
        if model is None:
            print("Failed to load model. Exiting.")
            return
    else:
        print("No checkpoint provided. Using default model.")
        model = init_model()
    
    # Test model with a simple input
    print("Testing model inference...")
    try:
        test_input = torch.tensor([50256, 1], dtype=torch.int32, device=device)
        with torch.no_grad():
            test_output = model(test_input)
        print(f"✓ Model test successful")
        if hasattr(test_output, 'logits'):
            print(f"  Logits shape: {test_output.logits.shape}")
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        traceback.print_exc()
        print("Attempting to continue anyway...")
    
    # Convert text to bytes
    document = args.text.encode("utf-8")
    print(f"Processing {len(document)} bytes of text")
    
    # Results storage
    results = {
        "text": args.text,
        "standard_bpb": None,
        "apbpb": None,
        "comparison": None,
        "position_probs": []
    }
    
    # Calculate standard BPB
    if not args.skip_standard:
        print("\n=== Calculating Standard Token-wise BPB ===")
        try:
            standard_bpb = calculate_standard_bpb(document, encoder, model, device)
            results["standard_bpb"] = standard_bpb
            print(f"Standard token-wise BPB: {standard_bpb:.6f}")
        except Exception as e:
            print(f"Failed to calculate standard BPB: {e}")
            traceback.print_exc()
            results["standard_bpb"] = float('inf')
    
    # Calculate APBPB
    if not args.skip_apbpb:
        print("\n=== Calculating All-Paths BPB (APBPB) ===")
        try:
            apbpb, prob_array = calculate_naive_apbpb(document, encoder, model, device)
            results["apbpb"] = apbpb
            print(f"All-paths BPB (APBPB): {apbpb:.6f}")
            
            # Store position probabilities
            for i in range(0, len(document) + 1, max(1, len(document) // 10)):
                pos_text = document[:i].decode('utf-8', errors='replace') if i > 0 else "<start>"
                prob = prob_array[i]
                results["position_probs"].append((i, prob, pos_text))
                if args.debug:
                    print(f"Position {i:3d}: {prob:.8e} - '{pos_text}'")
        except Exception as e:
            print(f"Failed to calculate APBPB: {e}")
            traceback.print_exc()
            results["apbpb"] = float('inf')
    
    # Compare results
    if results["standard_bpb"] is not None and results["apbpb"] is not None:
        print("\n=== Comparison ===")
        print(f"Standard BPB: {results['standard_bpb']:.6f}")
        print(f"APBPB:        {results['apbpb']:.6f}")
        is_better = results["apbpb"] < results["standard_bpb"]
        results["comparison"] = is_better
        print(f"APBPB < BPB?  {'Yes' if is_better else 'No'}")
    
    # Save results
    if args.output:
        output_path = args.output
    else:
        os.makedirs("logs", exist_ok=True)
        output_path = f"logs/bpb_results_{uuid.uuid4()}.txt"
    
    with open(output_path, "w") as f:
        f.write(f"Text: {results['text']}\n\n")
        
        if results["standard_bpb"] is not None:
            f.write(f"Standard BPB: {results['standard_bpb']:.6f}\n")
        
        if results["apbpb"] is not None:
            f.write(f"APBPB: {results['apbpb']:.6f}\n")
        
        if results["comparison"] is not None:
            f.write(f"APBPB < BPB? {'Yes' if results['comparison'] else 'No'}\n\n")
        
        # Write probability distribution
        if results["position_probs"]:
            f.write("Position probabilities:\n")
            for pos, prob, text in results["position_probs"]:
                f.write(f"Position {pos:3d}: {prob:.8e} - '{text}'\n")
    
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()