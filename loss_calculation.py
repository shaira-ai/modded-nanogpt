# apbpb_calculation.py
import os
import torch
import argparse
import math
import numpy as np
import tiktoken
import uuid
import mpmath
import torch.nn.functional as F
from pathlib import Path

# Set up device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

def get_token_to_id_mapping(tokenizer):
    """Get mapping from token bytes to token IDs."""
    return dict(tokenizer._mergeable_ranks)

def prep_tokenizer(tokenizer):
    """Prepare tokenizer with additional attributes for APBPB calculation."""
    if not hasattr(tokenizer, "non_special_vocab_xd"):
        tokenizer.non_special_vocab_xd = get_token_to_id_mapping(tokenizer)
    if not hasattr(tokenizer, "max_token_length"):
        vocab_max_token_length = 0
        for k in tokenizer.non_special_vocab_xd:
            vocab_max_token_length = max(vocab_max_token_length, len(k))
        tokenizer.max_token_length = vocab_max_token_length
    print(f"Max token length: {tokenizer.max_token_length}")
    print(f"Vocabulary size: {len(tokenizer.non_special_vocab_xd)}")

def load_model(checkpoint_path):
    """Load model from checkpoint file."""
    print(f"Loading model from {checkpoint_path}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Import model class (delayed import to avoid module execution)
        from model import GPT, Hyperparameters
        
        # Initialize model architecture
        args = Hyperparameters()
        model = GPT(
            vocab_size=args.vocab_size,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            model_dim=args.model_dim,
            max_seq_len=args.val_seq_len
        )
        
        # Load state dict with relaxed constraints
        model.load_state_dict(checkpoint['model'], strict=False)
        print("Model loaded successfully")
        
        # Move to device and set to evaluation mode
        model = model.to(device)
        model.eval()
        
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def find_valid_next_tokens(text_bytes, byte_position, tokenizer):
    """Find all valid tokens that can start at a given byte position."""
    remaining_bytes = text_bytes[byte_position:]
    valid_tokens = []
    
    # Convert bytes to string for printable characters
    try:
        remaining_str = remaining_bytes.decode('utf-8', errors='replace')
    except:
        remaining_str = str(remaining_bytes)
    
    # Try each token in the vocabulary
    for token_bytes, token_id in tokenizer.non_special_vocab_xd.items():
        if remaining_bytes.startswith(token_bytes):
            valid_tokens.append((token_id, token_bytes))
    
    return valid_tokens

def get_token_probabilities(context_bytes, model, tokenizer):
    """Get next token probabilities from model."""
    # Handle empty context specially
    if len(context_bytes) == 0:
        # For empty context, use a special token or just get probabilities for the first token
        context_tokens = [50256]  # <|endoftext|> token for GPT-2
    else:
        # Convert bytes to string for tiktoken
        context_str = context_bytes.decode('utf-8', errors='replace')
        # Tokenize with tiktoken
        context_tokens = tokenizer.encode(context_str)
    
    context_tensor = torch.tensor(context_tokens, dtype=torch.int32).to(device)
    
    # Capture logits
    logits_list = []
    
    def hook_fn(module, input, output):
        if isinstance(output, torch.Tensor):
            logits_list.append(output.detach().clone())
        return output
    
    # Register hook on lm_head
    handle = model.lm_head.register_forward_hook(hook_fn)
    
    # Forward pass
    with torch.no_grad():
        try:
            # First try with return_logits=True
            _ = model(context_tensor, return_logits=True)
        except:
            try:
                # Fall back to standard call
                _ = model(context_tensor)
            except Exception as e:
                print(f"Error in model forward pass: {e}")
                handle.remove()
                return None
    
    # Clean up hook
    handle.remove()
    
    # Process captured logits
    if logits_list and len(logits_list[0]) > 0:
        logits = logits_list[0]
        
        # Check tensor dimensions and handle each case
        if logits.dim() > 2:  # [B, T, V]
            if logits.size(1) > 0:  # Check if time dimension has at least 1 element
                logits = logits[0, -1]
            else:
                print(f"Warning: Empty time dimension in logits tensor with shape {logits.shape}")
                return None
        elif logits.dim() == 2:  # [T, V]
            if logits.size(0) > 0:  # Check if time dimension has at least 1 element
                logits = logits[-1]
            else:
                print(f"Warning: Empty time dimension in logits tensor with shape {logits.shape}")
                return None
        elif logits.dim() == 1:  # [V]
            # Already a single vector of logits
            pass
        else:
            print(f"Warning: Unexpected logits tensor with shape {logits.shape}")
            return None
        
        # Get probabilities
        probs = F.softmax(logits, dim=-1)
        return probs
    
    print("Warning: No logits captured from model")
    return None


def calculate_bpb(text, model, tokenizer):
    """Calculate standard bits-per-byte metric."""
    if isinstance(text, str):
        text_bytes = text.encode('utf-8')
    else:
        text_bytes = text
    
    byte_count = len(text_bytes)
    print(f"Evaluating {byte_count} bytes of text")
    
    # Tokenize input text
    tokens = tokenizer.encode(text)
    total_loss = 0.0
    
    # Evaluate token by token
    for i in range(len(tokens) - 1):
        context = tokens[:i+1]
        next_token = tokens[i+1]
        
        # Convert to tensor
        context_tensor = torch.tensor(context, dtype=torch.int32).to(device)
        
        # Capture logits using a hook
        logits_list = []
        
        def hook_fn(module, input, output):
            # Capture logits before any activation
            if isinstance(output, torch.Tensor):
                logits_list.append(output.detach().clone())
            return output
        
        # Register forward hook on lm_head
        handle = model.lm_head.register_forward_hook(hook_fn)
        
        # Run the model with the correct arguments
        with torch.no_grad():
            try:
                # Call the model with return_logits=True
                _ = model(context_tensor, return_logits=True)
            except Exception as e:
                print(f"Error during model forward pass: {e}")
                # Try with just context_tensor
                try:
                    _ = model(context_tensor)
                except Exception as e:
                    print(f"Second attempt failed: {e}")
                    handle.remove()
                    continue
            
        # Clean up the hook
        handle.remove()
        
        # Process captured logits
        if logits_list:
            logits = logits_list[0]
            
            # Extract logits for the last position
            if logits.dim() > 2:  # Handle case of [B, T, V]
                logits = logits[0, -1]
            elif logits.dim() == 2 and logits.size(0) > 1:
                logits = logits[-1]
            
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Get probability of next token
            prob = probs[next_token].item()
            
            # Calculate loss
            token_loss = -math.log2(max(prob, 1e-10))
            total_loss += token_loss
            
            if (i+1) % 5 == 0 or i == len(tokens) - 2:
                print(f"Processed {i+1}/{len(tokens)-1} tokens")
        else:
            print(f"Warning: Failed to capture logits for position {i}")
            continue
    
    # Calculate BPB
    bpb = total_loss / byte_count
    return bpb

def calculate_apbpb(text, model, tokenizer, verbose=False):
    """Calculate All-Paths Bits-Per-Byte metric."""
    # Ensure text is bytes
    if isinstance(text, str):
        text_bytes = text.encode('utf-8')
    else:
        text_bytes = text
    
    byte_count = len(text_bytes)
    print(f"Calculating APBPB for {byte_count} bytes of text")
    
    # High precision for calculations
    mpmath.mp.dps = 100
    
    # Initialize probability array with zeros except for position 0
    prob_array = [mpmath.mpf(0)] * (byte_count + 1 + tokenizer.max_token_length)
    prob_array[0] = mpmath.mpf(1.0)  # Start with 100% probability at position 0
    
    # Track valid tokens at each position
    position_tokens = {}
    valid_count = 0
    
    # Find all valid tokens at each position
    for pos in range(byte_count):
        tokens = find_valid_next_tokens(text_bytes, pos, tokenizer)
        position_tokens[pos] = tokens
        valid_count += len(tokens)
        if verbose or pos % 10 == 0:
            print(f"Position {pos}: found {len(tokens)} valid tokens")
    
    print(f"Total valid tokens found: {valid_count}")
    
    # Calculate token probabilities
    token_probs = {}
    positions_processed = 0
    
    # For each position with non-zero probability
    for pos in range(byte_count):
        # Skip positions that can't be reached
        if float(prob_array[pos]) <= 0:
            continue
        
        positions_processed += 1
        
        # Skip if no valid tokens
        if not position_tokens[pos]:
            continue
        
        # Get probabilities for next tokens
        context_bytes = text_bytes[:pos]
        probs = get_token_probabilities(context_bytes, model, tokenizer)
        
        if probs is None:
            print(f"Warning: Failed to get token probabilities at position {pos}")
            continue
        
        # Update probability for each valid token
        for token_id, token_bytes in position_tokens[pos]:
            if token_id < probs.size(0):
                token_prob = probs[token_id].item()
                token_probs[(token_id, pos)] = token_prob
                
                # Update probability array
                new_pos = pos + len(token_bytes)
                prob_array[new_pos] += prob_array[pos] * mpmath.mpf(token_prob)
                
                if verbose:
                    print(f"Position {pos}: token {token_id} ({token_bytes}) with prob {token_prob:.6f}")
        
        if positions_processed % 5 == 0:
            print(f"Processed {positions_processed} positions with non-zero probability")
    
    # Calculate final probability and APBPB
    final_prob = sum(prob_array[byte_count:])
    final_prob_float = float(final_prob)
    
    print(f"Final probability: {final_prob_float:.10e}")
    
    if final_prob > 0:
        apbpb = -float(mpmath.log(final_prob, 2)) / byte_count
    else:
        apbpb = float('inf')
    
    # Convert probability array to floats for reporting
    float_prob_array = [float(p) if p > 0 else 0.0 for p in prob_array]
    
    return apbpb, float_prob_array

def main():
    parser = argparse.ArgumentParser(description="Calculate APBPB for a trained model")
    parser.add_argument("--checkpoint", type=str, default="logs/0809cf1a-7c34-45c5-8d69-1c595d387cdc/state_step000100.pt", help="Path to model checkpoint file")
    parser.add_argument("--text", type=str, default="The quick brown fox jumps over the lazy dog.", help="Text to evaluate")
    parser.add_argument("--file", type=str, default=None, help="File containing text to evaluate")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")
    args = parser.parse_args()
    
    # Get evaluation text
    if args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                eval_text = f.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            eval_text = args.text
    else:
        eval_text = args.text
    
    print(f"Evaluating: {eval_text[:50]}...")
    
    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    prep_tokenizer(tokenizer)
    
    # Load model
    model = load_model(args.checkpoint)
    if model is None:
        print("Failed to load model. Exiting.")
        return
    
    # Calculate APBPB
    apbpb, prob_array = calculate_apbpb(eval_text, model, tokenizer, verbose=args.verbose)
    print(f"All-paths bits per byte (APBPB): {apbpb:.6f}")
    
    # Show probabilities at key positions
    print("\nShowing probabilities at key positions:")
    byte_text = eval_text.encode('utf-8')
    step = max(1, len(byte_text) // 10)
    for i in range(0, len(byte_text) + 1, step):
        pos_text = byte_text[:i] if i > 0 else b"<start>"
        try:
            pos_str = pos_text.decode("utf-8", "backslashreplace")
        except:
            pos_str = str(pos_text)
        print(f"Position {i:3d}: {prob_array[i]:.8e} - '{pos_str}'")
    
    # Log results
    run_id = uuid.uuid4()
    log_path = f"logs/apbpb_{run_id}.txt"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    with open(log_path, "w") as f:
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Text: {eval_text[:100]}...\n")
        f.write(f"APBPB: {apbpb:.6f}\n")
        f.write(f"Final probability: {sum(prob_array[len(byte_text):]):.10e}\n")
    
    print(f"Results saved to {log_path}")

if __name__ == "__main__":
    main()