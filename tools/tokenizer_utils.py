import tiktoken
import torch
import numpy as np

def find_all_tokens(text, token_dict):
    """
    Optimized token finder using length-based filtering with special handling for <|endoftext|> tokens.
    
    Args:
        text: Input text as string or bytes
        token_dict: Dictionary mapping token bytes to token IDs
        separator_id: The ID for the special <|endoftext|> token (default: 50256)
        
    Returns:
        List of tokens with positions and IDs
    """
    if isinstance(text, str):
        text = text.encode("utf-8")
    
    separator_id=50256
    # Find the byte representation of <|endoftext|> in the text
    eot_pattern = b"<|endoftext|>"
    eot_positions = []
    
    # Find all occurrences of <|endoftext|> in the text
    pos = 0
    while True:
        pos = text.find(eot_pattern, pos)
        if pos == -1:
            break
        eot_positions.append(pos)
        pos += len(eot_pattern)
    
    # Process the text normally to find all tokens
    valid_tokens = []
    text_length = len(text)
    
    # Pre-group tokens by first byte for faster filtering
    tokens_by_first_byte = {}
    max_token_length = 0
    
    for token_bytes, token_id in token_dict.items():
        if token_bytes:
            first_byte = token_bytes[0:1]  # Get first byte
            if first_byte not in tokens_by_first_byte:
                tokens_by_first_byte[first_byte] = []
            tokens_by_first_byte[first_byte].append((token_bytes, token_id))
            max_token_length = max(max_token_length, len(token_bytes))
    
    # Track positions to skip (where <|endoftext|> tokens are)
    skip_ranges = []
    for pos in eot_positions:
        skip_ranges.append((pos, pos + len(eot_pattern)))
    
    # Find all valid tokens, skipping the <|endoftext|> positions
    start_pos = 0
    while start_pos < text_length:
        # Check if this position is inside an <|endoftext|> region
        skip = False
        for skip_start, skip_end in skip_ranges:
            if skip_start <= start_pos < skip_end:
                skip = True
                start_pos = skip_end  # Skip to end of <|endoftext|>
                break
        
        if skip:
            continue
            
        # Get first byte at this position
        if start_pos >= text_length:
            break
            
        current_byte = text[start_pos:start_pos+1]
        
        # Only check tokens that start with this byte
        if current_byte in tokens_by_first_byte:
            # Limit how much text we examine to the maximum token length
            max_check_length = min(max_token_length, text_length - start_pos)
            current_text = text[start_pos:start_pos + max_check_length]
            
            # Check each candidate token
            for token_bytes, token_id in tokens_by_first_byte[current_byte]:
                if current_text.startswith(token_bytes):
                    valid_tokens.append((start_pos+1, start_pos+len(token_bytes), token_id, token_bytes))
        
        start_pos += 1
    
    # Now add the <|endoftext|> tokens with the correct ID
    for pos in eot_positions:
        valid_tokens.append((pos+1, pos+len(eot_pattern), separator_id, eot_pattern))
    
    # Sort by position
    valid_tokens.sort(key=lambda x: (x[0], x[1]))
    
    return valid_tokens

def calculate_positions(valid_tokens):
    """Calculate the position of each token using (end_byte_position)/4.0"""
    positions = {}
    for start_pos, token_len, token_id, token_bytes in valid_tokens:
        end_pos = start_pos + token_len - 1
        position = end_pos / 4.0
        positions[(token_id, start_pos)] = position
    return positions

def build_correct_ancestry(valid_tokens):
    """
    Build ancestry relationships based on ending bytes vs starting position.
    A token can attend to:
    1. Itself
    2. Any token whose ending byte position is less than its starting position
    """
    ancestry = {}
    
    sorted_by_end = sorted(valid_tokens, key=lambda t: t[1])
    
    # Process each token to build its ancestry
    for adj_start_pos, end_pos, token_id, token_bytes in valid_tokens:
        # Use the adjusted start position directly, no need to readjust
        current_key = (token_id, adj_start_pos)
        
        # Token always attends to itself
        ancestors = [current_key]
        
        # Find all tokens whose end position is < this token's start position
        # We need to use actual start position (adj_start_pos - 1) for comparison
        actual_start_pos = adj_start_pos - 1
        
        for other_adj_start, other_end, other_id, _ in sorted_by_end:
            # End position from optimized_find_all_tokens is exclusive
            # So we need to subtract 1 to get inclusive end position
            other_inclusive_end = other_end - 1
            
            if other_inclusive_end < actual_start_pos:
                other_key = (other_id, other_adj_start)
                if other_key != current_key:  # Avoid duplicating self
                    ancestors.append(other_key)
            else:
                # We can stop once we hit tokens ending at or after our start position
                # (since the list is sorted by ending position)
                break
        
        ancestry[current_key] = ancestors
    
    return ancestry
    
def build_optimized_attention_mask(token_positions, ancestry, position_map):
    """
    Build attention mask with improved performance by pre-allocating memory
    and minimizing dictionary lookups.
    """
    n = len(token_positions)
    
    # Pre-count total ancestry relationships for more efficient allocation
    total_relationships = sum(len(ancestors) for ancestors in ancestry.values())
    
    # Pre-allocate arrays with exact size needed
    all_i = [0] * total_relationships
    all_j = [0] * total_relationships
    
    idx = 0
    for i, (token_id, byte_pos, _) in enumerate(token_positions):
        key = (token_id, byte_pos)
        if key in ancestry:
            for ancestor_key in ancestry[key]:
                if ancestor_key in position_map:
                    all_i[idx] = i
                    all_j[idx] = position_map[ancestor_key]
                    idx += 1
    
    # Create the mask and set all values in one operation
    attn_mask = torch.zeros((n, n), dtype=torch.float32)
    
    if idx > 0:
        i_tensor = torch.tensor(all_i[:idx], dtype=torch.long)
        j_tensor = torch.tensor(all_j[:idx], dtype=torch.long)
        attn_mask[i_tensor, j_tensor] = 1.0
    
    return attn_mask

def seg_inv_enc(input_tensors, token_dict=None, debug=True, max_seq_len=None, trainable_pe=None):
    """
    Calculate segmentation-invariant positions and attention masks for input text.
    
    Args:
        input_tensors: PyTorch tensor of input token IDs
        token_dict: Dictionary mapping token bytes to token IDs
        debug: Whether to print debug information
        max_seq_len: Maximum sequence length to pad/truncate to
        trainable_pe: Optional trainable positional encoding module
        
    Returns:
        tuple: (positions tensor, attention mask tensor)
    """
    # Initialize tokenizer once
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Get separator token ID
    separator_id = 50256

    # Create document IDs using separator tokens (exactly as suggested)
    doc_ids = (input_tensors == separator_id).cumsum(0)

    # Convert input tensor to token IDs and decode to text
    token_ids = input_tensors.tolist()
    decoded_text = tokenizer.decode(token_ids)

    # Convert to bytes for processing
    if type(decoded_text) != bytes:
        text = decoded_text.encode("utf-8")
    
    if debug:
        print(f"Finding all tokens in: '{text}'")
    
    # Find all valid tokens in the text
    valid_tokens = find_all_tokens(text, token_dict)

    if debug:
        print(f"Found {len(valid_tokens)} valid tokens")
    
    # Create token_positions list with correct unpacking
    token_positions = [(token_id, start_pos, token_text) 
                      for start_pos, end_pos, token_id, token_text in valid_tokens]
    
    # Create position map for efficient lookups
    position_map = {(token_id, start_pos): i 
                   for i, (token_id, start_pos, _) in enumerate(token_positions)}
    
    # Calculate positions using either trainable or fixed approach
    if trainable_pe is not None:
        # Use trainable positional encoding
        pos_values = trainable_pe(valid_tokens)
        if debug:
            print("Using trainable segmentation-invariant positions")
    else:
        # Use the fixed formula
        pos_values = calculate_positions(valid_tokens)
        if debug:
            print("Using fixed segmentation-invariant positions: (byte end position)/4.0")


    # Build ancestry relationships
    if debug:
        print("Building ancestry relationships...")
    
    ancestry = build_correct_ancestry(valid_tokens)
    
    # Print ancestry relationships for debugging
    if debug:
        print("\nAncestry relationships:")
        
        # Helper function to get readable token text
        def get_readable_token(token_id, start_pos):
            for t_start, t_len, t_id, t_text in valid_tokens:
                if t_id == token_id and t_start == start_pos:
                    if t_id == separator_id and t_start == 0:
                        return "<|endoftext|>"
                    try:
                        return repr(t_text.decode('utf-8', errors='replace'))
                    except:
                        return f"[Binary token ID:{t_id}]"
            return f"Unknown token {token_id} at {start_pos}"
        
        # Print in a cleaner format with positions
        for i, (start_pos, token_len, token_id, token_text) in enumerate(valid_tokens):
            key = (token_id, start_pos)
            if key in ancestry:
                ancestors = ancestry[key]
                token_name = get_readable_token(token_id, start_pos)
                position = pos_values[key]
                
                print(f"Token {token_name} at pos {start_pos}")
                print("  Can attend to:")
                for ancestor_id, ancestor_pos in ancestors:
                    ancestor_name = get_readable_token(ancestor_id, ancestor_pos)
                    ancestor_position = pos_values[(ancestor_id, ancestor_pos)]
                    print(f"    - {ancestor_name} at pos {ancestor_pos} (position={ancestor_position:.2f})")
                print()
            
            # Limit output if there are too many tokens
            if i >= 20:
                remaining = len(valid_tokens) - i - 1
                print(f"... and {remaining} more tokens (omitted for brevity)")
                break

    # Step 4: Build attention mask
    if debug:
        print("\nBuilding attention mask...")

    attn_mask = build_optimized_attention_mask(token_positions, ancestry, position_map)

    # Build document boundary mask
    seq_len = len(input_tensors)
    
    # Create document mask
    doc_ids_expanded = doc_ids.unsqueeze(1)  # Shape [seq_len, 1]
    doc_ids_transposed = doc_ids.unsqueeze(0)  # Shape [1, seq_len]
    doc_mask = doc_ids_expanded == doc_ids_transposed  # Shape [seq_len, seq_len]
    device = input_tensors.device
    # Resize ancestry mask if needed
    if attn_mask.shape[0] != seq_len:
        n = attn_mask.shape[0]
        resized_mask = torch.zeros((seq_len, seq_len), dtype=torch.float32, device=device)
        copy_size = min(n, seq_len)
        resized_mask[:copy_size, :copy_size] = attn_mask[:copy_size, :copy_size]
        attn_mask = resized_mask
    
    # Convert masks to the same type and device for compatibility
    doc_mask = doc_mask.to(dtype=torch.float32, device=device)
    attn_mask = attn_mask.to(dtype=torch.float32, device=device)
    
    # Use multiplication instead of AND for float tensors
    final_mask = attn_mask * doc_mask

    # Print visualization of the attention mask
    if debug:
        print("\nAttention Mask Visualization:")
        print("(Rows: from tokens, Columns: to tokens - 1 means attention is allowed)")
        
        # For cleaner printing, use a numpy array
        attn_np = attn_mask.cpu().numpy()
        
        # Limit visualization to first 20x20 elements if mask is large
        max_display = min(20, len(token_positions))
        
        # Create token labels with consistent width for better display
        token_labels = []
        max_label_len = 0
        
        for i, (token_id, byte_pos, token_text) in enumerate(token_positions[:max_display]):
            if token_id == separator_id and byte_pos == 0:
                label = "<|eot|>"
            else:
                try:
                    # Get up to first 6 chars of token for labels
                    decoded = token_text.decode('utf-8', errors='replace')
                    # Replace spaces with underscores, and handle special characters
                    decoded = decoded.replace(' ', '_').replace('\n', '\\n').replace('\t', '\\t')
                    label = decoded[:6]
                except:
                    label = f"ID:{token_id}"
            
            token_labels.append(label)
            max_label_len = max(max_label_len, len(label))
        
        # Ensure all labels have consistent width with padding
        padded_labels = []
        for label in token_labels:
            padded_labels.append(label.ljust(max_label_len))
        
        # Create header row with column labels
        header = " " * (max_label_len + 2)  # Padding for row labels
        for label in padded_labels:
            header += label + " "
        print(header)
        
        # Print each row with padded row label
        for i in range(max_display):
            row_label = padded_labels[i]
            row_str = f"{row_label} |"
            for j in range(max_display):
                if attn_np[i, j] > 0:
                    row_str += " 1 " + " " * (max_label_len - 2)
                else:
                    row_str += " . " + " " * (max_label_len - 2)
            print(row_str)
        
        # If mask is larger than display limit, indicate truncation
        if len(token_positions) > max_display:
            print(f"\n[Showing only first {max_display}x{max_display} of {len(token_positions)}x{len(token_positions)} attention mask]")
    
    # Step 5: Build position encodings using segmentation-invariant approach
    if debug:
        print("\nBuilding position encodings...")
    positions = torch.zeros(len(token_positions), dtype=torch.float32)
    
    for i, (token_id, byte_pos, _) in enumerate(token_positions):
        key = (token_id, byte_pos)
        if key in pos_values:
            # The trainable model returns tensor values
            if isinstance(pos_values[key], torch.Tensor):
                positions[i] = pos_values[key].item()
            else:
                positions[i] = pos_values[key]
    
    # Print position encodings
    if debug:
        print("Position encodings (segmentation-invariant):")
        for i, (token_id, byte_pos, token_text) in enumerate(token_positions[:15]):  # Show first 15
            try:
                token_name = repr(token_text.decode('utf-8', errors='replace'))
            except:
                token_name = f"[Binary token ID:{token_id}]"
                
            print(f"  Token {i}: {token_name} - position {positions[i].item():.2f}")
        
        if len(token_positions) > 15:
            print(f"  ... and {len(token_positions) - 15} more tokens (omitted)")
    
    # handling max_seq_len padding/truncation for training
    orig_positions = positions
    
    if max_seq_len is not None:
        if len(positions) < max_seq_len:
            # Pad positions and attention mask
            padded_positions = torch.zeros(max_seq_len, dtype=positions.dtype)
            padded_positions[:len(positions)] = positions
            
            padded_attn_mask = torch.zeros((max_seq_len, max_seq_len), dtype=attn_mask.dtype)
            padded_attn_mask[:len(attn_mask), :len(attn_mask)] = attn_mask
            
            positions = padded_positions
            attn_mask = padded_attn_mask
        else:
            # Truncate to max_seq_len
            positions = positions[:max_seq_len]
            attn_mask = attn_mask[:max_seq_len, :max_seq_len]
    
    # Add this right after you calculate orig_positions and before any truncation
    print("Original positions (last 10):", orig_positions[-10:].tolist())

    # Add this right before returning the final positions
    print("Final positions (last 10):", positions[-10:].tolist())

    # For more detailed debugging, you could add this to show the total lengths
    print(f"Original position tensor length: {len(orig_positions)}")
    print(f"Final position tensor length: {len(positions)}")
    print(f"Input tensor length: {len(input_tensors)}")
    
    return positions, final_mask