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
from neon.pre_segment import please_encode, get_pre_segments
from neon.zig.bindings import get_all_matches, get_all_matches_including_past_end
from train_gpt_mod import GPT, Hyperparameters, init_model, device, get_window_size_blocks
from collections import defaultdict

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
            logits = model(input_tensor, sliding_window_num_blocks=window_blocks)
            
            # Calculate cross-entropy loss manually
            loss = 0
            for i in range(len(input_tensor)):
                token_logits = logits[0, i]
                target = target_tensor[i]
                
                # Apply log softmax to get log probabilities
                my_probs = torch.nn.functional.softmax(token_logits, dim=0)
                
                # Get negative log likelihood for this token
                print(f"Target log probability: {math.log(my_probs[target].item())}")
                if target < my_probs.size(0):
                    loss -= math.log(my_probs[target].item()) # Use log softmax
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

def calculate_optimized_apbpb(document, encoder, model, device):
    """
    Calculate All Paths Bits Per Byte (APBPB) in a single forward pass.
    This optimizes the calculation by processing all unique token positions together.
    """
    if type(document) != bytes:
        document = document.encode("utf-8")
    
    byte_count = len(document)
    print(f"Document length in bytes: {byte_count}")
    
    separator_id = encoder.encode("<|endoftext|>", allowed_special={'<|endoftext|>'})[0]
    separator_text = b"<|endoftext|>"
    
    mpmath.mp.dps = 1000

    # Step 0: Pre-segment the document to optimize tokenization
    print("Pre-segmenting document...")
    document_chunks = get_pre_segments(encoder, document)
    
    # Show a few chunks for debugging
    print(f"Document divided into {len(document_chunks)} chunks:")
    for i, chunk in enumerate(document_chunks[:5]):  # Show first 5 chunks
        try:
            chunk_str = chunk.decode('utf-8', errors='replace')
            print(f"  Chunk {i}: {repr(chunk_str)}")
        except:
            print(f"  Chunk {i}: [Binary chunk of length {len(chunk)}]")
    if len(document_chunks) > 5:
        print(f"  ... and {len(document_chunks) - 5} more chunks")
    
    # Step 1: Find canonical segmentations for all prefixes more efficiently
    print("Finding canonical segmentations for all prefixes using chunk optimization...")
    all_segmentations = {}

    # Add separator token as the segmentation for the empty prefix (position 0)
    all_segmentations[0] = [(separator_id, separator_text, 0)]
    print(f"  Prefix length 0: 1 token (<|endoftext|>)")

    # First, calculate the canonical segmentation for each chunk
    print("Pre-calculating segmentations for each chunk...")
    chunk_segmentations = []  # Will store segmentation for each chunk
    chunk_boundaries = [0]    # Start positions of each chunk in the document

    current_pos = 0
    for i, chunk in enumerate(document_chunks):
        chunk_boundaries.append(current_pos + len(chunk))
        
        # Get canonical segmentation for this chunk
        tokens = please_encode(encoder, chunk)
        
        # Decode each token to get its text representation
        segmentation = []
        current_bytes = b""
        for token_id in tokens:
            remaining = chunk[len(current_bytes):]
            
            # Try different lengths to find the token text
            found = False
            for token_len in range(1, min(len(remaining) + 1, encoder.max_token_length + 1)):
                token_text = remaining[:token_len]
                test_tokens = please_encode(encoder, token_text)
                if len(test_tokens) == 1 and test_tokens[0] == token_id:
                    # Store with position relative to document start
                    segmentation.append((token_id, token_text, current_pos + len(current_bytes) + 1))
                    current_bytes += token_text
                    found = True
                    break
            
            if not found:
                raise ValueError(f"Could not find token text for token ID {token_id} in chunk {i}")
        
        chunk_segmentations.append(segmentation)
        print(f"  Chunk {i} ({len(chunk)} bytes): {len(segmentation)} tokens")
        current_pos += len(chunk)

    # Now, calculate the segmentation for each prefix position
    print("Generating segmentations for all prefix positions...")
    for end_pos in range(1, byte_count+1):
        # Find which chunk this position falls into
        chunk_idx = 0
        while chunk_idx < len(chunk_boundaries) - 1 and end_pos > chunk_boundaries[chunk_idx + 1]:
            chunk_idx += 1
        
        # Calculate the position within the chunk
        pos_in_chunk = end_pos - chunk_boundaries[chunk_idx]
        
        if pos_in_chunk == len(document_chunks[chunk_idx]):
            # Position is at a chunk boundary - we can just combine full chunk segmentations
            segmentation = []
            for i in range(chunk_idx + 1):
                # Add segmentation for this complete chunk
                segmentation.extend(chunk_segmentations[i])
            
            all_segmentations[end_pos] = segmentation
        else:
            # Position is within a chunk - reuse previous chunks and compute partial chunk
            segmentation = []
            
            # Add segmentations for all complete previous chunks
            for i in range(chunk_idx):
                segmentation.extend(chunk_segmentations[i])
            
            # Calculate segmentation for the partial chunk
            partial_chunk = document_chunks[chunk_idx][:pos_in_chunk]
            
            # Get canonical segmentation for this partial chunk
            tokens = please_encode(encoder, partial_chunk)
            
            # Decode each token to get its text representation
            partial_segmentation = []
            current_bytes = b""
            offset = chunk_boundaries[chunk_idx]  # Start position of this chunk
            
            for token_id in tokens:
                remaining = partial_chunk[len(current_bytes):]
                
                # Try different lengths to find the token text
                found = False
                for token_len in range(1, min(len(remaining) + 1, encoder.max_token_length + 1)):
                    token_text = remaining[:token_len]
                    test_tokens = please_encode(encoder, token_text)
                    if len(test_tokens) == 1 and test_tokens[0] == token_id:
                        # Store with position relative to document start
                        partial_segmentation.append((token_id, token_text, offset + len(current_bytes) + 1))
                        current_bytes += token_text
                        found = True
                        break
                
                if not found:
                    raise ValueError(f"Could not find token text for token ID {token_id} in partial chunk")
            
            # Add segmentation for the partial chunk
            segmentation.extend(partial_segmentation)
            
            all_segmentations[end_pos] = segmentation
        
        # Print progress every 10 positions or for small positions
        if end_pos % 10 == 0 or end_pos < 10:
            print(f"  Prefix length {end_pos}: {len(all_segmentations[end_pos])} tokens")

    print(f"Generated segmentations for {len(all_segmentations)} prefix positions")
    
    # Step 2: Build the list of unique (token_id, byte_position, token_text) tuples
    print("Building list of unique token positions...")
    token_positions = set()
    position_map = {}  # Maps (token_id, byte_pos) to index in input sequence
    
    # Add all tokens from segmentations (including separator token which is already in all_segmentations[0])
    for prefix_len, segmentation in all_segmentations.items():
        for token_id, token_text, byte_pos in segmentation:
            token_positions.add((token_id, byte_pos, token_text))
    
    # Convert to list and calculate end positions for sorting
    token_positions_with_end = []
    for token_id, byte_pos, token_text in token_positions:
        end_pos = byte_pos + len(token_text)
        token_positions_with_end.append((token_id, byte_pos, token_text, end_pos))

    # Custom sorting function to ensure separator is first
    def sort_key(item):
        token_id, byte_pos, token_text, end_pos = item
        # If this is the separator token, give it lowest possible value to sort first
        if token_id == separator_id and byte_pos == 0:
            return (-1, -1, -1)
        # Otherwise sort by end position, then start position
        return (end_pos, byte_pos, token_id)

    # Sort with our custom sorting function
    token_positions_with_end.sort(key=sort_key)

    # Extract back to original format and create position map
    token_positions = [(t[0], t[1], t[2]) for t in token_positions_with_end[:-1]]
    
    # Create mapping from position to index
    for i, (token_id, byte_pos, token_text) in enumerate(token_positions):
        position_map[(token_id, byte_pos)] = i
        
    # Print the tokens in order
    print(f"Found {len(token_positions)} unique token positions (including <|endoftext|>)")
    print("Token positions in processing order (sorted by end byte position):")
    for i, (token_id, byte_pos, token_text) in enumerate(token_positions):
        # Get a readable representation of the token
        if token_id == separator_id and byte_pos == 0:
            token_name = "<|endoftext|>"
        else:
            try:
                token_name = repr(token_text.decode('utf-8', errors='replace'))
            except:
                token_name = f"[Binary token of length {len(token_text)}]"
        
        end_pos = byte_pos + len(token_text)
        print(f"  [{i}] Bytes {byte_pos}-{end_pos}: ID:{token_id} - {token_name}")
    
    # Step 3: Build the ancestry relationships
    print("Building ancestry relationships...")
    ancestry = {}
    
    separator_pos = (separator_id, 0)
    
    for prefix_len, segmentation in all_segmentations.items():
        for i, (token_id, token_text, byte_pos) in enumerate(segmentation):
            ancestors = []
            
            if not (token_id == separator_id and byte_pos == 0):
                ancestors.append(separator_pos)
                
            # Add preceding tokens in this segmentation as ancestors
            for j in range(i):
                ancestor_id, ancestor_text, ancestor_pos = segmentation[j]
                ancestors.append((ancestor_id, ancestor_pos))
            
            ancestry[(token_id, byte_pos)] = ancestors
            
    # Print ancestry relationships for debugging
    print("\nAncestry relationships:")
    for pos_idx, (token_id, byte_pos, token_text) in enumerate(token_positions):
        if (token_id, byte_pos) in ancestry:
            ancestors = ancestry[(token_id, byte_pos)]
            if ancestors:
                ancestor_texts = []
                for ancestor_id, ancestor_pos in ancestors:
                    # Find the ancestor's token text
                    for tok_id, tok_pos, tok_text in token_positions:
                        if tok_id == ancestor_id and tok_pos == ancestor_pos:
                            if tok_id == separator_id and tok_pos == 0:
                                ancestor_name = "<|endoftext|>"
                            else:
                                try:
                                    ancestor_name = repr(tok_text.decode('utf-8', errors='replace'))
                                except:
                                    ancestor_name = f"[Binary token ID:{tok_id}]"
                            ancestor_texts.append(ancestor_name)
                            break
                
                if token_id == separator_id and byte_pos == 0:
                    token_name = "<|endoftext|>"
                else:
                    try:
                        token_name = repr(token_text.decode('utf-8', errors='replace'))
                    except:
                        token_name = f"[Binary token ID:{token_id}]"
                    
                print(f"  {token_name} at pos {byte_pos} has ancestors: {', '.join(ancestor_texts)}")
        
        # Limit output if there are too many tokens
        if pos_idx >= 15:
            remaining = len(token_positions) - pos_idx - 1
            print(f"  ... and {remaining} more tokens (omitted for brevity)")
            break
    
    # Step 4: Build attention mask
    print("\nBuilding attention mask...")
    attn_mask = torch.zeros((len(token_positions), len(token_positions)), dtype=torch.float32)
    
    for i, (token_id, byte_pos, token_text) in enumerate(token_positions):
        # Token can attend to itself
        attn_mask[i, i] = 1
        
        # Token can attend to its ancestors
        if (token_id, byte_pos) in ancestry:
            for ancestor_id, ancestor_pos in ancestry[(token_id, byte_pos)]:
                if (ancestor_id, ancestor_pos) in position_map:
                    j = position_map[(ancestor_id, ancestor_pos)]
                    attn_mask[i, j] = 1
    
    # Print visualization of the attention mask
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
        if byte_pos == -1:
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
    
    # Print separator line
    print(" " * (max_label_len + 2) + "-" * (max_display * (max_label_len + 1)))
    
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
    
    # Step 5: Build position encodings based on attention mask rows
    print("\nBuilding position encodings...")
    positions = torch.zeros(len(token_positions), dtype=torch.int32)

    for i in range(len(token_positions)):
        # Count the number of 1s in this row of the attention mask (how many tokens this token can attend to)
        num_attended_tokens = torch.sum(attn_mask[i]).item()
        
        # Position is (number of attended tokens - 1)
        positions[i] = int(num_attended_tokens - 1)

    # Print position encodings
    print("Position encodings:")
    for i, (token_id, byte_pos, token_text) in enumerate(token_positions):  # Show first 15
        try:
            token_name = repr(token_text.decode('utf-8', errors='replace'))
        except:
            if token_id == separator_id and byte_pos == 0:
                token_name = "<|endoftext|>"
            else:
                token_name = f"[Binary token ID:{token_id}]"
            
        print(f"  Token {i}: {token_name} - position {positions[i].item()}")

    
    # Step 6: Build input tensor
    print("\nBuilding input tensor...")
    input_ids = torch.zeros(len(token_positions), dtype=torch.int32)
    
    for i, (token_id, byte_pos, token_text) in enumerate(token_positions):
        input_ids[i] = token_id
        
    # Print input IDs
    print("Input IDs:")
    for i, (token_id, byte_pos, token_text) in enumerate(token_positions[:15]):  # Show first 15
        try:
            token_name = repr(token_text.decode('utf-8', errors='replace'))
        except:
            token_name = f"[Binary token of length {len(token_text)}]"
            
        print(f"  Position {i}: ID {token_id} - {token_name}")
    if len(token_positions) > 15:
        print(f"  ... and {len(token_positions) - 15} more (omitted for brevity)")
    
    # Step 7: Move tensors to device
    input_ids = input_ids.to(device)
    positions = positions.to(device)
    attn_mask = attn_mask.to(device)
    
    # Step 8: Run single forward pass
    print("Running forward pass...")
    try:
        with torch.no_grad():
            model.eval()
            outputs = model(
                input_ids=input_ids.unsqueeze(0), 
                position_ids=positions.unsqueeze(0),
                attention_mask=attn_mask.unsqueeze(0)
            )
            
            # Extract logits
            if hasattr(outputs, 'logits'):
                logits = outputs.logits[0]  # [num_tokens, vocab_size]
            elif isinstance(outputs, torch.Tensor) and outputs.dim() >= 2:
                logits = outputs[0]  # [num_tokens, vocab_size]
            else:
                raise ValueError(f"Unexpected output format: {type(outputs)}")
            
            # Convert to probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)
    except Exception as e:
        print(f"Error during forward pass: {str(e)}")
        traceback.print_exc()
        return float('inf'), {}
    
    # Step 9: Extract probabilities and propagate them
    print("Extracting and propagating probabilities...")

    prob_array = [mpmath.mpf(0)] * (byte_count + 1 + encoder.max_token_length)
    prob_array[0] = mpmath.mpf(1.0)

    all_tokens_in_doc = get_all_matches_including_past_end(encoder, document)
    all_tokens_in_doc.sort()
    all_tokens_idx = 0

    # Process prefixes in ascending order to ensure proper probability propagation
    for prefix_len in sorted(all_segmentations.keys()):
        # Skip if this position has zero probability (unreachable)
        if prob_array[prefix_len] <= 0:
            continue
            
        # Find the last token in this prefix's segmentation
        segmentation = all_segmentations[prefix_len]
        if not segmentation:
            continue
            
        last_token = segmentation[-1]
        last_token_id, last_token_text, last_token_pos = last_token
        
        # Find this token's index in our token_positions list
        if (last_token_id, last_token_pos) in position_map:
            idx = position_map[(last_token_id, last_token_pos)]
            
            # Get the token's output distribution
            token_probs = probs[idx]
            
            # Find all valid next tokens at this position
            next_byte_pos = prefix_len

            range_lo = all_tokens_idx
            while(all_tokens_idx < len(all_tokens_in_doc) and all_tokens_in_doc[all_tokens_idx][0] == next_byte_pos):
                all_tokens_idx += 1
            valid_tokens = all_tokens_in_doc[range_lo:all_tokens_idx]
            valid_tokens = [(x[2], encoder.token_id_to_bytes[x[2]]) for x in valid_tokens]
            #other_valid_tokens = find_valid_next_tokens(document, next_byte_pos, encoder)
            #other_valid_tokens.sort(key=lambda x: len(x[1]))
            #assert(valid_tokens == other_valid_tokens)
            
            # Format prefix string for display
            if prefix_len == 0:
                prefix_str = "<empty>"
            else:
                try:
                    prefix_str = document[:prefix_len].decode('utf-8', errors='replace')
                except:
                    prefix_str = f"[Binary prefix of length {prefix_len}]"
            
            # Display prefix info
            if last_token_id == separator_id and last_token_pos == 0:
                last_token_display = "<|endoftext|>"
            else:
                try:
                    last_token_display = repr(last_token_text.decode('utf-8', errors='replace'))
                except:
                    last_token_display = f"[Binary token ID:{last_token_id}]"
                    
            print(f"\nPrefix: {repr(prefix_str)} (length {prefix_len})")
            print(f"  Last token: {last_token_display} (ID: {last_token_id})")
            print(f"  Current position probability: {float(prob_array[prefix_len]):.8e}")
            
            # Show valid completions with individual probabilities
            print("  Valid completions:")
            
            # Sort by probability (highest first) for more informative display
            token_prob_pairs = []
            position_total_prob = mpmath.mpf(0)
            
            for next_token_id, next_token_text in valid_tokens:
                if next_token_id < token_probs.size(0):
                    token_prob = token_probs[next_token_id].item()
                    if token_prob > 0:
                        # Calculate the new position
                        new_pos = prefix_len + len(next_token_text)
                        
                        # Update probability array (just like in naive method)
                        prob_array[new_pos] += prob_array[prefix_len] * mpmath.mpf(token_prob)
                        
                        try:
                            completion_str = next_token_text.decode('utf-8', errors='replace')
                            completion_display = repr(completion_str)
                        except:
                            completion_display = f"[Binary token ID:{next_token_id}]"
                        
                        token_prob_pairs.append((completion_display, token_prob, next_token_id, new_pos))
                        position_total_prob += mpmath.mpf(token_prob)
            
            # Sort by probability (highest first)
            token_prob_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # Display individual token probabilities (limit to top 10 for readability)
            display_count = min(10, len(token_prob_pairs))
            for i, (completion_display, prob, token_id, new_pos) in enumerate(token_prob_pairs[:display_count]):
                print(f"    {completion_display}: prob={prob:.8e} (ID: {token_id}) → pos {new_pos}")
            
            # Show count of remaining tokens if there are more than the display limit
            if len(token_prob_pairs) > display_count:
                print(f"    ... and {len(token_prob_pairs) - display_count} more tokens")
            
            # Show total probability for this position
            print(f"  Total probability at position {prefix_len}: {float(position_total_prob):.8e}")

    # Step 10: Compute APBPB
    print("Computing final APBPB...")

    # Calculate final probability (sum of probabilities at or beyond document length)
    final_prob = sum(prob_array[byte_count:])
    print(f"Final probability sum: {float(final_prob):.8e}")

    if final_prob > 0:
        apbpb = -mpmath.log(final_prob, 2) / byte_count
    else:
        apbpb = float('inf')

    print(f"APBPB: {float(apbpb):.6f}")

    print("\nCalculating standard BPB...")

    standard_bpb = float('inf')

    # Try to get the canonical segmentation for the full document
    # Fall back to the longest available segmentation if needed
    if byte_count in all_segmentations:
        full_segmentation = all_segmentations[byte_count]
        print(f"Found canonical segmentation for full document length ({byte_count} bytes)")
    else:
        # Find the longest available segmentation
        longest_len = max(all_segmentations.keys())
        if longest_len < byte_count:
            full_segmentation = all_segmentations[longest_len]
            print(f"Warning: No canonical segmentation found for full document length {byte_count}")
            print(f"Using longest available segmentation (length {longest_len} bytes) instead")
        else:
            # This shouldn't happen as we should have segmentations for all prefixes,
            # but just in case, grab a segmentation that's closest to the document length
            closest_len = min(all_segmentations.keys(), key=lambda x: abs(x - byte_count))
            full_segmentation = all_segmentations[closest_len]
            print(f"Warning: Using closest available segmentation (length {closest_len} bytes)")

    # For debugging, show the canonical segmentation
    print(f"Segmentation used for BPB ({len(full_segmentation)} tokens):")
    for i, (token_id, token_text, byte_pos) in enumerate(full_segmentation):
        try:
            token_display = repr(token_text.decode('utf-8', errors='replace'))
        except:
            token_display = f"[Binary token ID:{token_id}]"
        print(f"  Token {i}: {token_display} at position {byte_pos}")

    # Calculate standard BPB
    token_count = len(full_segmentation)
    total_neg_log_prob = 0.0

    if token_count > 0:
        # Create the sequence: separator token followed by segmentation tokens
        token_ids = [separator_id] + [t[0] for t in full_segmentation]
        
        # Process each token transition
        for i in range(token_count):
            # Get current and previous token
            current_token_id = token_ids[i+1]  # +1 because token_ids includes separator
            
            # Get position of previous token
            if i == 0:
                # Previous token is separator at position 0
                prev_token_key = (separator_id, 0)
            else:
                # Previous token is from segmentation
                prev_pos = full_segmentation[i-1][2]
                prev_token_key = (full_segmentation[i-1][0], prev_pos)
            
            # Find previous token's index in token_positions
            if prev_token_key in position_map:
                prev_idx = position_map[prev_token_key]
                
                # Get the logits predicted by the previous token
                prev_token_logits = logits[prev_idx]
                
                # Apply softmax to get probabilities
                prev_token_probs = torch.nn.functional.softmax(prev_token_logits, dim=-1)
                
                # Get the probability of the current token
                if current_token_id < prev_token_probs.size(0):
                    prob = prev_token_probs[current_token_id].item()
                    if prob > 0:
                        # Add negative log probability
                        neg_log_prob = -math.log(prob)
                        total_neg_log_prob += neg_log_prob
                        
                        if i < 5:
                            try:
                                token_display = repr(full_segmentation[i][1].decode('utf-8', errors='replace'))
                            except:
                                token_display = f"[Binary token ID:{current_token_id}]"
                            print(f"  Token {token_display}: prob={prob:.8e}, -log(prob)={neg_log_prob:.8f}")
                    else:
                        print(f"Warning: Zero probability for token ID {current_token_id}")
                        total_neg_log_prob += 100  # Large penalty for zero probability
                else:
                    print(f"Warning: Token ID {current_token_id} out of range")
                    total_neg_log_prob += 100  # Large penalty
            else:
                print(f"Warning: Previous token key {prev_token_key} not found in position map")
                total_neg_log_prob += 100  # Large penalty

        # Average loss
        loss_value = total_neg_log_prob / token_count
        
        standard_bpb = (token_count / byte_count) * loss_value * math.log2(math.e)
        print(f"Standard BPB: {standard_bpb:.6f}")
    else:
        print("Warning: No tokens found in segmentation")
        standard_bpb = float('inf')

    print(f"APBPB: {float(apbpb):.6f}, Standard BPB: {float(standard_bpb):.6f}")

    return float(standard_bpb), float(apbpb), {i: float(p) for i, p in enumerate(prob_array) if p > 0}
    
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

    all_tokens_in_doc = get_all_matches_including_past_end(encoder, document)
    all_tokens_in_doc.sort()
    all_tokens_idx = 0
    
    for pos in range(len(document)):
        range_lo = all_tokens_idx
        while(all_tokens_idx < len(all_tokens_in_doc) and all_tokens_in_doc[all_tokens_idx][0] == pos):
            all_tokens_idx += 1
        valid_tokens = all_tokens_in_doc[range_lo:all_tokens_idx]
        valid_tokens = [(x[2], encoder.token_id_to_bytes[x[2]]) for x in valid_tokens]
        #other_valid_tokens = find_valid_next_tokens(document, pos, encoder)
        #other_valid_tokens.sort(key=lambda x: len(x[1]))
        #assert(valid_tokens == other_valid_tokens)
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
            
            # Format prefix string for display
            if pos == 0:
                prefix_str = "<empty>"
            else:
                try:
                    prefix_str = document[:pos].decode('utf-8', errors='replace')
                except:
                    prefix_str = f"[Binary prefix of length {pos}]"
            
            print(f"\nPrefix: {repr(prefix_str)} (length {pos})")
            print("  Valid completions:")
            
            # Sort by probability (highest first) for more informative display
            token_prob_pairs = []
            
            # Store probabilities for valid tokens
            for token_id, token_text in position_tokens[pos]:
                if token_id < next_token_probs.size(0):
                    token_prob = next_token_probs[token_id].item()
                    token_probs[(token_id, pos)] = token_prob
                    
                    # Update probability array (original code unchanged)
                    new_pos = pos + len(token_text)
                    prob_array[new_pos] += prob_array[pos] * mpmath.mpf(token_prob)
                    
                    # Add to our display list
                    try:
                        completion_str = token_text.decode('utf-8', errors='replace')
                        completion_display = repr(completion_str)
                    except:
                        completion_display = f"[Binary token ID:{token_id}]"
                    
                    token_prob_pairs.append((completion_display, token_prob, token_id))
            
            # Sort by probability (highest first)
            token_prob_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # Display individual token probabilities (limit to top 10 for readability)
            display_count = min(10, len(token_prob_pairs))
            for i, (completion_display, prob, token_id) in enumerate(token_prob_pairs[:display_count]):
                print(f"    {completion_display}: prob={prob:.8e} (ID: {token_id})")
            
            # Show count of remaining tokens if there are more than the display limit
            if len(token_prob_pairs) > display_count:
                print(f"    ... and {len(token_prob_pairs) - display_count} more tokens")
            
            # Calculate and show total probability for this position
            total_pos_prob = sum(prob for _, prob, _ in token_prob_pairs)
            print(f"  Total probability: {total_pos_prob:.8e}")
                    
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
            config = Hyperparameters()  # Create with defaults
            # Override with inferred values
            config.vocab_size = vocab_size
            config.num_layers = num_layers
            config.num_heads = num_heads
            config.model_dim = model_dim
            # There's no max_seq_len in your class, so use appropriate field
            config.train_seq_len = max_seq_len
            config.val_seq_len = max_seq_len
            
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

import time
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

    # Record the start time
    start_time = time.time()
    
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
        model = load_model_from_checkpoint('logs/e4d007cc-b53b-42ee-90d4-75ad9d35dfa7/state_step001000.pt')
    
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
            bpb, apbpb, prob_array = calculate_optimized_apbpb(document, encoder, model, device)
            results["bpb"] = bpb
            results["apbpb"] = apbpb
            naive_apbpb, prob_array = calculate_naive_apbpb(document, encoder, model, device)
            results["naive_apbpb"] = naive_apbpb
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
    if results["bpb"] is not None and results["apbpb"] is not None:
        print("\n=== Comparison ===")
        if 'standard_bpb' in results:
            print(f"Standard BPB (old): {results['standard_bpb']:.6f}")
        print(f"Standard BPB:       {results['bpb']:.6f}")
        print(f"APBPB:              {results['apbpb']:.6f}")
        if 'naive_apbpb' in results:
            print(f"Naive APBPB:        {results['naive_apbpb']:.6f}")
        is_better = results["apbpb"] < results["bpb"]
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
    # Record the end time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.6f} seconds")

if __name__ == "__main__":
    main()
