import torch
import torch.nn as nn
from collections import defaultdict

class TrainableSegInvPositionalEncoding(nn.Module):
    def __init__(self, token_dict, init_std=0.02):
        """
        Trainable segmentation-invariant positional encoding with one parameter per token.
        """
        super().__init__()
        
        # Store token dictionary for reference
        self.token_dict = token_dict
        
        # Create a single trainable parameter per token
        token_params = {}
        for token_bytes, token_id in token_dict.items():
            token_params[str(token_id)] = nn.Parameter(torch.randn(1) * init_std)
            
        # Register all parameters as a ParameterDict
        self.token_params = nn.ParameterDict(token_params)
        
        print(f"Created trainable parameters for {len(token_dict)} tokens")
    
    def find_non_dominated_tokens(self, valid_tokens):
        """
        Optimized method to find tokens that aren't strict substrings of other tokens.
        """
        # Convert all tokens to consistent format and group by start position
        token_bytes_list = []
        tokens_by_start = defaultdict(list)
        
        for i, (start_pos, end_pos, token_id, token_bytes) in enumerate(valid_tokens):
            token_bytes_list.append(token_bytes)
            tokens_by_start[start_pos].append(i)
        
        # Set of non-dominated token indices
        non_dominated = set(range(len(valid_tokens)))
        
        # Sort tokens by length (shorter tokens are more likely to be dominated)
        sorted_indices = sorted(range(len(valid_tokens)), 
                               key=lambda i: len(token_bytes_list[i]))
        
        # Check for dominance with early stopping
        for idx in sorted_indices:
            if idx not in non_dominated:
                continue  # Already marked as dominated
                
            start_i, _, _, _ = valid_tokens[idx]
            bytes_i = token_bytes_list[idx]
            
            # Only check tokens that start at or before this one
            for start_j in range(start_i + 1):
                for j_idx in tokens_by_start[start_j]:
                    if idx == j_idx or j_idx not in non_dominated:
                        continue
                    
                    start_j, _, _, _ = valid_tokens[j_idx]
                    bytes_j = token_bytes_list[j_idx]
                    
                    # Quick length check before expensive substring check
                    offset = start_i - start_j
                    if offset >= 0 and offset + len(bytes_i) <= len(bytes_j):
                        # Do the substring check
                        try:
                            if bytes_j[offset:offset + len(bytes_i)] == bytes_i:
                                non_dominated.discard(idx)
                                break
                        except:
                            pass
            
        return [valid_tokens[i] for i in non_dominated]
    
    def forward(self, valid_tokens):
        """
        Optimized calculation of trainable segmentation-invariant positions.
        """
        device = next(self.parameters()).device
        
        non_dominated_tokens = self.find_non_dominated_tokens(valid_tokens)
        
        # Find the maximum byte position more efficiently
        max_byte_pos = max((start_pos + len(token_bytes) - 1) 
                           for start_pos, _, _, token_bytes in valid_tokens)
        
        # Create array for byte-level positions
        byte_positions = torch.zeros(max_byte_pos + 1, device=device)
        
        # Pre-allocate lists for parameters at each byte position
        # this stores the trainable params for each byte
        byte_params = [[] for _ in range(max_byte_pos + 1)]
        
        # Collect parameters for each byte position in a single pass
        for start_pos, _, token_id, token_bytes in non_dominated_tokens:
            token_key = str(token_id)
            if token_key not in self.token_params:
                continue
                
            param = self.token_params[token_key]
            
            # Add this parameter to all bytes in this token (single loop)
            for i in range(len(token_bytes)):
                byte_pos = start_pos + i
                if 0 <= byte_pos <= max_byte_pos:
                    byte_params[byte_pos].append(param)
        
        # Compute average parameters for each byte (vectorized)
        for byte_pos, params in enumerate(byte_params):
            if params:
                # Stack and average in a single operation
                byte_positions[byte_pos] = torch.stack(params).mean()
        
        # Compute cumulative sum
        cumulative_positions = torch.cumsum(byte_positions, dim=0)
        
        # Calculate positions for all tokens using lookup
        positions = {}
        for start_pos, _, token_id, token_bytes in valid_tokens:
            end_byte_pos = start_pos + len(token_bytes) - 1
            if 0 <= end_byte_pos <= max_byte_pos:
                positions[(token_id, start_pos)] = cumulative_positions[end_byte_pos]
            else:
                positions[(token_id, start_pos)] = torch.zeros(1, device=device)
        
        return positions