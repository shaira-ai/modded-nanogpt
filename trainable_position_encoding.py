import torch
import torch.nn as nn
from collections import defaultdict

class TrainableSegInvPositionalEncoding(nn.Module):
    def __init__(self, token_dict, init_std=0.02):
        """
        Trainable segmentation-invariant positional encoding.
        
        Args:
            token_dict: Dictionary mapping token bytes to token IDs
            init_std: Standard deviation for parameter initialization
        """
        super().__init__()
        
        # Store token dictionary for reference
        self.token_dict = token_dict
        
        # Create a ParameterDict to hold trainable parameters for each token's bytes
        self.byte_params = nn.ParameterDict()
        
        # For each token, create parameters for each byte
        for token_bytes, token_id in token_dict.items():
            # Ensure token_bytes is bytes (not str)
            if isinstance(token_bytes, str):
                token_bytes = token_bytes.encode('utf-8')
                
            # Create trainable parameters for each byte in this token
            param = nn.Parameter(torch.randn(len(token_bytes)) * init_std)
            self.byte_params[str(token_id)] = param
            
        print(f"Created trainable parameters for {len(token_dict)} tokens")
    
    def forward(self, text, valid_tokens):
        """
        Calculate trainable segmentation-invariant positions.
        
        Args:
            text: The input text as bytes
            valid_tokens: List of (start_pos, len, token_id, token_bytes) tuples
            
        Returns:
            Dictionary mapping (token_id, start_pos) to position value
        """
        # Get device from parameters
        device = next(self.parameters()).device
        
        # Step 1: Map each byte position to tokens that contain it
        byte_to_tokens = {}
        for start_pos, token_len, token_id, token_bytes in valid_tokens:
            for i in range(token_len):
                byte_pos = start_pos + i
                if byte_pos not in byte_to_tokens:
                    byte_to_tokens[byte_pos] = []
                byte_to_tokens[byte_pos].append((token_id, i))
        
        # Step 2: Calculate position weight for each byte
        byte_weights = {}
        for byte_pos, token_byte_positions in byte_to_tokens.items():
            # Sum parameters for all tokens that contain this byte
            param_sum = torch.zeros(1, device=device)
            count = 0
            
            for token_id, byte_idx in token_byte_positions:
                # Get trainable parameter for this byte in this token
                token_key = str(token_id)
                if token_key in self.byte_params:
                    params = self.byte_params[token_key]
                    if byte_idx < len(params):
                        param_sum += params[byte_idx]
                        count += 1
            
            # Average the parameters (if any were found)
            if count > 0:
                byte_weights[byte_pos] = param_sum / count
            else:
                byte_weights[byte_pos] = torch.zeros(1, device=device)
        
        # Step 3: For each token, calculate final position as sum of byte weights
        positions = {}
        for start_pos, token_len, token_id, _ in valid_tokens:
            position = torch.zeros(1, device=device)
            for i in range(token_len):
                byte_pos = start_pos + i
                if byte_pos in byte_weights:
                    position += byte_weights[byte_pos]
            
            positions[(token_id, start_pos)] = position
        
        return positions