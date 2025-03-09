import tiktoken
encoder = tiktoken.get_encoding("gpt2")
success_count = 0
for i in range(256):
    byte_seq = bytes([i])
    tokens = encoder._encode_bytes(byte_seq)
    decoded_bytes = b"".join(encoder.decode_single_token_bytes(token) for token in tokens)
    if decoded_bytes == byte_seq:
        success_count += 1
print(f"Successfully round-tripped: {success_count}/256 1-byte byte sequences")
print(f"Failed round-trip: {256-success_count}/256 1-byte byte sequences")
