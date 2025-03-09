import regex as re
from functools import lru_cache

# First, we need to define our escape/unescape functions
def escape_non_utf8(byte_data: bytes) -> str:
    """Convert any non-UTF-8 bytes to escaped Unicode characters in Private Use Area."""
    result = ""
    i = 0
    backslash = chr(0xE000)
    while i < len(byte_data):
        # Try to decode as much as possible
        for j in range(min(4, len(byte_data) - i), 0, -1):
            try:
                chunk = byte_data[i:i+j].decode('utf-8')
                for char in chunk:
                    if char == backslash:
                        # Escape the escape character with itself
                        result += backslash + backslash
                    else:
                        result += char
                i += j
                break
            except UnicodeDecodeError:
                if j == 1:
                    # Single invalid byte - encode it to Private Use Area
                    # Use U+E000 as an escape character
                    # and use U+E001 + byte_value to encode the byte
                    byte_val = byte_data[i]
                    result += backslash
                    result += chr(0xE001 + byte_val)
                    i += 1
    return result

def unescape_to_bytes(escaped_str: str) -> bytes:
    """Convert escaped Unicode characters back to their original bytes."""
    result = bytearray()
    i = 0
    backslash = chr(0xE000)
    while i < len(escaped_str):
        if escaped_str[i] == backslash:
            # This is an escaped character
            if i + 1 >= len(escaped_str):
                # Invalid escape sequence
                raise ValueError(f"Invalid escape sequence at position {i}")
            else:
                # Valid escape sequence - decode it
                i += 1
                char_code = ord(escaped_str[i])
                if escaped_str[i] == backslash:
                    # This is a 0xE000 - treat it as a 0xE000
                    result.extend(backslash.encode('utf-8'))
                elif 0xE001 <= char_code <= 0xE100:
                    # This is our escaped byte
                    original_byte = char_code - 0xE001
                    result.append(original_byte)
                else:
                    raise ValueError(f"Invalid escaped byte at position {i+1}")
        else:
            # Regular character - encode back to UTF-8
            result.extend(escaped_str[i].encode('utf-8'))
        i += 1
    return bytes(result)


@lru_cache(maxsize=32)
def get_compiled_pattern(pattern_str):
    return re.compile(pattern_str)

def please_encode(tokenizer, text, **kwargs):
    # already a string - just use the tokenizer
    if type(text) == str:
        return tokenizer.encode(text, **kwargs)
    # let's try to utf-8 decode it first
    try:
        text = text.decode('utf-8')
        return tokenizer.encode(text, **kwargs)
    except UnicodeDecodeError:
        pass
    # Using some tricks that would not be necessary if this world was sane,
    # apply the GPT-2 pre-tokenization regex to mixed UTF-8/non-UTF-8 content.
    # Step 1: Escape non-UTF-8 bytes
    escaped_text = escape_non_utf8(text)

    # Step 2: Apply the pre-tokenization regex
    pattern = get_compiled_pattern(tokenizer._pat_str)
    matches = pattern.finditer(escaped_text)
    segments = [match.group() for match in matches]

    # Step 3: Unescape each segment back to bytes and encode with the tokenizer
    ret = []
    for segment in segments:
        # Skip empty segments
        if len(segment) == 0:
            continue
        segment = unescape_to_bytes(segment)
        try:
            segment = segment.decode('utf-8')
            ret.extend(tokenizer.encode(segment, **kwargs))
        except UnicodeDecodeError:
            # published versions of tiktoken include a bug in _encode_bytes.
            # if the byte sequence begins with invalid bytes, it returns an
            # empty list.
            tokens = tokenizer._encode_bytes(segment, **kwargs)
            if len(tokens) == 0:
                tokens = tokenizer._encode_bytes(b"The" + segment, **kwargs)[1:]
            ret.extend(tokens)
    return ret
