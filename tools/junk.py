import unicodedata


def junk_ok():
    for ch in range(0,256):
        ch = bytes([ch])
        for x in range(6,7):
            document = ch * x
            standard_bpb = calculate_standard_bpb(document, encoder, model, device)
            naive_apbpb, prob_array = calculate_naive_apbpb(document, encoder, model, device)
            print(f"for {x} spaces, {(standard_bpb - naive_apbpb)/(standard_bpb*100):.6f}%")

def get_letter_byte_ranges():
    # Collect all Unicode code points that are letters
    letter_ranges = []
    current_start = None
    current_end = None
    
    # Check characters up to the Unicode limit
    for codepoint in range(0x110000):
        char = chr(codepoint)
        try:
            if unicodedata.category(char).startswith('L'):
                #print(f"Found letter at codepoint {codepoint:04x}: {char}")
                # This is a letter
                if current_start is None:
                    current_start = codepoint
                current_end = codepoint
            elif current_start is not None:
                # End of a range
                letter_ranges.append((current_start, current_end))
                current_start = None
        except:
            # Some code points may not be valid
            if current_start is not None:
                letter_ranges.append((current_start, current_end))
                current_start = None
    
    # Add the last range if needed
    if current_start is not None:
        letter_ranges.append((current_start, current_end))
    print(letter_ranges)
    
    # Convert ranges to UTF-8 byte patterns
    utf8_byte_patterns = []
    for start, end in letter_ranges:
        # Skip control characters
        if start < 32:
            continue
            
        # Handle ASCII range specially (for efficiency)
        if start < 128:
            ascii_start = max(start, 0)
            ascii_end = min(end, 127)
            if ascii_start <= ascii_end:
                if ascii_start == ascii_end:
                    utf8_byte_patterns.append(f"\\x{ascii_start:02x}")
                else:
                    utf8_byte_patterns.append(f"[\\x{ascii_start:02x}-\\x{ascii_end:02x}]")
        
        # Handle multi-byte UTF-8 sequences
        # This gets complex because UTF-8 encoding isn't a simple range
        # For simplicity, we'll just generate patterns for some key ranges
        
    return utf8_byte_patterns

# Generate and print patterns
patterns = get_letter_byte_ranges()
print("Byte patterns for \\p{L}:")
print("|".join(patterns))