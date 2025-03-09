import codecs
import os

# Japanese text (a short haiku about the moon)
japanese_text = "月の光\n静かな夜に\n影を落とす"
print("Original Japanese text:")
print(japanese_text)
print()

# Transcode to Shift-JIS encoding
shift_jis_bytes = japanese_text.encode('shift-jis')
print(f"Shift-JIS encoded bytes (hex):")
print(" ".join(f"{b:02x}" for b in shift_jis_bytes))
print()

# Check if these bytes are valid UTF-8
is_valid_utf8 = True
try:
    shift_jis_bytes.decode('utf-8')
    print("The Shift-JIS bytes are coincidentally valid UTF-8.")
except UnicodeDecodeError:
    is_valid_utf8 = False
    print("The Shift-JIS bytes are NOT valid UTF-8, as expected.")

# Check if these bytes are valid Shift-JIS
is_valid_shift_jis = True
try:
    shift_jis_bytes.decode('shift-jis')
    print("The Shift-JIS bytes are coincidentally valid Shift-JIS.")
    print(shift_jis_bytes.decode('shift-jis'))
except UnicodeDecodeError:
    is_valid_shift_jis = False
    print("The Shift-JIS bytes are NOT valid Shift-JIS, as expected.")

# Save the Shift-JIS encoded bytes to a file
with open("shifty.txt", "wb") as f:
    f.write(shift_jis_bytes)

print(f"\nShift-JIS encoded bytes have been saved to {os.path.abspath('shifty.txt')}")