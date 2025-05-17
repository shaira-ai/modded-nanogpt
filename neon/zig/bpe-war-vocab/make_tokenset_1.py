#!/usr/bin/env python3
import struct
import os

def generate_tokenset_file(filename="tokenset_1.bin"):
    """
    Generates a tokenset file compatible with the Zig tokenset combiner.
    
    The file format consists of:
    - A header of 256 u32 integers (1024 bytes total)
    - The first u32 is set to 256 (little endian)
    - The remaining header values are set to 0
    - 256 bytes of data, one for each byte value (0-255)
    """
    # Create header: 256 u32 integers (all initialized to 0)
    header = bytearray(256 * 4)  # 4 bytes per u32
    
    # Set the first u32 to 256 (little endian)
    struct.pack_into("<I", header, 0, 256)
    
    # Create the data: 256 bytes, one for each byte-value, in ascending order
    data = bytes(range(256))
    
    # Write the file
    with open(filename, "wb") as f:
        f.write(header)
        f.write(data)
    
    print(f"Created {filename} ({os.path.getsize(filename)} bytes)")
    print(f"Header: 256 u32 values, first value = 256")
    print(f"Data: 256 bytes (values 0-255)")

if __name__ == "__main__":
    generate_tokenset_file()