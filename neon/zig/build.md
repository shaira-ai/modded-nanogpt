## Prerequisites

- [Zig 0.14.0](https://ziglang.org/)
- Python 3.x

## Building the Library

### Compile the Zig Library

```zsh
zig build-lib uc_conversion.zig -dynamic
```

This will create:
- `libuc_conversion.dylib` - the required library file

### Using the Library from Python

I have used ctypes to load the library file and create the bindings.
The library path is hardcoded in the Python file.

Run with:
```zsh
python3 uc_binding.py
```