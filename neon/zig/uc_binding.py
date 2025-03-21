import ctypes
from ctypes import c_char_p, c_size_t, POINTER

# load the library
lib = ctypes.CDLL('./libuc_conversion.dylib')

# define function signatures
lib.convertToUC.argtypes = [
    POINTER(c_char_p), # input string array
    c_size_t, # input count
    POINTER(c_size_t) # result count
]

lib.convertToUC.restype = POINTER(c_char_p)

lib.free_uc_strings.argtypes = [
    POINTER(c_char_p),
    c_size_t
]
lib.free_uc_strings.restype = None

def UCconversion(strings):
    """Convert a list of strings to uppercase using Zig."""

    # convert python list to array of C strings
    string_count = len(strings)
    c_strings = (c_char_p * string_count)()
    for i, s in enumerate(strings):
        c_strings[i] = s.encode('utf-8')
    
    # prepare output parameter for result count
    result_count = c_size_t()

    # call the function
    result_ptr = lib.convertToUC(c_strings, string_count, ctypes.byref(result_count))

    # convert result to python strings
    result = []
    for i in range(result_count.value):
        result.append(result_ptr[i].decode('utf-8'))
    
    lib.free_uc_strings(result_ptr, result_count.value)

    return result

if __name__ == "__main__":
    test_strings = ["hello", "world", "zig", "Python", "Mixed123Case!"]
    uppercase = UCconversion(test_strings)
    print(f"Original: {test_strings}")
    print(f"Uppercase: {uppercase}")