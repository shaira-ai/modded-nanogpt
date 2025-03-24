import tiktoken
import time
import ctypes
from ctypes import c_char_p, c_size_t, POINTER, c_uint, c_void_p

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

lib.makeBC.argtypes = [
    POINTER(c_char_p), # input string array
    POINTER(c_size_t), # input count
    POINTER(c_uint), # token id array
    c_size_t # input count
]

lib.makeBC.restype = c_size_t

lib.getAllMatches.argtypes = [
    c_size_t, # pointer to BakaCorasick
    c_char_p, # input string
    c_size_t, # input count
    POINTER(c_size_t) # result count
]
lib.getAllMatches.restype = POINTER(c_uint)

lib.getAllMatchesIncludingPastEnd.argtypes = [
    c_size_t, # pointer to BakaCorasick
    c_char_p, # input string
    c_size_t, # input count
    POINTER(c_size_t) # result count
]
lib.getAllMatchesIncludingPastEnd.restype = POINTER(c_uint)

lib.getBCMemoryUsage.argtypes = [c_size_t]
lib.getBCMemoryUsage.restype = c_size_t

def maybe_utf8_decode(data):
    try:
        return data.decode('utf-8')
    except UnicodeDecodeError:
        return None

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
    if not hasattr(tokenizer, "token_id_to_bytes"):
        token_id_to_bytes = [b""] * len(tokenizer.non_special_vocab_xd)
        for k, v in tokenizer._mergeable_ranks.items():
            if type(k) != bytes:
                k = k.encode('utf-8')
            token_id_to_bytes[v] = k
        tokenizer.token_id_to_bytes = token_id_to_bytes

def main():
    encoder = tiktoken.get_encoding("gpt2")
    prep_tokenizer(encoder)
    n_tokens = len(encoder.non_special_vocab_xd)
    # prepare the arguments for makeBC
    input_arr = (c_char_p * n_tokens)()
    lengths = (c_size_t * n_tokens)()
    token_ids = (c_uint * n_tokens)()
    print_ids = (309, 289, 339, 365, 255, 10662)
    for i, k in enumerate(encoder.non_special_vocab_xd):
        if type(k) != bytes:
            k = k.encode('utf-8')
        # if i in print_ids:
        #     print(f"token {i} is {k} with length {len(k)}")
        input_arr[i] = k
        lengths[i] = len(k)
        token_ids[i] = i
    token_id_to_bytes = {}
    for k in encoder.non_special_vocab_xd:
        if type(k) != bytes:
            k = k.encode('utf-8')
        token_id_to_bytes[encoder.non_special_vocab_xd[k]] = k
    blah = lib.makeBC(input_arr, lengths, token_ids, n_tokens)
    # print(blah)
    result_length = c_size_t()
    input_str = b"The quick brown fox jumps over the lazy dog."
    input_ptr = c_char_p(input_str)
    input_len = len(input_str)
    # time this
    start = time.time()
    result = lib.getAllMatches(blah, input_ptr, input_len, ctypes.byref(result_length))
    end = time.time()
    print(f"Time to get all matches: {end - start}")
    # print("hewwo")
    result_list = []
    for i in range(result_length.value):
        result_list.append(result[i])
    # print(result_list)
    token_list = []
    for x in range(0, len(result_list), 3):
        token_list.append(result_list[x:x+3])
    for x in token_list:
        print(x, token_id_to_bytes[x[2]],  len(token_id_to_bytes[x[2]]))
    print(lib.getBCMemoryUsage(blah))
    with open("pg844.txt", 'rb') as file:
        content = file.read()  # content is a bytes object
    start = time.time()
    result = lib.getAllMatches(blah, content, len(content), ctypes.byref(result_length))
    end = time.time()
    print(f"Time to get all matches: {end - start}")
    print(result_length.value)
    start = time.time()
    result = lib.getAllMatchesIncludingPastEnd(blah, content, len(content), ctypes.byref(result_length))
    end = time.time()
    print(f"Time to get all matches including past end: {end - start}")
    print(result_length.value)

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
    main()
