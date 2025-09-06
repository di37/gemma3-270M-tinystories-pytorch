import os, sys
from os.path import dirname as up

sys.path.append(os.path.abspath(os.path.join(up(__file__), os.pardir)))

# Use the HuggingFace transformers library to load the google/gemma-3-270m tokenizer
from transformers import AutoTokenizer
import tiktoken


def processor_gemma3_tokenizer(example):
    """
    Tokenizes a batch or single example for HuggingFace Datasets .map() compatibility.
    This function is designed to work with both batched and non-batched calls.
    It returns a dictionary with 'ids' and 'len' fields, suitable for use with remove_columns.
    """
    # Load the tokenizer for google/gemma-3-270m
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m")
    # Check if 'text' is a list (batched) or a string (single example)
    texts = example['text']
    if isinstance(texts, list):
        # Batched mode: process each text in the batch
        ids = [tokenizer.encode(t, add_special_tokens=False) for t in texts]
        lens = [len(i) for i in ids]
        return {'ids': ids, 'len': lens}
    else:
        # Single example mode
        ids = tokenizer.encode(texts, add_special_tokens=False)
        return {'ids': ids, 'len': len(ids)}


# Some functions from https://github.com/karpathy/nanoGPT/blob/master/data/openwebtext/prepare.py

def processor_gpt2_tokenizer(example):
    enc = tiktoken.get_encoding("gpt2")
    ids = enc.encode_ordinary(example['text']) # encode_ordinary ignores any special tokens
    out = {'ids': ids, 'len': len(ids)}
    return out