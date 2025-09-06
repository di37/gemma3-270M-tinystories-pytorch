import os
from datasets import load_dataset
from data_processor import processor_gemma3_tokenizer, processor_gpt2_tokenizer
import numpy as np
from tqdm.auto import tqdm

ds = load_dataset("roneneldan/TinyStories")

if not os.path.exists("data/processed_datasets/train.bin"):
    tokenized = ds.map(
        processor_gpt2_tokenizer,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=8,
        )
    
    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = f'data/processed_datasets/{split}.bin'
        dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

