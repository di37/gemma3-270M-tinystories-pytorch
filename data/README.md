# Data Module

This directory contains processed datasets and trained model checkpoints.

## Directory Structure

### `models/`
Stores trained model checkpoints and parameters.

**Contents:**
- `best_model_params.pt`: Best model checkpoint based on validation loss
- Other checkpoint files from training runs

**Checkpoint Format:**
- PyTorch state dictionary format
- Contains all model parameters
- Can be loaded with `torch.load()` and `model.load_state_dict()`

### `processed_datasets/`
Contains preprocessed and tokenized training data in binary format.

**Files:**
- `train.bin`: Training dataset in memory-mapped format
- `validation.bin`: Validation dataset in memory-mapped format

**Data Format:**
- NumPy memmap files with dtype `uint16`
- Each file contains concatenated token IDs
- Efficient for streaming during training

## Data Pipeline

### 1. Raw Data Source
The project uses the **TinyStories** dataset from Hugging Face:
- Simple, coherent stories for language model training
- Designed for training small models efficiently
- Loaded via `datasets` library

### 2. Tokenization
Data is tokenized using one of two tokenizers:
- **GPT-2 Tokenizer** (default): 50,257 vocabulary size
  - Chosen for speed: ~10x faster than Gemma3 tokenizer
  - Sufficient for English-only TinyStories dataset
  - Reduces preprocessing time from hours to minutes
- **Gemma3 Tokenizer** (optional): Google's Gemma3 270M tokenizer with 256K vocabulary
  - Better for multilingual content but overkill for English text
  - Significantly slower due to large vocabulary size

### 3. Binary Format Conversion
Tokenized data is converted to binary format for efficiency:
```python
# Memory-mapped array for efficient access
arr = np.memmap(filename, dtype=np.uint16, mode='w+', shape=(arr_len,))
```

Benefits:
- Minimal memory usage during training
- Fast random access for batch creation
- Efficient storage (uint16 for token IDs)

## Data Statistics

### TinyStories Dataset
- **Training samples**: ~2.1M stories
- **Validation samples**: ~22K stories
- **Average story length**: ~300 tokens
- **Total tokens**: ~600M+ tokens

### Processed Binary Files
- **train.bin**: ~1.2GB (depending on tokenizer)
- **validation.bin**: ~13MB
- **Token range**: 0-50,256 (fits in uint16)

## Loading Data

### During Training
Data is loaded using memory-mapped arrays:
```python
def get_batch(split, block_size, batch_size, device):
    if split == 'train':
        data = np.memmap('data/processed_datasets/train.bin', dtype=np.uint16, mode='r')
    else:
        data = np.memmap('data/processed_datasets/validation.bin', dtype=np.uint16, mode='r')
    
    # Random sampling
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # Create batch
    x = torch.stack([torch.from_numpy(data[i:i+block_size]) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size]) for i in ix])
    return x, y
```

### Loading Model Checkpoints
```python
import torch
from architecture import Gemma3Model, model_config

# Initialize model
model = Gemma3Model(model_config)

# Load checkpoint
checkpoint = torch.load('data/models/best_model_params.pt')
model.load_state_dict(checkpoint)
```

## Storage Requirements

### Minimum Requirements
- **Training data**: ~1.2GB
- **Validation data**: ~15MB
- **Model checkpoints**: ~1GB per checkpoint

### Recommended Setup
- **SSD storage** for faster data loading
- **8GB+ available space** for multiple checkpoints
- **Memory-mapped files** stay on disk, minimal RAM usage

## Data Preparation Notes

### Running Data Preparation
Execute `01_data_preparation.py` to:
1. Download TinyStories dataset
2. Tokenize all samples
3. Concatenate tokens into continuous arrays
4. Save as memory-mapped binary files

### Custom Datasets
To use custom datasets:
1. Modify `01_data_preparation.py` to load your dataset
2. Ensure tokenization produces valid token IDs (0 to vocab_size-1)
3. Maintain the same binary format for compatibility
4. Split into train/validation sets

## Performance Considerations

### Memory Efficiency
- Memory-mapped files don't load entire dataset into RAM
- Only accessed portions are cached by OS
- Allows training on datasets larger than available RAM

### I/O Optimization
- Sequential access patterns during batch creation
- Random sampling ensures good data coverage
- Pinned memory for faster GPU transfers (when using CUDA)