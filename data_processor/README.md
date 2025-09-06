# Data Processor Module

This module handles tokenization and data preprocessing for the Gemma3 270M model training pipeline.

## Components

### `process.py`
Contains tokenization functions for converting text to token IDs.

### `__init__.py`
Module initialization that exports tokenization functions:
- `processor_gemma3_tokenizer`: Gemma3 model tokenizer
- `processor_gpt2_tokenizer`: GPT-2 tokenizer (default)

## Tokenization Functions

### `processor_gemma3_tokenizer`
Tokenizes text using Google's Gemma3 270M tokenizer from Hugging Face.

**Features:**
- Uses AutoTokenizer from transformers library
- Loads "google/gemma-3-270m" tokenizer
- Supports both single and batched processing
- Returns dictionary with 'ids' and 'len' fields

**Parameters:**
- `example`: Dictionary with 'text' field (single string or list of strings)

**Returns:**
- Dictionary with:
  - `ids`: Token IDs (list or list of lists)
  - `len`: Length of tokenized sequences

**Usage:**
```python
from data_processor import processor_gemma3_tokenizer

# Single text
result = processor_gemma3_tokenizer({'text': 'Hello world'})
# result = {'ids': [token_ids], 'len': length}

# Batched texts
result = processor_gemma3_tokenizer({'text': ['Hello', 'World']})
# result = {'ids': [[ids1], [ids2]], 'len': [len1, len2]}
```

### `processor_gpt2_tokenizer`
Tokenizes text using the GPT-2 tokenizer via tiktoken.

**Features:**
- Uses tiktoken library for efficient tokenization
- GPT-2 vocabulary (50,257 tokens)
- Ignores special tokens with `encode_ordinary`
- Simpler single-text processing

**Parameters:**
- `example`: Dictionary with 'text' field

**Returns:**
- Dictionary with:
  - `ids`: List of token IDs
  - `len`: Length of tokenized sequence

**Usage:**
```python
from data_processor import processor_gpt2_tokenizer

result = processor_gpt2_tokenizer({'text': 'Hello world'})
# result = {'ids': [15496, 995], 'len': 2}
```

## Tokenizer Comparison

### GPT-2 Tokenizer (Default)
- **Vocabulary Size**: 50,257
- **Library**: tiktoken (fast, efficient)
- **Special Tokens**: Ignored by default
- **Use Case**: General English text, compatible with many models
- **Performance**: ~10x faster tokenization speed

### Gemma3 Tokenizer (Official)
- **Vocabulary Size**: 256,000 (much larger)
- **Library**: Hugging Face transformers
- **Special Tokens**: Configurable
- **Use Case**: Better for multilingual, technical, or specialized text
- **Performance**: Slower due to large vocabulary
- **Official Model**: Uses this tokenizer with 170M embedding parameters

## Why GPT-2 Tokenizer is Used by Default

The project uses GPT-2 tokenizer instead of Gemma3's native tokenizer for several practical reasons:

1. **Tokenization Speed**: The Gemma3 tokenizer with its 256K vocabulary takes significantly longer to tokenize large datasets. For the TinyStories dataset (600M+ tokens), this difference can mean hours vs minutes of preprocessing time.

2. **English-Only Dataset**: Since TinyStories is an English-only dataset, the multilingual capabilities of Gemma3's large vocabulary are unnecessary. GPT-2's 50K vocabulary is sufficient for English text.

3. **Memory Efficiency**: The smaller vocabulary reduces embedding layer size from 170M to ~32M parameters, making the model more manageable for training on consumer GPUs.

4. **Training Efficiency**: Smaller vocabulary means faster softmax computation during training and inference, reducing overall training time.

5. **Compatibility**: GPT-2 tokenizer is widely supported and well-tested, making the model easier to integrate with existing tools and pipelines.

## Integration with Data Pipeline

The tokenizers are used in `01_data_preparation.py`:

```python
from data_processor import processor_gpt2_tokenizer

# Tokenize dataset
tokenized = ds.map(
    processor_gpt2_tokenizer,
    remove_columns=['text'],
    desc="tokenizing the splits",
    num_proc=8,  # Parallel processing
)
```

## Performance Considerations

### Batch Processing
- Gemma3 tokenizer supports batched processing for efficiency
- GPT-2 tokenizer processes single examples (parallelized via dataset.map)

### Memory Usage
- Tokenized IDs stored as lists of integers
- Converted to uint16 for storage (saves 50% memory)

### Speed
- tiktoken (GPT-2): Very fast C++ implementation
- Transformers (Gemma3): Python-based, slower but more features

## Switching Tokenizers

To switch from GPT-2 to Gemma3 tokenizer:

1. Update `01_data_preparation.py`:
```python
# Change from:
from data_processor import processor_gpt2_tokenizer
tokenized = ds.map(processor_gpt2_tokenizer, ...)

# To:
from data_processor import processor_gemma3_tokenizer
tokenized = ds.map(processor_gemma3_tokenizer, ...)
```

2. Update model configuration:
```json
// In config/model_config.json
"vocab_size": 256000,  // Update to match Gemma3 tokenizer (official size)
```

3. Regenerate processed datasets:
```bash
rm -rf data/processed_datasets/*.bin
python 01_data_preparation.py
```

## Custom Tokenizers

To add a custom tokenizer:

1. Create a new processor function in `process.py`:
```python
def processor_custom_tokenizer(example):
    # Your tokenization logic
    ids = custom_tokenize(example['text'])
    return {'ids': ids, 'len': len(ids)}
```

2. Export it in `__init__.py`:
```python
from .process import processor_custom_tokenizer
```

3. Update vocabulary size in model config if needed

## Dependencies

- **tiktoken**: For GPT-2 tokenization
- **transformers**: For Gemma3 and other Hugging Face tokenizers
- **numpy**: For efficient array operations

Install with:
```bash
pip install tiktoken transformers
```