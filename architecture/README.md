# Architecture Module

This module contains the main Gemma3 model implementation and configuration management.

## Files

### `gemma3.py`
The core Gemma3Model class implementation featuring:

- **Token Embeddings**: Scaled embedding layer with vocabulary size of 50,257
- **Transformer Blocks**: 18 layers with mixed attention patterns (sliding window and full attention)
- **Dual RoPE**: Two sets of rotary position embeddings for local and global context
- **Attention Masks**: Dynamic generation of causal and sliding window masks
- **Output Head**: Linear projection to vocabulary size for next-token prediction
- **Generation Method**: Temperature-controlled sampling with top-k filtering

Key components:
- `__init__`: Initializes model layers, embeddings, and precomputes RoPE parameters
- `_create_masks`: Generates causal and sliding window attention masks
- `forward`: Main forward pass with optional loss computation
- `generate`: Autoregressive text generation with temperature and top-k sampling

### `model_config.py`
Configuration loader that reads model hyperparameters from `config/model_config.json`.

### `__init__.py`
Module initialization that exports:
- `model_config`: Dictionary containing all model hyperparameters
- `Gemma3Model`: The main model class

## Model Architecture Details

### Layer Configuration
The model uses a strategic mix of attention types across 18 layers:
- **Layers 1-5**: Sliding window attention (512 token window)
- **Layer 6**: Full attention (checkpoint layer)
- **Layers 7-11**: Sliding window attention
- **Layer 12**: Full attention (checkpoint layer)
- **Layers 13-17**: Sliding window attention
- **Layer 18**: Full attention (final layer)

This pattern allows the model to:
- Efficiently process local context with sliding windows
- Capture long-range dependencies at strategic checkpoints
- Balance computational efficiency with modeling capability

### Embedding and Normalization
- **Embedding Scaling**: Input embeddings are scaled by √(embedding_dim) for training stability
- **Final Normalization**: RMS normalization before the output projection
- **Weight Tying**: Output projection weights are separate from input embeddings

### Position Encoding
The model uses dual RoPE (Rotary Position Embeddings):
- **Local RoPE**: θ_base = 10,000 for sliding window attention
- **Global RoPE**: θ_base = 1,000,000 for full attention layers

This dual approach allows different attention patterns to use position encodings optimized for their respective context ranges.

## Usage Example

```python
from architecture import Gemma3Model, model_config
import torch

# Initialize model
model = Gemma3Model(model_config)

# Forward pass
input_ids = torch.randint(0, 50257, (2, 128))  # batch_size=2, seq_len=128
logits, loss = model(input_ids, targets=None)

# Generation
prompt = torch.randint(0, 50257, (1, 10))  # Single prompt
generated = model.generate(prompt, max_new_tokens=50, temperature=0.8, top_k=40)
```

## Design Decisions

1. **Mixed Attention**: Combines efficiency of sliding windows with the modeling power of full attention
2. **Separate RoPE Bases**: Optimizes position encoding for different attention ranges
3. **Grouped Query Attention**: Reduces KV cache memory while maintaining performance
4. **Gemma3-style Normalization**: Uses (1 + weight) scaling for better training dynamics