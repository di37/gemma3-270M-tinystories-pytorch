# Config Module

This module contains JSON configuration files that define model architecture and training hyperparameters.

## Configuration Files

### `model_config.json`
Defines the Gemma3 270M model architecture parameters.

**Core Parameters:**
- `vocab_size`: 50,257 - Size of the token vocabulary (GPT-2 tokenizer)
  - Note: Official Gemma3 270M uses 256,000 vocab size with its native tokenizer
- `context_length`: 32,768 - Maximum sequence length the model can process
- `emb_dim`: 640 - Dimension of token embeddings and model hidden states
- `n_heads`: 4 - Number of attention heads
- `n_layers`: 18 - Number of transformer blocks
- `hidden_dim`: 2,048 - Dimension of feedforward network intermediate layer
- `head_dim`: 256 - Dimension per attention head

**Attention Configuration:**
- `qk_norm`: true - Enables RMS normalization for queries and keys
- `n_kv_groups`: 1 - Number of key-value groups (1 = Multi-Query Attention)
- `query_pre_attn_scalar`: 256 - Scaling factor applied to queries before attention

**Position Encoding:**
- `rope_local_base`: 10,000 - RoPE theta base for sliding window attention
- `rope_base`: 1,000,000 - RoPE theta base for full attention layers
- `sliding_window`: 512 - Size of the sliding attention window

**Layer Types:**
- Array of 18 attention types: mix of "sliding_attention" and "full_attention"
- Full attention at layers 6, 12, and 18 for capturing long-range dependencies
- Sliding attention for all other layers for efficiency

**Data Type:**
- `dtype`: "bfloat16" - Default precision for model weights and computations

### `training_config.json`
Defines training hyperparameters and optimization settings.

**Learning Rate Schedule:**
- `learning_rate`: 1e-4 - Base learning rate
- `warmup_steps`: 1,000 - Linear warmup period
- `min_lr`: 5e-4 - Minimum learning rate for cosine annealing

**Training Duration:**
- `max_iters`: 150,000 - Total training iterations
- `eval_iters`: 500 - Frequency of validation evaluation

**Batch Configuration:**
- `batch_size`: 32 - Number of sequences per batch
- `block_size`: 128 - Sequence length for training chunks
- `gradient_accumulation_steps`: 32 - Steps before weight update

**Hardware Settings:**
- `device`: "cuda" - Target device for training
- `dtype`: "bfloat16" - Training precision
- `ptdtype`: "float32" - PyTorch tensor dtype

## Usage

The configuration files are loaded by their respective Python modules:

```python
# Model configuration loading (in architecture/model_config.py)
import json

with open('config/model_config.json', 'r') as f:
    model_config = json.load(f)

# Training configuration loading (in training/training_config.py)
with open('config/training_config.json', 'r') as f:
    training_config = json.load(f)
```

## Configuration Philosophy

### Model Design Choices
1. **Vocabulary Size**: Uses GPT-2's vocabulary (50K) for compatibility and fast tokenization
   - Official Gemma3 270M uses 256K vocabulary for better multilingual support
   - This implementation trades vocabulary size for training speed and efficiency
2. **Context Length**: 32K tokens enables processing of long documents (matches official)
3. **Embedding Dimension**: 640 provides a good balance between capacity and efficiency
4. **Mixed Attention**: Strategic placement of full attention layers maintains long-range modeling

### Training Strategy
1. **Learning Rate**: Conservative 1e-4 for stable training
2. **Warmup**: 1,000 steps prevents early training instability
3. **Gradient Accumulation**: 32 steps simulates larger batch sizes on limited hardware
4. **Mixed Precision**: bfloat16 speeds up training while maintaining numerical stability

## Modifying Configurations

When adjusting configurations:

### Model Architecture
- Ensure `head_dim * n_heads` is compatible with `emb_dim` if not explicitly set
- `layer_types` array must have exactly `n_layers` elements
- Sliding window should be significantly smaller than context length

### Training Parameters
- Adjust `batch_size` and `gradient_accumulation_steps` based on available GPU memory
- Scale `learning_rate` proportionally if changing effective batch size
- Ensure `device` and `dtype` are compatible with your hardware

## Performance Considerations

### Memory Usage
- Model size: ~270M parameters
- Activation memory scales with `batch_size * block_size * emb_dim`
- KV cache memory reduced by using `n_kv_groups=1` (Multi-Query Attention)

### Computational Efficiency
- Sliding window attention reduces quadratic complexity for most layers
- bfloat16 precision halves memory usage and speeds up computation
- Gradient accumulation allows training with limited GPU memory