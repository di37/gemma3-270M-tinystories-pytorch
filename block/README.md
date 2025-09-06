# Block Module

This module contains the fundamental building blocks of the Gemma3 transformer architecture.

## Components

### `attention.py` - Grouped Query Attention (GQA)
Implements the efficient attention mechanism with configurable key-value groups.

**Key Features:**
- **Grouped Queries**: Reduces memory by sharing key-value pairs across query heads
- **QK Normalization**: Optional RMS normalization of queries and keys for training stability
- **Query Scaling**: Configurable pre-attention scaling factor
- **RoPE Integration**: Applies rotary position embeddings to queries and keys

**Parameters:**
- `num_heads`: Number of attention heads (4 in Gemma3 270M)
- `num_kv_groups`: Number of key-value groups (1 = Multi-Query Attention)
- `head_dim`: Dimension per attention head (256)
- `qk_norm`: Whether to apply RMS normalization to Q and K
- `query_pre_attn_scalar`: Custom scaling factor for queries

### `feedforward.py` - Feedforward Network
Implements the feedforward network with GELU activation.

**Architecture:**
- Two parallel input projections (fc1, fc2)
- GELU activation with tanh approximation
- Multiplication of activated and projected features: `GELU(fc1(x)) * fc2(x)`
- Output projection back to model dimension

**Dimensions:**
- Input/Output: 640 (embedding dimension)
- Hidden: 2048 (intermediate dimension)

### `transformer.py` - Transformer Block
Combines attention and feedforward layers with residual connections and normalization.

**Structure:**
1. **Attention Sub-block:**
   - Input layer norm → Attention → Post-attention norm → Residual add
2. **Feedforward Sub-block:**
   - Pre-feedforward norm → FFN → Post-feedforward norm → Residual add

**Attention Types:**
- `sliding_attention`: Uses local mask and local RoPE
- `full_attention`: Uses global mask and global RoPE

The block automatically selects the appropriate mask and RoPE parameters based on its configured attention type.

### `rope.py` - Rotary Position Embeddings
Implements RoPE for encoding relative positions in attention.

**Functions:**
- `compute_rope_params`: Precomputes cos and sin matrices for a given context length
- `apply_rope`: Applies rotary embeddings to query/key tensors

**Features:**
- Configurable theta base for different frequency distributions
- Efficient precomputation and caching
- Supports different bases for local vs global attention

### `rms_norm.py` - Root Mean Square Normalization
Implements Gemma3-style RMS normalization.

**Key Features:**
- Zero-centered weight parameters
- (1 + weight) scaling during forward pass
- Float32 computation for numerical stability
- Optional bias parameter (not used in Gemma3)

**Formula:**
```
x_norm = x / sqrt(mean(x²) + eps)
output = x_norm * (1 + weight)
```

### `__init__.py`
Module initialization file for clean imports.

## Architecture Flow

```
Input → TransformerBlock:
    ├─→ [Residual] → InputLayerNorm → Attention → PostAttnNorm → Add
    └─→ [Residual] → PreFFNorm → FeedForward → PostFFNorm → Add → Output
```

## Key Design Principles

1. **Dual Normalization**: Each sub-block has pre and post normalization for gradient stability
2. **Flexible Attention**: Support for both sliding window and full attention patterns
3. **Memory Efficiency**: Grouped query attention reduces KV cache requirements
4. **Numerical Stability**: Strategic use of float32 in normalization layers
5. **Modular Design**: Each component is self-contained and reusable

## Usage Example

```python
from block import TransformerBlock
import torch

# Configuration
cfg = {
    "emb_dim": 640,
    "n_heads": 4,
    "n_kv_groups": 1,
    "head_dim": 256,
    "hidden_dim": 2048,
    "qk_norm": True,
    "query_pre_attn_scalar": 256,
    "dtype": torch.bfloat16
}

# Create a transformer block
block = TransformerBlock(cfg, attn_type="sliding_attention")

# Forward pass
x = torch.randn(2, 128, 640)  # [batch, seq_len, emb_dim]
# Assume masks and RoPE parameters are precomputed
output = block(x, mask_global, mask_local, cos_global, sin_global, cos_local, sin_local)
```

## Performance Optimizations

- **Precomputed RoPE**: Position embeddings computed once and cached
- **Efficient Masking**: Masks generated once per forward pass
- **Mixed Precision**: Strategic use of bfloat16 with float32 for critical operations
- **Fused Operations**: Utilizes PyTorch's optimized GELU and softmax implementations