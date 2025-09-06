# Training Module

This module contains utilities for model training, including batch generation, loss computation, and configuration management.

## Components

### `training_config.py`
Loads and manages training hyperparameters from JSON configuration.

**Key Variables:**
- `learning_rate`: Base learning rate (1e-4)
- `max_iters`: Total training iterations (150,000)
- `warmup_steps`: Linear warmup period (1,000)
- `min_lr`: Minimum learning rate for cosine decay (5e-4)
- `eval_iters`: Validation frequency (500)
- `batch_size`: Training batch size (32)
- `block_size`: Sequence length (128)
- `gradient_accumulation_steps`: Gradient accumulation (32)
- `device`: Training device (cuda/cpu)
- `dtype`: Data type for training (bfloat16)
- `ctx`: Autocast context for mixed precision

**Features:**
- Automatic device selection based on CUDA availability
- Mixed precision context management
- Data type compatibility checks

### `get_batch.py`
Handles batch generation from memory-mapped datasets.

**Function: `get_batch(split, block_size, batch_size, device, device_type)`**

**Parameters:**
- `split`: "train" or "val" for dataset selection
- `block_size`: Length of sequences to extract
- `batch_size`: Number of sequences per batch
- `device`: Target device for tensors
- `device_type`: "cuda" or "cpu" for optimization

**Returns:**
- `x`: Input sequences [batch_size, block_size]
- `y`: Target sequences (shifted by 1) [batch_size, block_size]

**Features:**
- Memory-mapped file access (minimal RAM usage)
- Random sampling from dataset
- Pinned memory for faster GPU transfer
- Recreates memmap each call to prevent memory leaks

### `loss.py`
Computes training and validation losses.

**Function: `estimate_loss(model, eval_iters, ctx, block_size, batch_size, device, device_type)`**

**Parameters:**
- `model`: The model to evaluate
- `eval_iters`: Number of iterations for averaging
- `ctx`: Autocast context for mixed precision
- `block_size`: Sequence length
- `batch_size`: Batch size for evaluation
- `device`: Computation device
- `device_type`: Device type string

**Returns:**
- Dictionary with 'train' and 'val' average losses

**Features:**
- Evaluates on both training and validation sets
- Uses inference mode for efficiency
- Averages over multiple batches for stable estimates
- Automatically switches model to eval/train mode

### `__init__.py`
Module initialization that exports all training utilities:
- Configuration variables
- `get_batch` function
- `estimate_loss` function

## Training Pipeline

### 1. Configuration Loading
```python
from training import (
    learning_rate, max_iters, warmup_steps, 
    batch_size, block_size, device, ctx
)
```

### 2. Optimizer Setup
```python
# AdamW with weight decay
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=learning_rate,
    betas=(0.9, 0.95),
    weight_decay=0.1
)

# Learning rate scheduling
scheduler_warmup = LinearLR(optimizer, total_iters=warmup_steps)
scheduler_decay = CosineAnnealingLR(optimizer, T_max=max_iters-warmup_steps)
scheduler = SequentialLR(optimizer, [scheduler_warmup, scheduler_decay])
```

### 3. Training Loop
```python
for epoch in range(max_iters):
    # Get batch
    X, y = get_batch("train", block_size, batch_size, device, device_type)
    
    # Forward pass with mixed precision
    with ctx:
        logits, loss = model(X, y)
        loss = loss / gradient_accumulation_steps
        
    # Backward pass
    scaler.scale(loss).backward()
    
    # Gradient accumulation
    if (epoch + 1) % gradient_accumulation_steps == 0:
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    
    # Learning rate scheduling
    scheduler.step()
```

### 4. Validation
```python
if epoch % eval_iters == 0:
    losses = estimate_loss(model, eval_iters, ctx, ...)
    print(f"Train loss: {losses['train']:.4f}")
    print(f"Val loss: {losses['val']:.4f}")
```

## Optimization Techniques

### Mixed Precision Training
- Uses bfloat16 for faster computation
- GradScaler for float16 (automatic)
- Maintains float32 master weights

### Gradient Accumulation
- Simulates larger batch sizes
- 32 steps × 32 batch size = 1024 effective batch
- Reduces memory requirements

### Learning Rate Schedule
1. **Linear Warmup**: 0 → learning_rate over 1000 steps
2. **Cosine Annealing**: learning_rate → min_lr over remaining steps

### Gradient Clipping
- Maximum gradient norm: 0.5
- Prevents exploding gradients
- Stabilizes training

## Memory Optimization

### Data Loading
- Memory-mapped files don't load full dataset
- Only accessed portions cached by OS
- Allows training on large datasets

### Batch Processing
- Pinned memory for async GPU transfer
- Efficient tensor creation from numpy
- Non-blocking GPU operations

### Model Training
- Gradient accumulation reduces peak memory
- Mixed precision halves activation memory
- `set_to_none=True` in optimizer.zero_grad()

## Monitoring with Weights & Biases

The training script integrates with W&B for experiment tracking:

```python
with wandb.init(project="pretraining-gemma3_270b", config=config) as run:
    run.watch(model, log_freq=100)
    
    # Log metrics
    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "learning_rate": current_lr,
        "grad_norm": grad_norm
    })
```

## Performance Tips

### Hardware Requirements
- **GPU**: NVIDIA GPU with 24GB+ VRAM recommended
- **RAM**: 16GB+ for data preprocessing
- **Storage**: SSD for faster data loading

### Training Speed
- ~10 hours for full 150K iterations over A6000
- Checkpoint regularly to resume training

### Debugging
- Start with smaller `max_iters` for testing
- Monitor gradient norms for instability
- Check validation loss for overfitting