## Loading Training configuration
from training import learning_rate, max_iters, warmup_steps, min_lr, eval_iters, batch_size, block_size, gradient_accumulation_steps, device, device_type, dtype, ctx
from torch.optim.lr_scheduler import LinearLR,SequentialLR, CosineAnnealingLR
import torch

## Model architecture
from architecture import model_config, Gemma3Model

# Observe progress
from tqdm.auto import tqdm

## Data Loading
from training import get_batch, estimate_loss


model_config["dtype"] = torch.bfloat16

torch.manual_seed(123)
model = Gemma3Model(model_config)

# For reproducibility 
torch.set_default_device(device)
torch.manual_seed(42)


##PUT IN WEIGHT DECAY, CHANGED BETA2 to 0.95
optimizer =  torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.1, eps=1e-9) #weight decay for regularization

scheduler_warmup = LinearLR(optimizer, total_iters = warmup_steps) #Implement linear warmup
scheduler_decay = CosineAnnealingLR(optimizer,T_max = max_iters - warmup_steps, eta_min = min_lr) #Implement lr decay
scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_decay], milestones=[warmup_steps]) #Switching from warmup to decay

# https://stackoverflow.com/questions/72534859/is-gradscaler-necessary-with-mixed-precision-training-with-pytorch
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))


# ### Pre-training the SLM 
import wandb
wandb.login()


training_config_selected_for_logs = {
    "learning_rate": learning_rate, 
    "max_iters": max_iters, #increase from 25000
    "warmup_steps": warmup_steps, #smoother initial train, earlier 100
    "min_lr": min_lr, #lower rate, earlier 5e-4
    "eval_iters": eval_iters, # increased from 100
    "batch_size": batch_size, # changed from 16, better gradient estimate
    "block_size": block_size, #changed from 64, capture longer range dependencies
}


best_val_loss = float('inf')
best_model_params_path = "data/models/best_model_params.pt"
train_loss_list, validation_loss_list = [], []


with wandb.init(project="pretraining-gemma3_270b", config=training_config_selected_for_logs) as run:
    # Ensure model is on the correct device
    model = model.to(device)

    # Magic
    run.watch(model, log_freq=100)

    # In your training loop
    for epoch in tqdm(range(max_iters)):
        if epoch % eval_iters == 0 and epoch != 0:
            # Ensure estimate_loss uses the correct device
            losses = estimate_loss(model, eval_iters, ctx, block_size, batch_size, device, device_type)
            train_loss = losses['train']
            val_loss = losses['val']
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            print(f"The current learning rate: {optimizer.param_groups[0]['lr']:.5f}")
            train_loss_list += [losses['train']]
            validation_loss_list += [losses['val']]

            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": current_lr,
                "best_val_loss": best_val_loss
            }, step=epoch)

            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                torch.save(model.state_dict(), best_model_params_path)
                wandb.log({"best_model_saved_at_epoch": epoch, "best_val_loss": best_val_loss}, step=epoch)

        # Ensure X and y are on the correct device
        X, y = get_batch("train", block_size, batch_size, device, device_type)
        X, y = X.to(device), y.to(device)

        with ctx:
            logits, loss = model(X, y)
            loss = loss / gradient_accumulation_steps
            scaler.scale(loss).backward()
            wandb.log({"batch_loss": loss.item()}, step=epoch)

        if ((epoch + 1) % gradient_accumulation_steps == 0) or (epoch + 1 == max_iters):
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            wandb.log({"grad_norm": grad_norm}, step=epoch)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        scheduler.step()

