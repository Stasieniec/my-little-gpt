import math
import os
import sys
import time
from pathlib import Path

# Add project root to python path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import fire
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from models import ByteTransformer
from data.enwik8_dataset import get_loader

def train_model(
    # Model parameters
    n_layer=8,
    n_head=8,
    n_embd=512,
    dropout=0.1,
    
    # Training parameters
    batch_size=64,
    block_size=256,
    learning_rate=3e-4,
    weight_decay=0.1,
    epochs=5,
    warmup_epochs=1,
    
    # Optimization parameters
    betas=(0.9, 0.95),
    grad_clip=1.0,
    
    # Logging parameters
    log_interval=100,
    checkpoint_interval=1000,
    
    # Cuda
    device="cuda" if torch.cuda.is_available() else "cpu",
    num_workers=4,
    
    # Naming
    checkpoint_dir="checkpoints",
    experiment_name=None,
    
    # wandb parameters
    wandb_project="my-little-gpt",
    wandb_entity='wasilewski-sf',
    use_wandb=True,
):


    # Experiment name and directory
    if experiment_name is None:
        experiment_name = f"enwik8_transformer_{time.strftime('%Y%m%d_%H%M%S')}"
    
    checkpoint_path = Path(checkpoint_dir) / experiment_name
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    # Experiment tracking with wandb
    if use_wandb:
        config = {
            "n_layer": n_layer,
            "n_head": n_head,
            "n_embd": n_embd,
            "dropout": dropout,
            "block_size": block_size,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "epochs": epochs,
            "warmup_epochs": warmup_epochs,
            "betas": betas,
            "grad_clip": grad_clip,
            "device": device,
        }
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=experiment_name,
            config=config,
        )
    
    # Dataloaders
    print(f"Loading the dataset. block_size={block_size}, batch_size={batch_size}")
    train_loader = get_loader(
        split="train",
        block_size=block_size,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )
    
    val_loader = get_loader(
        split="val",
        block_size=block_size,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )
    
    # Model, optimizer, and scheduler
    print(f"Creating ByteTransformer with {n_layer} layers, {n_head} heads, {n_embd} dims")
    model = ByteTransformer(
        vocab_size=256,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
    ).to(device)
    
    # Number of parameters
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {param_count:,}")
    
    if use_wandb:
        wandb.watch(model, log="all", log_freq=log_interval)
    
    # Set up optimizer with weight decay separation
    # (Only apply weight decay to non-bias/norm parameters)
    decay_params = []
    nodecay_params = []
    
    for name, param in model.named_parameters():
        if "bias" in name or "ln" in name or "norm" in name:
            nodecay_params.append(param)
        else:
            decay_params.append(param)
    
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    
    optimizer = Adam(optim_groups, lr=learning_rate, betas=betas)
    
    # Cosine learning rate scheduler with warmup
    total_steps = len(train_loader) * epochs
    warmup_steps = len(train_loader) * warmup_epochs
    
    def lr_lambda(step):
        # Linear warmup followed by cosine decay
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Initialize training state
    global_step = 0
    best_val_loss = float('inf')
    
    # Save config
    config = {
        "n_layer": n_layer,
        "n_head": n_head,
        "n_embd": n_embd,
        "dropout": dropout,
        "block_size": block_size,
        "vocab_size": 256,
        "experiment_name": experiment_name,
    }
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_losses = []
        start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            
            # Forward pass
            logits, loss = model(inputs, targets)
            
            # bpb = loss / ln(2) - conversion from nats to bits
            bpb = loss.item() / math.log(2)
            train_losses.append(bpb)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (no exploding gradients)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            optimizer.step()
            scheduler.step()
            
            # Logging
            if global_step % log_interval == 0:
                lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - start_time
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f} | BPB: {bpb:.4f} | "
                      f"LR: {lr:.6f} | {elapsed:.2f}s elapsed")
                
                if use_wandb:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/bpb": bpb,
                        "train/lr": lr,
                        "train/epoch": epoch + (batch_idx / len(train_loader)),
                        "train/global_step": global_step,
                    }, step=global_step)
                
                start_time = time.time()
            
            # Save checkpoint
            if global_step % checkpoint_interval == 0:
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "config": config,
                    "global_step": global_step,
                    "epoch": epoch,
                }
                
                checkpoint_file = checkpoint_path / f"step_{global_step}.pt"
                torch.save(checkpoint, checkpoint_file)
                print(f"Saved checkpoint to {checkpoint_file}")
                
                if use_wandb:
                    wandb.save(str(checkpoint_file))
            
            global_step += 1
        
        # Validation at the end of each epoch
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["input"].to(device)
                targets = batch["target"].to(device)
                
                logits, loss = model(inputs, targets)
                val_losses.append(loss.item() / math.log(2))  # Convert to BPB
        
        avg_val_loss = np.mean(val_losses)
        print(f"Epoch {epoch+1}/{epochs} validation: BPB: {avg_val_loss:.4f}")
        
        if use_wandb:
            wandb.log({
                "val/bpb": avg_val_loss,
                "val/epoch": epoch + 1,
            }, step=global_step)
        
        # Save checkpoint if validation loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "config": config,
                "global_step": global_step,
                "epoch": epoch,
                "val_loss": best_val_loss,
            }
            
            best_checkpoint_file = checkpoint_path / "best_model.pt"
            torch.save(checkpoint, best_checkpoint_file)
            print(f"New best validation loss: {best_val_loss:.4f}, saved to {best_checkpoint_file}")
            
            if use_wandb:
                wandb.save(str(best_checkpoint_file))
    
    # Save final model
    final_checkpoint = {
        "model": model.state_dict(),
        "config": config,
        "val_loss": best_val_loss,
    }
    
    final_checkpoint_file = checkpoint_path / "final_model.pt"
    torch.save(final_checkpoint, final_checkpoint_file)
    print(f"Training complete. Final model saved to {final_checkpoint_file}")
    
    # Log final model to wandb
    if use_wandb:
        model_artifact = wandb.Artifact(f"model-{experiment_name}", type="model")
        model_artifact.add_file(str(final_checkpoint_file))
        wandb.log_artifact(model_artifact)
        
        wandb.finish()
    
    return final_checkpoint_file

if __name__ == "__main__":
    fire.Fire(train_model)
