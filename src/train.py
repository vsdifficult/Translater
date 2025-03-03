import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from datetime import datetime

from model import TranslationModel
from tokenizer import TranslationTokenizer
from dataset import get_translation_dataloader
from config import TranslationConfig

def train(config_path):
    """Train the translation model."""
    
    # Load configuration
    config = TranslationConfig.load(config_path)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(config.output_dir, f"{config.source_lang}_{config.target_lang}_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)
    
    # Setup tensorboard
    tensorboard_dir = os.path.join(config.tensorboard_dir, f"{config.source_lang}_{config.target_lang}_{timestamp}")
    writer = SummaryWriter(tensorboard_dir)
    
    # Initialize tokenizer
    tokenizer = TranslationTokenizer(config)
    
    # Update vocab sizes in config
    config.update_vocab_sizes(tokenizer)
    
    # Save tokenizer
    tokenizer.save_tokenizers(model_dir)
    
    # Save config
    config.save(os.path.join(model_dir, "config.json"))
    
    # Get dataloaders
    train_dataloader = get_translation_dataloader(tokenizer, config, split="train")
    val_dataloader = get_translation_dataloader(tokenizer, config, split="validation", shuffle=False)
    
    # Initialize model
    model = TranslationModel(config)
    
    # Move model to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Setup optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Setup loss function
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.target_tokenizer.pad_token_id)
    
    # Calculate total training steps
    total_steps = len(train_dataloader) * config.num_epochs // config.gradient_accumulation_steps
    
    # Training loop
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0
        
        # Training
        with tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}") as pbar:
            for step, batch in enumerate(pbar):
                # Move batch to device
                source_ids = batch['source_ids'].to(device)
                source_mask = batch['source_mask'].to(device)
                target_ids = batch['target_ids'].to(device)
                target_mask = batch['target_mask'].to(device)
                
                # Forward pass
                logits = model(source_ids, target_ids[:, :-1], source_mask, target_mask[:, :-1])
                
                # Calculate loss
                loss = loss_fn(logits.reshape(-1, config.target_vocab_size), target_ids[:, 1:].reshape(-1))
                
                # Scale loss for gradient accumulation
                loss = loss / config.gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Update metrics
                epoch_loss += loss.item() * config.gradient_accumulation_steps
                
                # Update progress bar
                pbar.set_postfix({'loss': epoch_loss / (step + 1)})
                
                # Gradient accumulation
                if (step + 1) % config.gradient_accumulation_steps == 0:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    
                    # Update weights
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    # Log to tensorboard
                    writer.add_scalar('Training/Loss', loss.item() * config.gradient_accumulation_steps, global_step)
                    
                    global_step += 1
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            with tqdm(val_dataloader, desc="Validation") as pbar:
                for step, batch in enumerate(pbar):
                    # Move batch to device
                    source_ids = batch['source_ids'].to(device)
                    source_mask = batch['source_mask'].to(device)
                    target_ids = batch['target_ids'].to(device)
                    target_mask = batch['target_mask'].to(device)
                    
                    # Forward pass
                    logits = model(source_ids, target_ids[:, :-1], source_mask, target_mask[:, :-1])
                    
                    # Calculate loss
                    loss = loss_fn(logits.reshape(-1, config.target_vocab_size), target_ids[:, 1:].reshape(-1))
                    
                    # Update metrics
                    val_loss += loss.item()
                    
                    # Update progress bar
                    pbar.set_postfix({'loss': val_loss / (step + 1)})
        
        # Calculate average validation loss
        val_loss = val_loss / len(val_dataloader)
        
        # Log to tensorboard
        writer.add_scalar('Validation/Loss', val_loss, epoch)
        
        print(f"Epoch {epoch+1}/{config.num_epochs} - Validation Loss: {val_loss:.4f}")
        
        # Save model if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            # Save model
            torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pt"))
            print(f"Model saved to {os.path.join(model_dir, 'best_model.pt')}")
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(model_dir, "final_model.pt"))
    
    # Close tensorboard writer
    writer.close()
    
    print(f"Training completed. Model saved to {model_dir}")
    
    return model_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a neural translation model")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    
    args = parser.parse_args()
    
    train(args.config)