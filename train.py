"""
Training Script for ST-ViWT Model
Handles complete training pipeline including data loading, training loop, and checkpointing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Tuple
import argparse
from tqdm import tqdm

from st_viwt_model import STViWTModel, create_model


class XCO2Dataset(Dataset):
    """PyTorch Dataset for XCO₂ reconstruction."""
    
    def __init__(self, spectrograms, features, targets, has_data):
        """
        Args:
            spectrograms: Array of shape (n, H, W)
            features: Array of shape (n, num_features)
            targets: Array of shape (n,)
            has_data: Boolean mask of shape (n,)
        """
        self.spectrograms = torch.FloatTensor(spectrograms)
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.has_data = torch.BoolTensor(has_data)
        
    def __len__(self):
        return len(self.spectrograms)
    
    def __getitem__(self, idx):
        return {
            'spectrogram': self.spectrograms[idx].unsqueeze(0),  # Add channel dim
            'aux_features': self.features[idx],
            'target': self.targets[idx],
            'has_data': self.has_data[idx]
        }


def train_epoch(model, dataloader, optimizer, criterion, device, desc="Training"):
    """
    Train for one epoch.
    
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=desc)
    for batch in pbar:
        # Get data
        spec = batch['spectrogram'].to(device)
        aux = batch['aux_features'].to(device)
        target = batch['target'].to(device)
        mask = batch['has_data'].to(device)
        
        # Filter valid samples
        if mask.sum() == 0:
            continue
        
        spec = spec[mask]
        aux = aux[mask]
        target = target[mask]
        
        # Forward pass
        optimizer.zero_grad()
        pred = model(spec, aux).squeeze()
        
        # Handle single sample case
        if pred.dim() == 0:
            pred = pred.unsqueeze(0)
        if target.dim() == 0:
            target = target.unsqueeze(0)
        
        # Compute loss
        loss = criterion(pred, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})
    
    return total_loss / max(num_batches, 1)


def evaluate(model, dataloader, criterion, device, desc="Validation"):
    """
    Evaluate model on validation/test set.
    
    Returns:
        avg_loss, predictions, actuals
    """
    model.eval()
    total_loss = 0
    num_batches = 0
    predictions = []
    actuals = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=desc)
        for batch in pbar:
            spec = batch['spectrogram'].to(device)
            aux = batch['aux_features'].to(device)
            target = batch['target'].to(device)
            mask = batch['has_data'].to(device)
            
            if mask.sum() == 0:
                continue
            
            spec = spec[mask]
            aux = aux[mask]
            target = target[mask]
            
            pred = model(spec, aux).squeeze()
            
            if pred.dim() == 0:
                pred = pred.unsqueeze(0)
            if target.dim() == 0:
                target = target.unsqueeze(0)
            
            loss = criterion(pred, target)
            total_loss += loss.item()
            num_batches += 1
            
            predictions.extend(pred.cpu().numpy().flatten())
            actuals.extend(target.cpu().numpy().flatten())
            
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
    
    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss, np.array(predictions), np.array(actuals)


def train_model(
    config_path: str,
    data_path: str,
    output_dir: str = 'outputs',
    resume_from: str = None
):
    """
    Complete training pipeline.
    
    Args:
        config_path: Path to training configuration JSON
        data_path: Path to preprocessed data (.npz file)
        output_dir: Directory for outputs
        resume_from: Optional checkpoint path to resume training
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print("=" * 60)
    print("ST-ViWT Training Pipeline")
    print("=" * 60)
    print(f"Configuration: {config_path}")
    print(f"Data: {data_path}")
    print(f"Output: {output_dir}")
    print()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Load data
    print("Loading data...")
    data = np.load(data_path)
    spectrograms = data['spectrograms']
    features = data['features']
    targets = data['targets']
    has_data = data['has_data']
    print(f"  Spectrograms: {spectrograms.shape}")
    print(f"  Features: {features.shape}")
    print(f"  Targets: {targets.shape}")
    print(f"  Valid samples: {has_data.sum()}/{len(has_data)}")
    print()
    
    # Split data
    from sklearn.model_selection import train_test_split
    
    indices = np.arange(len(has_data))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=config['data']['validation_split'],
        random_state=config['data']['random_seed']
    )
    
    # Create datasets
    train_dataset = XCO2Dataset(
        spectrograms[train_idx],
        features[train_idx],
        targets[train_idx],
        has_data[train_idx]
    )
    
    val_dataset = XCO2Dataset(
        spectrograms[val_idx],
        features[val_idx],
        targets[val_idx],
        has_data[val_idx]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=config.get('pin_memory', True)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=config.get('pin_memory', True)
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Batches per epoch: {len(train_loader)}")
    print()
    
    # Create model
    print("Creating model...")
    model = create_model(config['model_architecture'])
    model = model.to(device)
    
    total_params = model.count_parameters()
    print(f"  Total parameters: {total_params:,}")
    print()
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['num_epochs'],
        eta_min=config['training']['scheduler_params']['eta_min']
    )
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_val_loss = float('inf')
    
    if resume_from and Path(resume_from).exists():
        print(f"Resuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"  Resuming from epoch {start_epoch}")
        print()
    
    # Training loop
    print("Starting training...")
    print("=" * 60)
    
    training_history = []
    
    for epoch in range(start_epoch, config['training']['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['training']['num_epochs']}")
        print("-" * 60)
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device,
            desc=f"Epoch {epoch + 1} [Train]"
        )
        
        # Validate
        val_loss, val_preds, val_actuals = evaluate(
            model, val_loader, criterion, device,
            desc=f"Epoch {epoch + 1} [Val]"
        )
        
        # Update scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Log results
        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss:   {val_loss:.6f}")
        print(f"  LR:         {current_lr:.2e}")
        
        # Save history
        history_entry = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': current_lr,
            'is_best': val_loss < best_val_loss
        }
        training_history.append(history_entry)
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
            'config': config
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, output_dir / 'latest_checkpoint.pth')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, output_dir / 'best_model.pth')
            print(f"  ✓ New best model saved! (val_loss: {val_loss:.6f})")
    
    # Save training history
    history_df = pd.DataFrame(training_history)
    history_df.to_csv(output_dir / 'training_history.csv', index=False)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved to: {output_dir / 'best_model.pth'}")
    print(f"Training history: {output_dir / 'training_history.csv'}")


def main():
    parser = argparse.ArgumentParser(description='Train ST-ViWT model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration JSON file')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to preprocessed data (.npz)')
    parser.add_argument('--output', type=str, default='outputs',
                       help='Output directory for checkpoints and logs')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    train_model(
        config_path=args.config,
        data_path=args.data,
        output_dir=args.output,
        resume_from=args.resume
    )


if __name__ == "__main__":
    main()
