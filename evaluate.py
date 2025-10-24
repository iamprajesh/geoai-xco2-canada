"""
Evaluation Script for ST-ViWT Model
Comprehensive model evaluation including metrics, visualizations, and analysis
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from st_viwt_model import STViWTModel, create_model
from train import XCO2Dataset, evaluate as evaluate_model
from torch.utils.data import DataLoader


def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive evaluation metrics."""
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    bias = np.mean(y_pred - y_true)
    
    metrics = {
        'R²': r2,
        'MAE (ppm)': mae,
        'RMSE (ppm)': rmse,
        'MAPE (%)': mape,
        'Bias (ppm)': bias
    }
    
    return metrics


def plot_scatter(y_true, y_pred, save_path):
    """Create scatter plot of predictions vs observations."""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.5, s=20, edgecolors='k', linewidths=0.5)
    
    # 1:1 line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='1:1 Line')
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred)
    
    # Add metrics text
    text_str = '\n'.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, family='monospace')
    
    ax.set_xlabel('Observed XCO₂ (ppm)', fontsize=14, weight='bold')
    ax.set_ylabel('Predicted XCO₂ (ppm)', fontsize=14, weight='bold')
    ax.set_title('ST-ViWT Predictions vs Observations', fontsize=16, weight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Scatter plot saved: {save_path}")
    plt.close()


def plot_residuals(y_true, y_pred, save_path):
    """Create residual analysis plots."""
    residuals = y_pred - y_true
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Residuals vs Predicted
    axes[0, 0].scatter(y_pred, residuals, alpha=0.5, s=20)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 0].set_xlabel('Predicted XCO₂ (ppm)', fontsize=12)
    axes[0, 0].set_ylabel('Residuals (ppm)', fontsize=12)
    axes[0, 0].set_title('Residuals vs Predicted', fontsize=14, weight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Residual distribution
    axes[0, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Residuals (ppm)', fontsize=12)
    axes[0, 1].set_ylabel('Frequency', fontsize=12)
    axes[0, 1].set_title('Residual Distribution', fontsize=14, weight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot', fontsize=14, weight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Residuals vs Observed
    axes[1, 1].scatter(y_true, residuals, alpha=0.5, s=20)
    axes[1, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1, 1].set_xlabel('Observed XCO₂ (ppm)', fontsize=12)
    axes[1, 1].set_ylabel('Residuals (ppm)', fontsize=12)
    axes[1, 1].set_title('Residuals vs Observed', fontsize=14, weight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Residual plots saved: {save_path}")
    plt.close()


def plot_training_history(history_path, save_path):
    """Plot training and validation loss curves."""
    df = pd.read_csv(history_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    axes[0].plot(df['epoch'], df['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(df['epoch'], df['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss (MSE)', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, weight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Learning rate
    axes[1].plot(df['epoch'], df['learning_rate'], color='green', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Learning Rate', fontsize=12)
    axes[1].set_title('Learning Rate Schedule', fontsize=14, weight='bold')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history saved: {save_path}")
    plt.close()


def evaluate_model_complete(
    model_path: str,
    data_path: str,
    output_dir: str = 'results'
):
    """
    Complete model evaluation pipeline.
    
    Args:
        model_path: Path to trained model checkpoint
        data_path: Path to test data
        output_dir: Output directory for results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_dir = output_dir / 'performance_metrics'
    figures_dir = output_dir / 'figures'
    metrics_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("ST-ViWT Model Evaluation")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Data: {data_path}")
    print()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Load checkpoint
    print("Loading model...")
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    # Create model
    model = create_model(config['model_architecture'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from epoch {checkpoint['epoch'] + 1}")
    print(f"Parameters: {model.count_parameters():,}\n")
    
    # Load data
    print("Loading test data...")
    data = np.load(data_path)
    spectrograms = data['spectrograms']
    features = data['features']
    targets = data['targets']
    has_data = data['has_data']
    
    # Create dataset and dataloader
    dataset = XCO2Dataset(spectrograms, features, targets, has_data)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    print(f"Test samples: {len(dataset)}\n")
    
    # Evaluate
    print("Evaluating model...")
    criterion = torch.nn.MSELoss()
    test_loss, predictions, actuals = evaluate_model(
        model, dataloader, criterion, device, desc="Evaluation"
    )
    
    print(f"\nTest Loss: {test_loss:.6f}")
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(actuals, predictions)
    
    print("\n" + "=" * 60)
    print("Performance Metrics")
    print("=" * 60)
    for key, value in metrics.items():
        print(f"{key:15s}: {value:.6f}")
    print("=" * 60)
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(metrics_dir / 'evaluation_metrics.csv', index=False)
    print(f"\nMetrics saved: {metrics_dir / 'evaluation_metrics.csv'}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    plot_scatter(
        actuals, predictions,
        figures_dir / 'scatter_plot.png'
    )
    
    plot_residuals(
        actuals, predictions,
        figures_dir / 'residual_analysis.png'
    )
    
    # Save predictions
    results_df = pd.DataFrame({
        'observed': actuals,
        'predicted': predictions,
        'residual': predictions - actuals
    })
    results_df.to_csv(metrics_dir / 'predictions.csv', index=False)
    print(f"Predictions saved: {metrics_dir / 'predictions.csv'}")
    
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Evaluate ST-ViWT model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint (.pth)')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to test data (.npz)')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    evaluate_model_complete(
        model_path=args.model,
        data_path=args.data,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()
