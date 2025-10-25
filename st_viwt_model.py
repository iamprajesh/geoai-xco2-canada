"""
ST-ViWT Model Architecture
Spatio-Temporal Vision Transformer with Wavelet Transform for XCO₂ Reconstruction

This module implements the complete ST-ViWT framework combining:
- Vision Transformer for spatial pattern learning
- Wavelet spectrograms for temporal feature extraction
- Auxiliary feature fusion for enhanced predictions
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class PatchEmbedding(nn.Module):
    """
    Convert spectrogram into patches and embed them.
    
    Args:
        img_size: Size of input spectrogram (default: 64)
        patch_size: Size of each patch (default: 8)
        in_channels: Number of input channels (default: 1)
        embed_dim: Embedding dimension (default: 256)
    """
    def __init__(self, img_size: int = 64, patch_size: int = 8, 
                 in_channels: int = 1, embed_dim: int = 256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # Convolutional projection
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, 1, img_size, img_size)
        Returns:
            (batch_size, n_patches, embed_dim)
        """
        x = self.proj(x)  # (B, embed_dim, n_patches**0.5, n_patches**0.5)
        x = x.flatten(2)  # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, embed_dim)
        return x


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout rate (default: 0.1)
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, n_patches + 1, embed_dim)
        Returns:
            (batch_size, n_patches + 1, embed_dim)
        """
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        
        return x


class MLP(nn.Module):
    """
    Multi-layer perceptron with GELU activation.
    
    Args:
        in_features: Input dimension
        hidden_features: Hidden layer dimension
        out_features: Output dimension
        dropout: Dropout rate (default: 0.1)
    """
    def __init__(self, in_features: int, hidden_features: int, 
                 out_features: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer block with multi-head attention and MLP.
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        mlp_ratio: Ratio of MLP hidden dim to embedding dim (default: 4.0)
        dropout: Dropout rate (default: 0.1)
    """
    def __init__(self, embed_dim: int, num_heads: int, 
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(embed_dim, mlp_hidden_dim, embed_dim, dropout)
        
    def forward(self, x):
        # Multi-head attention with residual
        x = x + self.attn(self.norm1(x))
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer for processing wavelet spectrograms.
    
    Args:
        img_size: Size of input spectrogram (default: 64)
        patch_size: Size of each patch (default: 8)
        in_channels: Number of input channels (default: 1)
        embed_dim: Embedding dimension (default: 256)
        num_layers: Number of transformer layers (default: 6)
        num_heads: Number of attention heads (default: 8)
        mlp_ratio: MLP hidden dimension ratio (default: 4.0)
        dropout: Dropout rate (default: 0.1)
    """
    def __init__(self, img_size: int = 64, patch_size: int = 8,
                 in_channels: int = 1, embed_dim: int = 256,
                 num_layers: int = 6, num_heads: int = 8,
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches
        
        # Class token for aggregation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, 1, img_size, img_size)
        Returns:
            (batch_size, embed_dim)
        """
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, n_patches + 1, embed_dim)
        
        # Add positional encoding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Extract class token
        return x[:, 0]


class STViWTModel(nn.Module):
    """
    Complete ST-ViWT model for XCO₂ reconstruction.
    
    Combines Vision Transformer for spectrogram processing with
    auxiliary feature fusion for final prediction.
    
    Args:
        num_features: Number of auxiliary features (default: 31)
        img_size: Spectrogram size (default: 64)
        patch_size: Patch size for ViT (default: 8)
        num_layers: Number of transformer layers (default: 6)
        num_heads: Number of attention heads (default: 8)
        hidden_dim: Embedding dimension (default: 256)
        mlp_dim: MLP hidden dimension (default: 512)
        dropout: Dropout rate (default: 0.1)
    """
    def __init__(self, num_features: int = 31, img_size: int = 64,
                 patch_size: int = 8, num_layers: int = 6,
                 num_heads: int = 8, hidden_dim: int = 256,
                 mlp_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        
        # Vision Transformer for spectrograms
        self.vit = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=1,
            embed_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_dim / hidden_dim,
            dropout=dropout
        )
        
        # Auxiliary feature processing
        self.aux_encoder = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Fusion and prediction head
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + 64, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        
    def forward(self, spectrogram, aux_features):
        """
        Forward pass of ST-ViWT model.
        
        Args:
            spectrogram: (batch_size, 1, img_size, img_size)
            aux_features: (batch_size, num_features)
            
        Returns:
            predictions: (batch_size, 1)
        """
        # Process spectrogram through ViT
        vit_features = self.vit(spectrogram)  # (B, hidden_dim)
        
        # Process auxiliary features
        aux_encoded = self.aux_encoder(aux_features)  # (B, 64)
        
        # Fuse features
        fused = torch.cat([vit_features, aux_encoded], dim=1)  # (B, hidden_dim + 64)
        
        # Final prediction
        output = self.fusion(fused)  # (B, 1)
        
        return output
    
    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(config: dict) -> STViWTModel:
    """
    Factory function to create ST-ViWT model from configuration.
    
    Args:
        config: Dictionary containing model hyperparameters
        
    Returns:
        Initialized ST-ViWT model
    """
    model = STViWTModel(
        num_features=config.get('num_features', 31),
        img_size=config.get('img_size', 64),
        patch_size=config.get('patch_size', 8),
        num_layers=config.get('num_layers', 6),
        num_heads=config.get('num_heads', 8),
        hidden_dim=config.get('hidden_dim', 256),
        mlp_dim=config.get('mlp_dim', 512),
        dropout=config.get('dropout', 0.1)
    )
    
    return model


if __name__ == "__main__":
    # Test model
    model = STViWTModel(num_features=31)
    
    # Dummy input
    spec = torch.randn(4, 1, 64, 64)
    aux = torch.randn(4, 31)
    
    # Forward pass
    output = model(spec, aux)
    
    print(f"Model created successfully!")
    print(f"Total parameters: {model.count_parameters():,}")
    print(f"Output shape: {output.shape}")
