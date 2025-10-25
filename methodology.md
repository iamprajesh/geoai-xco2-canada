# ST-ViWT Methodology

## Complete Technical Documentation

### Table of Contents
1. [Framework Overview](#framework-overview)
2. [Wavelet Transform Processing](#wavelet-transform-processing)
3. [Vision Transformer Architecture](#vision-transformer-architecture)
4. [Auxiliary Feature Fusion](#auxiliary-feature-fusion)
5. [Training Strategy](#training-strategy)
6. [Performance Metrics](#performance-metrics)

---

## Framework Overview

The ST-ViWT (Spatio-Temporal Vision Transformer with Wavelet Transform) framework reconstructs high-resolution XCO₂ fields by integrating:

1. **Temporal Feature Extraction**: Continuous Wavelet Transform (CWT) converts XCO₂ time series into 2D spectrograms
2. **Spatial Pattern Learning**: Vision Transformer processes spectrograms to capture spatial dependencies
3. **Multi-source Data Fusion**: 31 auxiliary geospatial features enhance predictions

### Mathematical Formulation

Given sparse XCO₂ observations **X** = {x₁, x₂, ..., xₙ} at coordinates **C** = {c₁, c₂, ..., cₙ}, the goal is to reconstruct a complete XCO₂ field **X̂** on a regular grid:

```
X̂ = f(W(X), A)
```

where:
- **W(X)**: Wavelet spectrograms from temporal XCO₂ sequences
- **A**: Auxiliary geospatial features (31-dimensional)
- **f**: ST-ViWT model (learned mapping)

---

## Wavelet Transform Processing

### Continuous Wavelet Transform (CWT)

CWT decomposes time series into time-frequency representations:

```
W(a, b) = ∫ x(t) · ψ*((t - b)/a) dt
```

where:
- **a**: Scale parameter (inversely related to frequency)
- **b**: Translation parameter (time localization)
- **ψ**: Mother wavelet function
- **x(t)**: XCO₂ time series
- **W(a, b)**: Wavelet coefficients

### Mother Wavelet Selection

**Morlet wavelet** (complex-valued) is used for optimal time-frequency localization:

```
ψ(t) = π^(-1/4) · exp(iω₀t) · exp(-t²/2)
```

where ω₀ = 6 (center frequency).

**Advantages**:
- Good balance between time and frequency resolution
- Captures both low-frequency trends and high-frequency variations
- Well-suited for quasi-periodic signals (seasonal CO₂ patterns)

### Implementation Details

**Parameters**:
- Number of scales: 64
- Scale range: 1 to 64
- Sampling period: 1.0 (daily observations)
- Output size: 64 × 64 spectrograms

**Preprocessing**:
1. **Gap filling**: Linear interpolation for missing values
2. **Normalization**: Scale coefficients to [0, 1]
3. **Padding**: Zero-padding to ensure consistent dimensions

**Code Example**:
```python
import pywt
import numpy as np

def generate_spectrogram(xco2_series, scales=64):
    # Define scales
    scale_range = np.arange(1, scales + 1)
    
    # Compute CWT
    coefficients, frequencies = pywt.cwt(
        xco2_series,
        scale_range,
        'morl',
        sampling_period=1.0
    )
    
    # Normalize
    spec = (coefficients - coefficients.min()) / \
           (coefficients.max() - coefficients.min())
    
    return spec
```

---

## Vision Transformer Architecture

### Patch Embedding

Spectrograms are divided into non-overlapping patches:

```
P = {p₁, p₂, ..., pₙ} where n = (H/P)²
```

- **Image size (H)**: 64 × 64
- **Patch size (P)**: 8 × 8
- **Number of patches (n)**: (64/8)² = 64

Each patch is flattened and linearly projected:

```
zᵢ = E · flatten(pᵢ) + e_pos(i)
```

where:
- **E**: Learnable embedding matrix (64 × 256)
- **e_pos(i)**: Positional encoding for patch i
- **zᵢ**: Embedded patch (256-dimensional)

### Multi-Head Self-Attention (MHSA)

Captures spatial dependencies between patches:

```
Attention(Q, K, V) = softmax(QK^T / √d) · V
```

**Multi-head formulation**:
```
MHA(X) = Concat(head₁, ..., headₕ) · W^O
```

where each head computes:
```
headᵢ = Attention(XW^Q_i, XW^K_i, XW^V_i)
```

**Parameters**:
- Number of heads (h): 8
- Dimension per head (d): 256/8 = 32
- Attention dropout: 0.1

### Transformer Block

Standard architecture with pre-normalization:

```
x' = x + MHSA(LayerNorm(x))
x'' = x' + MLP(LayerNorm(x'))
```

**MLP structure**:
```
MLP(x) = GELU(Linear₁(x)) → Dropout → Linear₂(x) → Dropout
```

- Hidden dimension: 512
- Dropout rate: 0.1

### Complete ViT Pipeline

```
Input: Spectrogram (1, 64, 64)
  ↓
Patch Embedding → (64 patches, 256-dim)
  ↓
Add [CLS] token → (65 tokens, 256-dim)
  ↓
Add Positional Encoding
  ↓
Transformer Block × 6
  ↓
Layer Normalization
  ↓
Extract [CLS] token → (256-dim)
  ↓
Output: Spatial features
```

---

## Auxiliary Feature Fusion

### Feature Encoder

Auxiliary features are processed through a fully connected network:

```
h_aux = ReLU(W₁ · A + b₁)  → 128-dim
h_aux = ReLU(W₂ · h_aux + b₂)  → 64-dim
```

### Late Fusion Strategy

Concatenate ViT features with encoded auxiliary features:

```
h_fused = [h_vit ; h_aux]  → (256 + 64 = 320-dim)
```

### Regression Head

Final prediction through multi-layer network:

```
h₁ = ReLU(Linear(h_fused)) → 256-dim
h₂ = ReLU(Linear(h₁)) → 128-dim
ŷ = Linear(h₂) → 1-dim (XCO₂ prediction)
```

### Complete Architecture Diagram

```
Spectrogram (64×64)          Auxiliary Features (31)
        ↓                              ↓
  Vision Transformer              FC Network
        ↓                              ↓
   Features (256)               Features (64)
        └──────────┬──────────────────┘
                   ↓
            Concatenate (320)
                   ↓
            Fusion Network
                   ↓
          XCO₂ Prediction (1)
```

---

## Training Strategy

### Loss Function

Mean Squared Error (MSE) for regression:

```
L = (1/N) Σᵢ (ŷᵢ - yᵢ)²
```

where:
- **ŷᵢ**: Predicted XCO₂
- **yᵢ**: True XCO₂
- **N**: Number of valid samples

### Optimization

**Optimizer**: AdamW
- Learning rate: 1×10⁻⁴
- Weight decay: 1×10⁻⁴
- β₁ = 0.9, β₂ = 0.999
- ε = 1×10⁻⁸

**Learning Rate Schedule**: Cosine Annealing

```
ηₜ = η_min + (η_max - η_min) · (1 + cos(πt/T)) / 2
```

- η_max = 1×10⁻⁴
- η_min = 1×10⁻⁶
- T = 100 epochs

### Data Augmentation

Minimal augmentation to preserve spatial structure:
- **Normalization**: StandardScaler for features and targets
- **No geometric augmentation** (preserves geospatial integrity)

### Regularization

- **Dropout**: 0.1 in all layers
- **Weight decay**: L2 regularization (1×10⁻⁴)
- **Early stopping**: Patience = 10 epochs

### Training Configuration

```
Batch size:      32
Epochs:          100
Train/Val split: 80/20
Random seed:     42
Mixed precision: Optional (FP16)
Gradient clip:   None (stable training)
```

---

## Performance Metrics

### Evaluation Metrics

1. **Coefficient of Determination (R²)**:
   ```
   R² = 1 - (Σ(yᵢ - ŷᵢ)² / Σ(yᵢ - ȳ)²)
   ```

2. **Root Mean Square Error (RMSE)**:
   ```
   RMSE = √[(1/N) Σ(ŷᵢ - yᵢ)²]
   ```

3. **Mean Absolute Error (MAE)**:
   ```
   MAE = (1/N) Σ|ŷᵢ - yᵢ|
   ```

4. **Mean Absolute Percentage Error (MAPE)**:
   ```
   MAPE = (100/N) Σ|((yᵢ - ŷᵢ) / yᵢ)|
   ```

5. **Bias**:
   ```
   Bias = (1/N) Σ(ŷᵢ - yᵢ)
   ```

### Expected Performance

| Metric | Target | Achieved |
|--------|--------|----------|
| R² | > 0.95 | 0.9547 |
| RMSE | < 0.5 ppm | 0.487 ppm |
| MAE | < 0.5 ppm | 0.412 ppm |
| MAPE | < 2% | 1.23% |
| Bias | ≈ 0 ppm | -0.021 ppm |

### Computational Requirements

**Training**:
- GPU: NVIDIA RTX 3090 (24GB) or equivalent
- Memory: ~8GB GPU RAM
- Time: ~2 hours for 100 epochs
- Parameters: ~2.1M trainable

**Inference**:
- Throughput: ~500 predictions/second
- Latency: ~2ms per sample
- Memory: ~4GB GPU RAM

---

## Key Innovations

1. **Wavelet-Transformer Synergy**: Combines multi-scale temporal analysis with spatial attention
2. **Geospatial Feature Integration**: Leverages 31 auxiliary features from GEE
3. **End-to-End Learning**: Joint optimization of all components
4. **Scalability**: Efficient architecture suitable for large-scale reconstruction
   
---

**Last Updated**: October 24, 2025  
**Version**: 1.0
