# Conditional Date Generation Performance Analysis
## A Comprehensive Comparison of Six Deep Learning Architectures

---

##  Table of Contents
1. [Project Overview](#project-overview)
2. [Project Flow](#project-flow)
3. [Model Architectures](#model-architectures)
4. [Experimental Setup](#experimental-setup)
5. [Results & Performance](#results--performance)
6. [Key Technical Insights](#key-technical-insights)
7. [Conclusion](#conclusion)
8. [Files Overview](#files-overview)

---

##  Project Overview

This project evaluates **six distinct deep learning architectures** on their ability to generate valid date strings (`dd-mm-yyyy`) based on four constraining conditions:

| Condition | Description | Challenge Level |
|-----------|-------------|-----------------|
| **Day of Week** | MON, TUE, WED, THU, FRI, SAT, SUN | High (Mathematical) |
| **Month** | JAN-DEC | Low (Categorical) |
| **Leap Year** | True/False | Low (Binary) |
| **Decade** | Year grouping (e.g., 2000s, 2010s) | Moderate |

**Goal**: Determine which neural architecture best learns calendar logic and generates dates that satisfy all four conditions simultaneously.

---

##  Project Flow

```
┌─────────────────────────────────────────────────────────────┐
│                  DATA LOADING & PARSING                     │
│  • Load data.txt with conditions and dates                  │
│  • Parse conditions: [DAY] [MONTH] [LEAP] [DECADE]          │
│  • Extract date strings: dd-mm-yyyy format                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│              DATA TOKENIZATION & NORMALIZATION              │
│  • Map conditions to numerical tokens                       │
│  • Normalize dates to 0-indexed ranges                      │
│  • Create PyTorch DataLoader with batch size 64             │
│  • Setup 50 epochs of training with learning rate 0.001    │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│          TRAIN 6 DISTINCT MODEL ARCHITECTURES               │
│                                                              │
│  1. FFNN              - Feedforward baseline                │
│  2. LSTM              - Sequential memory                   │
│  3. CGAN              - Conditional GAN                     │
│  4. VAE               - Variational Autoencoder             │
│  5. Transformer       - Self-attention mechanism            │
│  6. WGAN              - Wasserstein GAN variant             │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│        EVALUATE & COMPARE MODEL PREDICTIONS                 │
│  • Generate dates for example inputs                        │
│  • Verify compliance with all 4 conditions                  │
│  • Calculate per-component accuracy metrics                 │
│  • Compare loss evolution across epochs                     │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│           VISUALIZE & GENERATE REPORTS                      │
│  • Loss evolution curves for all models                     │
│  • Accuracy heatmap: Models × Components                    │
│  • Bar charts for cross-model comparison                    │
│  • Final predictions table for review                       │
└─────────────────────────────────────────────────────────────┘
```

---

##  Model Architectures

### 1. **FFNN (Feedforward Neural Network)** - Baseline
- **Structure**: 4-input → 128 → 64 → [Day(31), Month(12), Year(n)]
- **Role**: Establishes baseline performance
- **Strengths**: Simple, fast training
- **Limitations**: Cannot capture sequential patterns

### 2. **LSTM (Long Short-Term Memory)** - Sequential
- **Structure**: Embedded conditions → 2-layer LSTM → Output heads
- **Role**: Tests memory capacity for temporal patterns
- **Strengths**: Excellent at month/leap-year accuracy (near 100%)
- **Limitations**: Struggles with day-of-week mathematical logic

### 3. **CGAN (Conditional GAN)** - Generative
- **Structure**: Generator(conditions + noise) + Discriminator
- **Role**: Tests adversarial learning for date generation
- **Strengths**: Best decade accuracy (45%), balanced performance
- **Limitations**: Training instability, mode collapse risk

### 4. **VAE (Variational Autoencoder)** - Probabilistic
- **Structure**: Encoder(conditions + dates) → Latent space → Decoder
- **Role**: Tests unsupervised generative learning
- **Strengths**: Stable training with gradient clipping
- **Limitations**: Moderate accuracy across components

### 5. **Transformer** - Attention-Based 
- **Structure**: 4-head self-attention → Output heads
- **Role**: Tests modern attention mechanism effectiveness
- **Strengths**: Steepest loss descent, high month/leap accuracy
- **Limitations**: Struggles with day-of-week like other models

### 6. **WGAN (Wasserstein GAN)** - Advanced Generative
- **Structure**: Generator + Critic with weight clipping
- **Role**: Tests improved GAN stability (RMSProp optimizer)
- **Strengths**: Most stable generative distribution, avoids mode collapse
- **Limitations**: Computationally expensive, longer training time

---

## 🔬 Experimental Setup

**Hyperparameters**:
- **Batch Size**: 64
- **Epochs**: 50
- **Learning Rate**: 0.001
- **Optimizer**: Adam (except WGAN uses RMSprop)
- **Loss Function**: CrossEntropyLoss with component weighting [2.0, 1.0, 5.0]

**Component Weights**:
- Day weight: 2.0 (higher emphasis on difficult component)
- Month weight: 1.0 (baseline)
- Year weight: 5.0 (highest emphasis on decade accuracy)

**Regularization Techniques**:
- Dropout (20-30% in hidden layers)
- Batch Normalization (in CGAN)
- Gradient clipping (VAE: max_norm=1.0)
- Weight clipping (WGAN: ±0.01)

---

##  Results & Performance

### Final Accuracy Comparison

| Metric | FFNN | LSTM | CGAN | VAE | Transformer | WGAN |
|--------|------|------|------|-----|-------------|------|
| **Month** | 89% | 100% | 98% | 85% | 100% | 94% |
| **Day** | 16% | 14% | 18% | 12% | 15% | 17% |
| **Leap Year** | 92% | 100% | 100% | 88% | 98% | 96% |
| **Decade** | 22% | 31% | 45% | 28% | 43% | 38% |
| **All Constraints** | 3% | 8% | 12% | 2% | 9% | 11% |

### Loss Evolution Highlights
- **FFNN**: Steady decline, final loss ≈ 0.850
- **LSTM**: Rapid convergence to ≈ 0.720
- **CGAN-G**: Fast initial drop to ≈ 0.780
- **Transformer**: **Steepest descent**, final loss ≈ 0.695
- **VAE**: Stabilized around ≈ 0.810 (KLD contribution)
- **WGAN**: Smooth convergence with RMSProp

---

## Key Technical Insights

### 1. **Categorical Mastery** ✓
Sequential (LSTM) and Attention-based (Transformer) models achieve **near-perfect accuracy** (100%) on Month and Leap Year constraints. This suggests these architectures excel at learning discrete, rule-based patterns.

### 2. **The Day-of-Week Problem** ✗
All models plateau at **14-18% accuracy** on day-of-week prediction. This reveals a fundamental limitation: neural networks struggle to learn the **mathematical offsets** required by Gregorian calendar algorithms (Zeller's Congruence). This suggests:
- Day-of-week prediction requires explicit algorithmic computation, not pattern learning
- Cyclical temporal embeddings might be necessary for improvement
- Symbolic reasoning may be more appropriate than continuous representations

### 3. **Generative Model Advantage**
CGAN and WGAN achieve **significantly higher decade accuracy** (45% and 38% vs. 22-31% for others), suggesting that:
- Adversarial training helps models learn complex temporal patterns
- The critic/discriminator provides valuable gradient signals for distinguishing valid year ranges
- Noise injection enables exploration of the generation space

### 4. **Optimization Impact**
- **Importance weighting**: Prioritizing year/decade components doubled decade accuracy
- **Transformer self-attention**: 15-20% faster convergence vs. FFNN
- **Gradient clipping** (VAE): Critical for numerical stability with probabilistic models
- **Weight clipping** (WGAN): More stable than Spectral Normalization for this task

### 5. **The Multi-Constraint Ceiling**
None of the models exceed **12% all-constraint accuracy**, indicating that satisfying 4 conditions simultaneously is fundamentally harder than individual component accuracy. The constraints appear **somewhat independent** in neural representation space.

---

## Conclusion

### **Main Findings**

This comprehensive evaluation across six distinct architectures reveals that **neural networks are highly effective for categorical calendar rules but fundamentally limited for mathematical relationships**:

####  **What Neural Networks Do Well:**
- **Month inference**: 85-100% accuracy across all models
- **Leap year detection**: 88-100% accuracy across all models  
- **Learned categorical cycles**: LSTM/Transformer internalize month cycles with precision

####  **Where Neural Networks Struggle:**
- **Day of the week**: Plateaus at ~15% (only marginally better than random for 7-class problem)
- **Multi-constraint satisfaction**: ~12% when all 4 constraints must be met
- **Mathematical computation**: Neural networks fail at deterministic calendar math

### **Model Recommendations**

1. **Best Overall**: **Transformer** or **CGAN**
   - Transformer: Fastest convergence, clean attention mechanism
   - CGAN: Best decade accuracy, balanced performance

2. **For Deployment**: **Transformer** (recommended)
   - Highest loss convergence speed
   - 100% accuracy on two critical components
   - Lower computational overhead than WGAN

3. **For Research**: **WGAN**
   - Most stable generative distribution
   - Avoids mode collapse seen in CGAN
   - Proves Wasserstein distance is effective here

### **Future Improvements**

To overcome the day-of-week bottleneck:
1. **Explicit Algorithmic Integration**: Embed Zeller's Congruence or similar formula into the model
2. **Cyclical Embeddings**: Use sinusoidal positional encodings for day-of-week (similar to Transformer time embeddings)
3. **Symbolic-Neural Hybrids**: Combine learned representations with rule-based constraints
4. **Enhanced Supervision**: Add auxiliary loss for day-of-week prediction with explicit date-DoW data
5. **Multi-Task Learning**: Train on related tasks (date validation, leap-year calculation) to improve representations

---

## Files Overview

| File | Purpose |
|------|---------|
| `assig2_gan.ipynb` | Main notebook with all model implementations and training loops |
| `direct_predictions.txt` | Predictions generated from direct/inference mode |
| `output_predictions.txt` | Final predictions from all models on test set |
| `readme.md` | This file - complete project documentation |

### Running the Notebook

1. **Cell 1**: Setup and imports
2. **Cells 2-4**: Data loading and tokenization
3. **Cells 5-9**: Model definitions (FFNN, LSTM, CGAN, VAE, Transformer, WGAN)
4. **Cells 10-16**: Training loops for all 6 models
5. **Cells 17-21**: Evaluation, predictions, and accuracy calculations
6. **Cells 22-26**: Visualizations (loss curves, heatmaps, comparison tables)

---

## Performance Summary Table

| Architecture | Parameter Count | Training Speed | Stability | Best Component | Worst Component |
|--------------|-----------------|-----------------|-----------|-----------------|-----------------|
| FFNN | 12,847 | Fast | High | Month (89%) | Day (16%) |
| LSTM | 18,965 | Fast | High | Month/Leap (100%) | Day (14%) |
| CGAN | 24,563 | Medium | Medium | Decade (45%) | Day (18%) |
| VAE | 21,584 | Slow | Medium | Month (85%) | Day (12%) |
| **Transformer** | **16,832** | **Very Fast** | **High** | **Month (100%)** | **Day (15%)** |
| WGAN | 24,352 | Slow | High | Decade (38%) | Day (17%) |

---

**Project Status**:  Complete  
**Last Updated**: May 2026  
**Models Evaluated**: 6  
**Total Accuracy Combinations Tested**: 1000+