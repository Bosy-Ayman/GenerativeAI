
## PAPER 1: VQ-VAE

### Equation 1: Vector Quantization (Discrete Lookup)

```
q(z = k|x) = 1 for k = argmin_j ||z_e(x) - e_j||²
             0 otherwise
```

**What it means:**

- This equation defines a **discrete posterior distribution** over the codebook
- For any input x:
    - Find the closest codebook entry: k = argmin_j ||z_e(x) - e_j||²
    - Set probability of that code to 1: q(z = k|x) = 1
    - Set all others to 0: q(z ≠ k|x) = 0
- This is a **one-hot distribution** (probability 1 for one outcome, 0 elsewhere)

**Breaking it down:**

```
z_e(x) = encoder output (continuous vector in R^D)
e_j = j-th codebook entry (learned, in R^D)

||z_e(x) - e_j||² = L2 distance from encoder output to code j
                   = sum of squared differences

argmin_j = find which j minimizes this distance
         = find closest codebook entry
         
k = index of closest entry
```

**Example with numbers:**

```
Encoder outputs: z_e(x) = [0.5, 0.3, 0.8]
Codebook has 4 entries:
  e_1 = [0.1, 0.2, 0.3]
  e_2 = [0.4, 0.2, 0.7]
  e_3 = [0.9, 0.8, 0.9]
  e_4 = [0.2, 0.1, 0.2]

Calculate distances:
  ||[0.5,0.3,0.8] - [0.1,0.2,0.3]||² = (0.4)² + (0.1)² + (0.5)² = 0.16 + 0.01 + 0.25 = 0.42
  ||[0.5,0.3,0.8] - [0.4,0.2,0.7]||² = (0.1)² + (0.1)² + (0.1)² = 0.03 ← closest!
  ||[0.5,0.3,0.8] - [0.9,0.8,0.9]||² = (0.4)² + (0.5)² + (0.1)² = 0.42
  ||[0.5,0.3,0.8] - [0.2,0.1,0.2]||² = (0.3)² + (0.2)² + (0.6)² = 0.49

argmin = 2 (k=2 is closest)

Result:
  q(z = 1|x) = 0
  q(z = 2|x) = 1  ← selected!
  q(z = 3|x) = 0
  q(z = 4|x) = 0
```

**Why this matters:**

```
Discrete assignment (not sampling):
- Deterministic (always choose same code for same input)
- One-hot (no "soft" assignment, hard discrete choice)
- No randomness (needed for gradients)

This prevents posterior collapse:
- Can't ignore latents (must choose one)
- Encoder must output something that quantizes well
- Decoder must learn from discrete codes
```

**Connection to standard VAEs:**

```
Standard VAE posterior: q(z|x) = N(μ, σ²)
  - Continuous Gaussian distribution
  - Can have high variance (collapse to prior)
  - Reparameterization trick for gradients

VQ-VAE posterior: q(z = k|x) = 1-hot vector
  - Discrete categorical distribution
  - No variance (always 1 or 0)
  - Can't collapse (must pick something)
  - Straight-through estimator for gradients
```

---

### Equation 2: Quantized Code Mapping

```
z_q(x) = e_k, where k = argmin_j ||z_e(x) - e_j||²
```

**What it means:**

- After finding the closest code k (from Equation 1)
- Replace encoder output with actual codebook entry
- z_q(x) is the **decoder input** (the actual code vector, not index)

**Visual representation:**

```
Encoder output:          Codebook:              Selected code:
z_e(x) = [0.4]         e_1 = [0.1]            z_q(x) = [0.4]
         [0.2]         e_2 = [0.4]                    [0.2]
         [0.7]         e_3 = [0.9]   ← pick e_2   [0.7]

Step 1: Calculate distances to all entries
Step 2: Find closest (k=2)
Step 3: Replace z_e(x) with e_2
Step 4: Pass z_q(x) to decoder
```

**Key difference from Equation 1:**

```
Equation 1: q(z = k|x) = one-hot distribution (probability)
            - Used for computing KL divergence
            - Theoretical quantity

Equation 2: z_q(x) = actual codebook vector
            - Used for reconstruction
            - Practical quantity fed to decoder
```

**Example showing the difference:**

```
Same setup as before:
Encoder output: z_e(x) = [0.5, 0.3, 0.8]
Codebook: e_1, e_2, e_3, e_4
Closest: k = 2

Equation 1 gives: q(z|x) = [0, 1, 0, 0] (probability distribution)
Equation 2 gives: z_q(x) = e_2 = [0.4, 0.2, 0.7] (actual vector!)
```

---

### Equation 3: Complete VQ-VAE Loss Function

```
L = log p(x|z_q(x)) + ||sg[z_e(x)] - e_k||² + β||z_e(x) - sg[e_k]||²
    ├─ Term 1          ├─ Term 2              ├─ Term 3
    │ Reconstruction   │ Codebook loss        │ Commitment loss
    │ (both optimize)  │ (only e optimizes)   │ (only z_e optimizes)
    └─────────────────┴──────────────────────┴─────────────────
```

**This is THE MOST IMPORTANT equation in the paper**

Let me explain each term completely:

---

#### **TERM 1: Reconstruction Loss**

```
log p(x|z_q(x))
```

**What it is:**

- Log-likelihood of image x given quantized codes z_q(x)
- Modeled as Gaussian: p(x|z_q) = N(x; Decoder(z_q), σ²I)
- Negative log-likelihood becomes MSE

**In practice:**

```
L_rec = ||x - Decoder(z_q(x))||²

Why MSE?
If p(x|z_q) = (1/(2πσ²)^(D/2)) exp(-||x - μ||²/(2σ²))

Then: -log p(x|z_q) ∝ ||x - μ||²

So we minimize -log p by minimizing MSE
```

**Who optimizes this term:**

- **Encoder**: Gets gradients through z_q and backward to z_e
- **Decoder**: Gets direct gradients from reconstruction error
- **NOT codebook**: Uses stop-gradient (sg) on codebook side

**Gradient flow for Term 1:**

```
Input x
  ↓
Encoder → z_e(x)
  ↓
[Quantize]
  ↓
z_q(x) → Decoder → x̂
  ↓
Loss: ||x - x̂||²
  ↓
∇L_rec propagates:
  - To decoder: normal backprop
  - To z_e: through z_q via straight-through estimator
  - Codebook: BLOCKED by stop-gradient
```

**Interpretation:**

- Pushes encoder to output codes that quantize well
- Pushes decoder to reconstruct from codes well
- Incentivizes meaningful code usage

---

#### **TERM 2: Vector Quantization (VQ) Loss**

```
||sg[z_e(x)] - e_k||²
```

**Breaking it down:**

```
sg[z_e(x)] = stop-gradient on encoder output
             = treat z_e(x) as constant (no gradients)
             = [0.4, 0.2, 0.7] (numerical value, not as variable)

e_k = selected codebook entry (this CAN be updated)
    = [0.4, 0.2, 0.7]

||sg[z_e(x)] - e_k||² = squared L2 distance
                       = sum of squared differences
```

**Example calculation:**

```
Encoder output: z_e(x) = [0.50, 0.30, 0.80]
Selected code:  e_k = [0.45, 0.28, 0.78]

Difference: [0.05, 0.02, 0.02]
Squared:    [0.0025, 0.0004, 0.0004]
Sum:        0.0033

This IS L_vq for this example
```

**Stop-gradient (sg) explained:**

```
Normal gradient flow:
  ∂L/∂e_k ← ∂L/∂z_e ← ...
  (gradients flow both directions)

With stop-gradient:
  ∂L/∂e_k ← ∂L/∂z_e is BLOCKED
  
  Specifically:
  sg[x]: During forward pass = x (identity)
         During backward pass = 0 gradient (blocks flow)

Why block encoder gradients?
- We want VQ loss to ONLY update codebook
- Encoder already gets gradients from reconstruction loss
- Codebook learning via VQ loss (dictionary learning)
```

**Who optimizes Term 2:**

- **ONLY codebook e_k**: Gets gradient ∂L_vq/∂e_k = 2(e_k - sg[z_e])
- **NOT encoder**: sg blocks gradient

**Gradient for codebook:**

```
L_vq = ||sg[z_e(x)] - e_k||²

∂L_vq/∂e_k = 2(e_k - sg[z_e(x)])
           = 2(e_k - z_e(x))  [since sg[z_e] is constant]

Update: e_k ← e_k - η·2(e_k - z_e(x))
            = e_k - 2η·e_k + 2η·z_e(x)
            = (1 - 2η)e_k + 2η·z_e(x)
            ≈ e_k + Δ·(z_e(x) - e_k)  [for small learning rate]

Effect: Codebook moves TOWARD encoder output!
This is exactly what we want for clustering.
```

**Interpretation (dictionary learning):**

```
This is K-means clustering!

Standard K-means update:
  c_k ← mean({x_i : assigned to cluster k})

VQ loss effect:
  e_k moves toward {z_e(x) : closest to e_k}
  
Over time:
  Codebook entries become cluster centers
  Encoder outputs form natural clusters
  Same code used for similar inputs
```

---

#### **TERM 3: Commitment Loss**

```
β||z_e(x) - sg[e_k]||²
```

**Breaking it down:**

```
β = scalar coefficient (typically 0.25)
    Controls how much encoder "commits" to selected code

z_e(x) = encoder output (CAN be updated)
sg[e_k] = selected codebook entry (treated as constant)
||...||² = L2 distance
```

**Stop-gradient on different variable!**

```
TERM 2: sg[z_e(x)] - e_k
        ↑ stop-gradient here
        Codebook learns to follow encoder

TERM 3: z_e(x) - sg[e_k]
                 ↑ stop-gradient here
                 Encoder learns to stay with codebook
```

**Example:**

```
Encoder output: z_e(x) = [0.50, 0.30, 0.80]
Selected code:  e_k = [0.45, 0.28, 0.78]

Difference: [0.05, 0.02, 0.02]
Squared:    [0.0025, 0.0004, 0.0004]
Sum:        0.0033

L_commit = β × 0.0033
         = 0.25 × 0.0033 ≈ 0.0008 (scaled by β!)
```

**Who optimizes Term 3:**

- **ONLY encoder z_e**: Gets gradient ∂L_commit/∂z_e = 2β(z_e - sg[e_k])
- **NOT codebook**: sg blocks gradient

**Gradient for encoder (from commitment loss):**

```
L_commit = β||z_e(x) - sg[e_k]||²

∂L_commit/∂z_e = 2β(z_e(x) - sg[e_k])
               = 2β(z_e(x) - e_k)  [sg is constant]

Update: z_e ← z_e - η·2β(z_e - e_k)
            = (1 - 2ηβ)z_e + 2ηβ·e_k

Effect: Encoder output moves TOWARD selected code!
```

**Why is this needed? Intuitive explanation:**

```
Without commitment loss:

Time step 1:
  Encoder output z_e = [0.5, 0.3, 0.8]
  Closest code: e_2 = [0.4, 0.2, 0.7]
  Reconstruction loss: decoder learns to reconstruct from e_2
  VQ loss: e_2 moves toward z_e

Time step 2:
  Encoder output changes to z_e = [0.2, 0.9, 0.3]
  Closest code now: e_7 = [0.2, 0.8, 0.2]
  Reconstruction loss: decoder sees new code e_7
  VQ loss: e_7 moves toward new z_e
  
  Problem: Decoder confused! One moment learning from e_2, next from e_7
           Encoder constantly switching codes → unstable training!

With commitment loss:

Encoder "penalized" for moving away from selected code
Encourages: Stay with current code while decoder learns
Result: Stable training, smooth code assignments
        Decoder doesn't get confused by jumping codes
```

**Concrete example of instability without commitment:**

```
Training step 1:
  z_e(x₁) close to e_2 → select e_2
  Decoder trains: "learn from e_2"

Training step 2:
  z_e(x₂) close to e_2 initially, but far after gradient update
  Now closer to e_5 → select e_5
  Decoder trains: "learn from e_5"
  
Both x₁ and x₂ similar, but decoder gets different codes!
Decoder can't generalize
Training becomes noisy and slow

Commitment loss prevents this by saying:
"If you pick e_2 now, you pay a cost if you leave it next step"
```

**Why β = 0.25 specifically?**

```
β too small (β ≈ 0):
  - Commitment loss has minimal effect
  - Encoder drifts away from codes
  - Training unstable
  
β too large (β ≈ 10):
  - Encoder heavily penalized for moving
  - Gets stuck near initial codes
  - Can't explore codebook
  - Training frozen
  
β = 0.25 (empirically good):
  - Balances encoder freedom with stability
  - Paper: "robust to β ∈ [0.1, 2.0]"
  - Depends on reconstruction loss scale
  - Larger reconstruction loss → larger β needed
```

---

### Equation 4: Codebook Update with EMA (Appendix A.1)

```
N_i^(t) := N_i^(t-1) * γ + n_i^(t) * (1 - γ)
m_i^(t) := m_i^(t-1) * γ + Σ_j z_e(x)_j * (1 - γ)
e_i^(t) := m_i^(t) / N_i^(t)
```

**What it is:**

- **Alternative to VQ loss** for updating codebook
- Uses exponential moving average (EMA) instead of gradient descent
- More stable, no gradient computation needed

**Breaking it down:**

```
N_i^(t) = count of encodings assigned to code i at time t

E_i^(t-1) * γ = "remember" 99% of old count (γ=0.99)
n_i^(t) * (1-γ) = "add" 1% of new assignments this batch
Result: Smooth update of count estimate

m_i^(t) = sum of encoder outputs assigned to code i

m_i^(t-1) * γ = "remember" 99% of old sum
Σ_j z_e(x)_j * (1-γ) = "add" 1% of new sums this batch
Result: Smooth update of sum estimate

e_i^(t) = updated codebook entry

e_i^(t) = m_i^(t) / N_i^(t)
        = (sum of assigned codes) / (count of assignments)
        = cluster center! (K-means!)
```

**Concrete example:**

```
Suppose code e_1 has been trained:
  N_1^(999) = 150 (seen 150 times)
  m_1^(999) = [75, 45, 120] (sum of all codes assigned to e_1)
  e_1^(999) = [75/150, 45/150, 120/150] = [0.5, 0.3, 0.8]

New batch arrives:
  Codes assigned to e_1 this batch: 10 encoders
  Sum of these 10 codes: [4, 2, 5]

Update with γ=0.99:
  N_1^(1000) = 150 * 0.99 + 10 * 0.01 = 148.5 + 0.1 = 148.6
  m_1^(1000) = [75, 45, 120] * 0.99 + [4, 2, 5] * 0.01
             = [74.25, 44.55, 118.8] + [0.04, 0.02, 0.05]
             = [74.29, 44.57, 118.85]
  e_1^(1000) = [74.29/148.6, 44.57/148.6, 118.85/148.6]
             = [0.5001, 0.3002, 0.7998]

Result: Smooth update! e_1 barely changed (as expected)
```

**Comparison: VQ loss vs EMA**

```
VQ LOSS APPROACH:
  L_vq = ||sg[z_e] - e_k||²
  Gradient: ∂L/∂e_k = 2(e_k - z_e)
  Update: e_k ← e_k - η·∇L
  Needs: Gradient computation, learning rate tuning
  
EMA APPROACH:
  N_i ← γ·N_i + (1-γ)·count
  m_i ← γ·m_i + (1-γ)·Σz_e
  e_i = m_i / N_i
  Needs: γ (decay factor, usually 0.99)
  Advantage: Direct, no gradient variance
  Advantage: Automatically prevents unused codes
```

**Why EMA prevents unused codes:**

```
If code e_k is never assigned:
  n_k^(t) = 0 for all t
  
  N_k^(t) = N_k^(t-1) * γ (decays to 0)
  m_k^(t) = m_k^(t-1) * γ (decays to 0)
  e_k = m_k / N_k → undefined or stays fixed

In VQ loss:
  If code never used, never gets gradient
  Stays at initialization (dead code)

With EMA:
  Code "resets" naturally as count decays
  Makes room for new codes
  More balanced codebook utilization
```

---

## PAPER 2: PIXELCNN

### Equation 1: Autoregressive Factorization

```
p(x) = ∏_{i=1}^{n²} p(x_i | x_1, ..., x_{i-1})
```

**What it means:**

- Factorize joint probability into product of conditionals
- Exact (no approximation) due to chain rule of probability
- Each pixel depends on all previous pixels in raster order

**Derivation from chain rule:**

```
Start with joint: p(x_1, x_2, ..., x_n)

Apply chain rule:
p(A, B, C) = p(A) · p(B|A) · p(C|A,B)

For n pixels:
p(x_1, ..., x_n) = p(x_1) 
                 · p(x_2|x_1) 
                 · p(x_3|x_1, x_2)
                 · ...
                 · p(x_n|x_1, ..., x_{n-1})
                 
                 = ∏_i p(x_i | x_{<i})
```

**Why this is powerful:**

```
Standard approach (naive):
- One neural network outputs distribution over all pixels
- Exponential in number of variables
- Intractable for high-dimensional x

Autoregressive approach:
- One network per pixel (or same network at each step)
- Each outputs distribution for x_i given previous
- Product is still exact!
- Tractable (sum of logs = log-likelihood)
```

**For images specifically:**

```
Raster scan order (standard):
[0,0] → [0,1] → [0,2] → ...
[1,0] → [1,1] → ...

For pixel at position (i,j):
  Can see: all pixels with position < (i,j) in raster order
           = all above this row + all to left in this row
           
  Cannot see: all to the right + all below
              (these are future in generation order)

Example for 3×3 image:
Position (0,0): depends on nothing (first)
Position (0,1): depends on (0,0)
Position (0,2): depends on (0,0), (0,1)
Position (1,0): depends on (0,0), (0,1), (0,2)
Position (1,1): depends on (0,0), (0,1), (0,2), (1,0)
...
```

---

### Equation 2: Gated Activation Unit

```
y = tanh(W_{k,f} * x) ⊙ σ(W_{k,g} * x)
```

**Breaking it down term by term:**

```
* = convolution operator
  W_{k,f} * x means: convolve input x with filter weights W_{k,f}
  Result: same spatial dimensions as x, new feature channels

W_{k,f} = filter weights (learned parameters)
          one set of weights for the "feature" gate
          
tanh(...) = activation function
            Range: [-1, 1]
            Non-linear, allows signal shaping
            
σ = sigmoid function (written as σ)
  σ(z) = 1 / (1 + e^{-z})
  Range: [0, 1]
  Interpreted as: gate strength (0=off, 1=on)

⊙ = element-wise (Hadamard) product
    element at position [i,j,c] of left multiplies
    element at position [i,j,c] of right
```

**Concrete example with dimensions:**

```
Input x: shape [H, W, C_in] = [32, 32, 128]
         = 32×32 spatial grid, 128 feature channels

W_{k,f}: shape [3, 3, 128, 256]
         = 3×3 filter, 128 input channels, 256 output channels
         
W_{k,f} * x: shape [32, 32, 256]
             = same spatial size, 256 output channels
             
tanh(W_{k,f} * x): shape [32, 32, 256]
                   = apply tanh element-wise

W_{k,g}: shape [3, 3, 128, 256]
         = same structure as W_{k,f}
         
σ(W_{k,g} * x): shape [32, 32, 256]
                = sigmoid applied element-wise
                = values in [0, 1]
                
y = tanh(...) ⊙ σ(...): shape [32, 32, 256]
                        = element-wise multiply
                        = feature_vector × gate_strength
```

**Numerical example (tiny):**

```
Suppose at one spatial location [0,0]:

Feature vector after tanh: f = [0.5, -0.3, 0.9]
Gate vector after sigmoid: g = [0.8, 0.2, 0.95]

Element-wise product:
y[0,0] = [0.5×0.8, -0.3×0.2, 0.9×0.95]
       = [0.4, -0.06, 0.855]

Interpretation:
- Channel 0: feature 0.5, gate 0.8 → 40% passes
- Channel 1: feature -0.3, gate 0.2 → only 6% passes (low gate)
- Channel 2: feature 0.9, gate 0.95 → 85.5% passes (high gate)

Network learns which channels to "activate" via gates!
```

**Why this improves performance:**

```
Standard ReLU:
  y = max(0, W * x)
  Each channel either passes or gets zeroed
  All or nothing decision

Gated unit:
  y = tanh(W_f * x) ⊙ σ(W_g * x)
  Continuous gates [0,1]
  Flexible: can pass 10%, 50%, 90%, etc.
  Two separate weights to tune (f and g)
  More expressive (multiplicative interactions)
  
Analogy to LSTM:
  LSTM has input gate: i_t = σ(W_i·[h_{t-1}, x_t])
  Multiplication: signal × gate is used throughout LSTM
  Proven effective for capturing long-range dependencies
  
Gated CNN similarly gains:
  - Multiplicative interactions
  - Flexible gating
  - Better expressivity than simple ReLU
```

---

### Equation 3: Conditional PixelCNN (Non-spatial)

```
y = tanh(W_{k,f} * x + V_{k,f}^T h) ⊙ σ(W_{k,g} * x + V_{k,g}^T h)
```

**Modification from Equation 2:**

```
Original:  y = tanh(W * x) ⊙ σ(W * x)
Modified:  y = tanh(W * x + V^T h) ⊙ σ(W * x + V^T h)
                           ↑ ADD THIS ↑
```

**Breaking down the new terms:**

```
h = conditioning information (non-spatial)
    Example: one-hot class vector [0,0,1,0,...,0] for class 2
    Shape: [D_cond] = [1000] for 1000 classes
    
V_{k,f} = weight matrix for feature gate conditioning
          Shape: [D_cond, C_out] = [1000, 256]
          Learned during training
          
V_{k,f}^T = transpose
            Shape: [C_out, D_cond] = [256, 1000]
            
V_{k,f}^T h = matrix-vector multiplication
              Shape: [C_out] = [256]
              Result: adds same bias to every spatial location!
              
W_{k,f} * x = convolution (spatial-dependent)
              Shape: [H, W, C_out] = [32, 32, 256]
              
W_{k,f} * x + V_{k,f}^T h = broadcasting!
                            Each spatial location adds same bias h
                            Like adding location-invariant offset
```

**Example with numbers:**

```
Spatial input: x ∈ R^{32×32×128}
Conditioning: h ∈ R^{1000}  (one-hot for class 2)
              h = [0, 1, 0, ..., 0]

Convolution: W * x ∈ R^{32×32×256}
After conv at location (0,0): [0.5, -0.2, 0.7, ...]

Matrix-vector: V^T h ∈ R^{256}
Here: V^T picks out 2nd row of V (since h selects class 2)
V^T h = [0.3, 0.1, -0.1, ...]

Addition (broadcasting):
At location (0,0): [0.5, -0.2, 0.7] + [0.3, 0.1, -0.1]
                 = [0.8, -0.1, 0.6]

At location (0,1): [1.2, 0.4, 0.3] + [0.3, 0.1, -0.1]
                 = [1.5, 0.5, 0.2]

All locations add SAME offset [0.3, 0.1, -0.1]!
This is "class-dependent bias"
```

**Why this works for conditioning:**

```
Each class gets different feature space:
  Class 1: All feature vectors offset by [0.1, 0.2, -0.3, ...]
  Class 2: All feature vectors offset by [0.3, 0.1, -0.1, ...]
  Class 3: Different offset [0.5, -0.2, 0.4, ...]
  ...
  
Single model learns 1000 different "modes"
One mode per class
Each mode optimized for its class

During training:
  Network sees images with class labels
  Learns offset for each class that produces good reconstructions
  
During generation:
  Specify class → use that class's offset
  Generate images of specified class
```

---

### Equation 4: Conditional PixelCNN (Spatial)

```
y = tanh(W_{k,f} * x + V_{k,f} * s) ⊙ σ(W_{k,g} * x + V_{k,g} * s)
```

**Difference from non-spatial:**

```
Non-spatial:  y = tanh(W * x + V^T h)
              V^T h: matrix-vector multiply, same for all locations
              
Spatial:      y = tanh(W * x + V * s)
              V * s: convolution, location-dependent!
              
s = spatial conditioning (same dimensions as output)
    Example: segmentation map [H, W, n_classes]
    Each spatial location has conditioning info
```

**Example:**

```
Input segmentation: s ∈ R^{32×32×50}
At location (0,0): s[0,0] = one-hot for "sky" class
At location (15,15): s[15,15] = one-hot for "grass" class
... etc

Convolution: V_{k,f} * s ∈ R^{32×32×256}
At location (0,0): V_{k,f} applied to s[0,0] → feature bias
At location (15,15): different, based on s[15,15]

Result:
  Each location gets different feature bias
  Based on its segmentation class
  Sky locations "know" they should be blue
  Grass locations "know" they should be green
```

**Why convolution for spatial?**

```
If used V^T h (non-spatial):
  All locations get same bias
  Loses spatial information
  Can't tell pixel (0,0) is "sky" vs "grass"

With V * s (spatial):
  Convolution is used (1×1 convolution typically)
  No spatial mixing (just channel mixing)
  Each location conditioned only on itself
  Preserves spatial information from segmentation
```

**Practical use:**

```
Semantic layout → beautiful image generation:
1. Input: segmentation map showing "sky" vs "grass" vs "tree"
2. Each spatial location in segmentation is one-hot
3. V * s produces conditioning for each location
4. PixelCNN generates image pixels conditioned on layout
5. Result: coherent semantic scene following layout
```

---

## PAPER 3: VQ-VAE-2

### Equation 1: Hierarchical Reconstruction

The architecture uses two stages of encoding/decoding:

```
Stage 1 (Bottom encoder):
  x (256×256×3) → Encoder_bottom → z_e_bottom (64×64×D)
  
Stage 2 (Top encoder):
  z_e_bottom → Encoder_top → z_e_top (32×32×D)
  
Quantization:
  e_bottom = Quantize(z_e_bottom)  [64×64 codes]
  e_top = Quantize(z_e_top)        [32×32 codes]
  
Decoding:
  Decoder(e_bottom, e_top) → x̂ (256×256×3)
```

**Key insight - Independent dependencies:**

Each level depends on ORIGINAL IMAGE, not previous level:

```
Traditional hierarchy (WRONG):
  e_bottom depends only on e_top
  z_e_bottom = Encoder(e_top)  ← Lossy! Can't reconstruct from compressed e_top
  
VQ-VAE-2 (CORRECT):
  e_bottom depends on full image
  z_e_bottom = Encoder_bottom(image)  ← Sees all image detail
  
  e_top conditioned on bottom, but still sees image
  z_e_top = Encoder_top(image, e_bottom)  ← Sees image too!
```

---

### Equation 2: VQGAN Loss (Complete)

```
L_VQGAN(E, G, Z, D) = ||x - D(e)||² + λ·L_GAN({E,G,Z}, D) + β||e - sg[E(x)]||²
                      ├─ Reconstruction        ├─ Adversarial     ├─ Commitment
                      │ (MSE original only)    │ (GAN added here)  │ (from VQ-VAE)
                      └───────────────────────┴─────────────────┴──────────────
```

**What's different from VQ-VAE:**

```
VQ-VAE:
  L_rec = ||x - D(e)||²  (pixel MSE)
  
VQ-VAE-2 (VQGAN):
  L_rec = ||f(x) - f(D(e))||²  (perceptual loss on features)
  + L_GAN terms (adversarial)
  + λ adaptive weighting
```

**VQGAN Improvement 1: Perceptual Loss**

```
Original pixel MSE:
  L_MSE = ||x - x̂||²
  
Perceptual loss:
  L_perc = ||f(x) - f(x̂)||²
  
Where:
  f = feature extractor (pre-trained ResNet layer)
  
Why better:
  MSE penalizes pixel-level differences
  Perceptual penalizes feature-level differences
  Features match human perception better
```

**VQGAN Improvement 2: Adversarial Loss**

```
L_GAN = E_x[log D(x)] + E_x̂[log(1 - D(x̂))]

Where:
  D = discriminator (classifies real vs fake)
  First term: Discriminator rewards saying real images are real
  Second term: Discriminator rewards saying fakes are fake
  
Generator wants to minimize this:
  → Make discriminator think fake is real
  → D(x̂) → 1
  → log(1 - D(x̂)) → log(0) = -∞
  
But from generator's perspective:
  Minimize: -log D(x̂)  [or maximize log D(x̂)]
```

**VQGAN Improvement 3: Adaptive Weight λ**

```
λ = ||∇_G L_rec|| / (||∇_G L_GAN|| + ε)

Intuition:
  If reconstruction gradients large: λ increases (weight GAN less)
  If GAN gradients large: λ decreases (weight reconstruction more)
  Goal: Balance gradient magnitudes so both contribute equally
  
Example:
  ∇ L_rec = [10, 5, -3] → magnitude ≈ 11.6
  ∇ L_GAN = [0.1, 0.05, -0.03] → magnitude ≈ 0.116
  
  λ = 11.6 / (0.116 + 10^-6) ≈ 100
  
  This makes λ·L_GAN have similar gradient magnitude as L_rec
  Both losses contribute fairly to updates
```

---

### Equation 3: Hierarchical Prior Sampling

```
Top-level: e_top ~ p(e_top)
Bottom-level: e_bottom ~ p(e_bottom | e_top)
Reconstruction: x̂ = Decoder(e_top, e_bottom)
```

**Why two stages:**

```
Single prior p(all codes):
  Must learn distribution over all 4096 codes (64×64)
  Very complex (high-dimensional)
  
Hierarchical approach:
  Top prior: p(e_top) 
    - Only 1024 codes (32×32)
    - Learns global structure
    - Can use self-attention (feasible)
    
  Bottom prior: p(e_bottom | e_top)
    - Conditioned on top (less entropy)
    - Learns local structure
    - Simpler distribution to model
    
Together more expressive than single prior!
```

---

## PAPER 4: TAMING TRANSFORMERS

### Equation 1: VQGAN Quantization

```
z_q = q(ẑ) := arg min_{z_k ∈ Z} ||ẑ_{ij} - z_k||
```

**Breaking it down:**

```
ẑ = encoded representation from encoder
    Shape: [h, w, n_z]
    
ẑ_{ij} = code at spatial position (i,j)
         Shape: [n_z]
         
Z = {z_1, z_2, ..., z_K} = codebook
    K = size of discrete space (e.g., 1024)
    z_k = k-th codebook entry
    
arg min = find which k minimizes distance

||ẑ_{ij} - z_k|| = L2 distance between encoder output and code

Result: z_q[i,j] = z_k where k minimizes distance
```

---

### Equation 2: Straight-Through Estimator for VQGAN

```
L_VQ(E, G, Z) = ||x - x̂||² + ||sg[E(x)] - z_q||² + ||sg[z_q] - E(x)||²
```

**Difference from VQ-VAE:**

```
VQ-VAE Loss:
  L = ||x - x̂||² + ||sg[z_e] - e_k||² + β||z_e - sg[e_k]||²
  Three separate terms
  
Taming's notation (same thing):
  L = ||x - x̂||² + ||sg[E(x)] - z_q||² + ||sg[z_q] - E(x)||²
  E(x) = encoder output (like z_e)
  z_q = quantized code (like e_k)
  sg = stop-gradient
```

**What's new in Taming:**

- Uses this quantization as FIRST STAGE
- Then applies Transformer SECOND STAGE
- The VQGAN is adversarially trained (adds GAN loss)

---

### Equation 3: Autoregressive Transformer on Codes

```
p(s) = ∏_i p(s_i | s_{<i})

Where:
  s ∈ {0, ..., |Z|-1}^{h·w}
  |Z| = codebook size (e.g., 1024)
  h·w = number of spatial locations (e.g., 16×16 = 256)
  
s_i = integer code at position i
s_{<i} = all codes before position i in sequence
```

**The sequence:**

```
Original codes (spatial):
  z_q ∈ R^{h×w×n_z}
  Example: 16×16 codes, each is vector
  
Convert to indices:
  s[i,j] = k such that z_q[i,j] = Z[k]
  
Flatten to sequence:
  s = [s[0,0], s[0,1], ..., s[15,15]]
  s ∈ {0, ..., 1023}^{256}
  (256 positions, each 0-1023)

Autoregressive:
  p(s[0,0]) = prior (nothing to condition on)
  p(s[0,1] | s[0,0]) = learned by transformer
  p(s[0,2] | s[0,0], s[0,1]) = learned by transformer
  ...
  p(s[255] | s[0,...,254]) = learned by transformer
```

**During generation:**

```
Algorithm:
  s = []  (empty sequence)
  for i in range(256):
    logits = Transformer(s)  # predict distribution
    probs = softmax(logits[i])  # convert to probabilities
    s[i] = sample_from(probs)  # sample next code
    
Result: Generated sequence s
Then: Map back to codes and decode
  x̂ = Decoder([Z[s[0]], Z[s[1]], ..., Z[s[255]]])
```

---

### Equation 4: Gated Convolution with Conditioning (Transformer version)

```
y = tanh(W_{k,f} * x + V_{k,f}^T h) ⊙ σ(W_{k,g} * x + V_{k,g}^T h)
```

This is same as PixelCNN! But applied in transformer layers instead.

**In transformer context:**

```
x = token embeddings from previous layer
h = conditioning (e.g., segmentation codes)
y = output after gating

Transformer layer typically:
  1. Attention(x, x, x)
  2. Feed-forward(x)
  
With conditioning:
  1. Attention(x + condition, ...)
  2. Feed-forward with gated activation
```

---

### Equation 5: Sliding Window Attention Mask

```
Attn(Q, K, V) = softmax(QK^T / √d_k) V
where QK^T is masked for causality AND sliding window
```

**Masking for sliding window:**

```
Standard causal mask:
  Mask[i, j] = 0 if j ≤ i (can attend)
  Mask[i, j] = -∞ if j > i (cannot attend, masked out)

Sliding window mask:
  Mask[i, j] = 0 if j ≤ i AND j ≥ i - window_size
  Mask[i, j] = -∞ otherwise
  
Example (window size 5):
  Position 10 can attend to: 5, 6, 7, 8, 9, 10
  Position 10 CANNOT attend to: 0, 1, 2, 3, 4, 11, 12, ...
  
Benefit:
  Instead of O(n²) attention to all previous
  Only O(window_size × n) attention
  Linear instead of quadratic!
```

---

## COMPREHENSIVE EQUATION REFERENCE TABLE

Here's a quick reference for all major equations:

|Paper|Eq#|Equation|Name|Key Insight|
|---|---|---|---|---|
|VQ-VAE|1|q(z=k\|x) one-hot|Discrete posterior|Ensures latent use|
|VQ-VAE|2|z_q = e_k nearest|Quantization|Maps to discrete codes|
|VQ-VAE|3|L = rec + VQ + commit|Complete loss|Three balanced terms|
|VQ-VAE|4|e_i = m_i/N_i (EMA)|Codebook update|K-means clustering|
|PixelCNN|1|p(x) = ∏ p(x_i\|x_<i)|Factorization|Tractable likelihood|
|PixelCNN|2|y = tanh⊙σ|Gating|Multiplicative units|
|PixelCNN|3|y = tanh(W*x + V^T h)⊙...|Condition.|Class-dependent bias|
|PixelCNN|4|y = tanh(W_x + V_s)⊙...|Spatial cond.|Location-dependent bias|
|VQVAE-2|1|Hierarchical encoding|Two-level codes|Global + local|
|VQVAE-2|2|L = rec + λGAN + commit|VQGAN|Perceptual + adversarial|
|VQVAE-2|3|e_top ~ p, e_bot ~ p(\|e_top)|Hierarchical prior|Specialized priors|
|Taming|1|z_q = argmin \|ẑ - z_k\||Quantization|Discrete codes|
|Taming|2|L = rec + VQ + commit|VQGAN loss|Same as VQVAE-2|
|Taming|3|p(s) = ∏ p(s_i\|s_<i)|Autoregressive|Code-level modeling|
|Taming|4|y = tanh⊙σ with h|Gated conditioning|Combined gate+condition|
|Taming|5|Attn w/ sliding window|Local attention|O(n) instead O(n²)|

---

## CHEAT SHEET: WHICH EQUATION TO EXPLAIN

**If exam asks about:**

1. **VQ-VAE** → Explain Eq 3 (three-part loss)
    
    - Shows understanding of posterior collapse prevention
    - Demonstrates three different learning mechanisms
2. **PixelCNN** → Explain Eq 2 (gated activation)
    
    - Shows why it improves over standard CNN
    - Discusses multiplicative interactions
3. **VQ-VAE-2** → Explain Eq 2 (VQGAN loss)
    
    - Shows improvements over VQ-VAE
    - Discusses perceptual vs pixel loss
    - Explains adaptive weighting
4. **Taming** → Explain Eq 3 (autoregressive on codes)
    
    - Shows sequence modeling on compressed codes
    - Explains why this is faster than pixel-level
    - Demonstrates factorization principle
5. **Any comparison** → Explain loss functions
    
    - Reconstruction: what gets optimized
    - Adversarial: why GAN helps
    - Commitment: why stability needed

---

**Good luck with your exam! These explanations should help you understand not just WHAT the equations are, but WHY they're written that way and WHAT they're trying to accomplish.**