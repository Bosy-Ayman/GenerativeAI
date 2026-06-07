# 📖 Repo Sheet Analysis & Core Relations

This document analyzes the questions in [Sheet.pdf](file:///c:/Users/pouss/Documents/CSAI/4th%20Year/Spring/GenerativeAI/Slides/Sheet.pdf), breaks down the **mathematical relations** that always exist, and explains how to solve any variation the professor might introduce in tomorrow's final exam.

---

## 🗺️ The 5 Core Questions on the Sheet

The sheet contains 5 distinct questions:
1. **VAE Normal Distribution & Z-Score Peaks Tracing** (Mickey & Donald)
2. **KL Divergence Step-by-Step Calculation**
3. **Forward Diffusion (Noising) Step Tracing**
4. **VAE Latent Dimension Manipulation Coding** (Even/Odd index shifts)
5. **GAN Latent Walk & Image Normalization Coding** (Interpolating $z_1 + z_2$)

---

## 🧠 Core Relation 1: VAE Normal Distribution & Standard Deviation Units

### The Sheet Question:
Given the probability densities for Mickey ($\mu=1, \sigma=1$) and Donald ($\mu=4, \sigma=1$):
* $q(z|x=\text{Mickey}) = \mathcal{N}(z; 1, 1)$
* $q(z|x=\text{Donald}) = \mathcal{N}(z; 4, 1)$
Determine their peaks.

### 📐 The Hidden Mathematical Relation:
A normal distribution is defined as:
$$\mathcal{N}(z; \mu, \sigma) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(z-\mu)^2}{2\sigma^2}}$$

1. **Peak Location**: The maximum value (peak) of any Gaussian distribution is **always at the mean ($z = \mu$)**.
   * Mickey's peak is at $z = 1$.
   * Donald's peak is at $z = 4$.
2. **Symmetry**: The density is perfectly symmetric around the mean:
   $$\mathcal{N}(\mu - \Delta; \mu, \sigma) = \mathcal{N}(\mu + \Delta; \mu, \sigma)$$
3. **Z-Score (Standardized Distance) Rule**:
   The density value is determined solely by how many standard deviations ($Z$-score) the point is from the mean:
   $$Z = \frac{z - \mu}{\sigma}$$
   Because both Mickey and Donald have $\sigma = 1$, points that are the same distance $|z - \mu|$ from their respective means have **exactly the same density**:
   * **Distance = 0 (Peak)**: $\mathcal{N}_m(z=1) = \mathcal{N}_d(z=4) \approx \mathbf{0.398}$
   * **Distance = $1\sigma$**: $\mathcal{N}_m(z=0) = \mathcal{N}_m(z=2) = \mathcal{N}_d(z=3) = \mathcal{N}_d(z=5) \approx \mathbf{0.241}$
   * **Distance = $2\sigma$**: $\mathcal{N}_m(z=-1) = \mathcal{N}_m(z=3) = \mathcal{N}_d(z=2) = \mathcal{N}_d(z=6) \approx \mathbf{0.053}$
   * **Distance = $3\sigma$**: $\mathcal{N}_m(z=-2) = \mathcal{N}_m(z=4) = \mathcal{N}_d(z=1) = \mathcal{N}_d(z=7) \approx \mathbf{0.004}$

### 🔄 How the Professor Might Change It:
If the professor changes the means or standard deviations (e.g., Donald has $\mu=5, \sigma=2$):
* The peak will shift to the new mean ($z = 5$).
* The peak value will scale by $1/\sigma$: peak value = $0.3989 / 2 = 0.199$.
* Standardized distances will scale: a point $1\sigma$ away from the mean ($z = 5 \pm 2$) will have density $0.241 / 2 = 0.1205$.

---

## 🧠 Core Relation 2: KL Divergence Asymmetry & Log Base Conversion

### The Sheet Question:
Calculate $D_{KL}(P \| Q)$ and $D_{KL}(Q \| P)$ for given discrete distributions over $x \in \{-1, -0.5, 0.5, 1\}$.

### 📐 The Hidden Mathematical Relation:
1. **Asymmetry**:
   $$D_{KL}(P \| Q) \neq D_{KL}(Q \| P)$$
   This relation always holds because KL divergence is a directional measure of information difference.
2. **Log base conversion**:
   $$\ln(y) = 2.303 \log_{10}(y)$$
   In exams, write out your calculations in a table format to ensure you receive full credit for steps.
3. **Non-negativity**:
   $$D_{KL}(P \| Q) \geq 0$$
   If you calculate a negative sum, you made an arithmetic error.

---

## 🧠 Core Relation 3: Forward Diffusion Coefficients & Variance Preservation

### The Sheet Question:
Trace the $x_0$ image to reach $x_3$ using $\alpha_0=0.9$, $\alpha_1=0.8$, $\alpha_2=0.6$.

### 📐 The Hidden Mathematical Relation:
The single-step forward diffusion equation is:
$$x_t = \sqrt{\alpha_{t-1}} x_{t-1} + \sqrt{1 - \alpha_{t-1}} \epsilon_{t-1}$$

The coefficients for the image component ($\sqrt{\alpha}$) and the noise component ($\sqrt{1 - \alpha}$) satisfy:
$$\left(\sqrt{\alpha}\right)^2 + \left(\sqrt{1 - \alpha}\right)^2 = \alpha + (1 - \alpha) = 1$$

This is the **Variance Preservation Relation**. It ensures that the overall variance of the noised image remains scaled and stable (does not blow up to infinity).
* For $\alpha_0 = 0.9$:
  $$\sqrt{0.9} \approx 0.949 \quad \text{and} \quad \sqrt{0.1} \approx 0.316 \implies (0.949)^2 + (0.316)^2 \approx 1$$
* For $\alpha_1 = 0.8$:
  $$\sqrt{0.8} \approx 0.894 \quad \text{and} \quad \sqrt{0.2} \approx 0.447 \implies (0.894)^2 + (0.447)^2 \approx 1$$
* For $\alpha_2 = 0.6$:
  $$\sqrt{0.6} \approx 0.775 \quad \text{and} \quad \sqrt{0.4} \approx 0.632 \implies (0.775)^2 + (0.632)^2 \approx 1$$

### 🔄 How the Professor Might Change It:
The professor will change the values of $\alpha$ (e.g., $\alpha_0 = 0.95, \alpha_1 = 0.9, \alpha_2 = 0.85$). 
**Rule**: Simply compute the square root of your given $\alpha$ for the first term, and the square root of $(1 - \alpha)$ for the noise term.

---

## 🧠 Core Relation 4: Latent Variables vs. Distribution Parameters

### The Sheet Question:
You write code to modify dimension values of random latent vectors $z$ (adding/subtracting `val = 1.3`). Does this change the distribution parameters or the latent values?

### 📐 The Hidden Coding Relation:
* **The Answer**: This modifies **latent values ($Z$) directly**, NOT the distribution parameters ($\mu, \sigma$).
* **The Reason**:
  * The encoder outputs the distribution parameters $\mu$ and $\sigma$ representing the Gaussian distribution of the input.
  * $z$ is a vector sampled from that distribution ($z = \mu + \sigma \cdot \epsilon$).
  * Modifying the elements of $z$ (adding/subtracting values) shifts the point we feed to the decoder, but it does **not** change the parameters $\mu$ and $\sigma$ that define the encoder's original distribution.

---

## 🧠 Core Relation 5: GAN LatentWalk & Output Image Normalization

### The Sheet Question:
Write code to create $z = z_1 + z_2$, step-modify them by adding 0.3, generate images, and normalize colors.

### 📐 The Hidden Coding Relation:
1. **Linear Latent Addition**:
   Combining $z_1$ and $z_2$ by direct addition ($z = z_1 + z_2$) creates a composite latent vector that merges features of both sources.
2. **PyTorch Gradient Freezing**:
   Modifying latent variables in-place during tracing loops must bypass PyTorch's tracking to avoid modifying the graph. We use `.data` to modify values in-place without triggering autograd tracking:
   ```python
   z1.data[0, i, 0, 0] += 0.3
   ```
3. **Tanh Output Normalization**:
   GAN generators typically use the **Tanh activation function** in their final layer, which outputs pixel values in the range $[-1, 1]$. To display or save them as standard images in the range $[0, 1]$, you **must** apply the normalization relation:
   $$\text{img}_{\text{normalized}} = \frac{\text{img} + 1}{2}$$
   This division by 2 and shift by 0.5 is a standard pattern in all GAN/CGAN code.
