# 🔮 Final Exam Predictions — Generative AI (AI442)

This document contains predictions and analysis for tomorrow's exam based on actual exam patterns from the past two years. Since **PCA and Huffman Coding are excluded**, we focus entirely on **Word2Vec, Autoencoders, VAEs, GANs/CGANs, and Diffusion Models**.

The exam is structured as:
* **8 Marks**: Multiple Choice Questions (MCQs) — 8 Questions
* **20 Marks**: Tracing Questions — 3 Questions
* **12 Marks**: Coding Questions — 2 Questions

---

## 🎯 Part 1: MCQ Predictions [8 Marks]
The MCQs will test conceptual understanding, parameter counts, activation functions, and mathematical properties. Expect exactly 8 questions on the following concepts:

### Prediction 1: Autoencoder Architecture & Sizing
* **Concept**: Sizing layer counts symmetrically.
  * **Rule**: In an Encoder-Decoder structure `[N1 - N2 - N3 - N4 - N5]`, the bottleneck `N3` is the compressed latent dimension (always less than the surrounding layers), and the final layer `N5` must equal the input layer `N1` to reconstruct the image.
  * **Likely Q**: "Find the missed layer to make the model an Encoder-Decoder: `[80 - 15 - ? - 15 - 80]`" → *Answer: any number less than 15 (bottleneck)*.
  * **Likely Q**: "Find the missed layer: `[40 - 20 - 5 - 20 - ?]`" → *Answer: 40 (must match input)*.

### Prediction 2: UpSampling vs. Conv2DTranspose
* **Concept**: How spatial size increases in decoders.
  * **Rule**: `UpSampling2D` has **zero** learnable parameters (just duplicates pixels/interpolates). `Conv2DTranspose` does learned upscaling and has **learnable weights/parameters**.
  * **Likely Q**: "What is the number of parameters for the UpSampling2D layer?" → *Answer: zero*.
  * **Likely Q**: "Scaling up an image using learning weights is..." → *Answer: Conv2DTranspose*.

### Prediction 3: KL Divergence Properties
* **Concept**: Mathematical characteristics of Kullback-Leibler Divergence.
  * **Rule**:
    1. KL Divergence is asymmetric: $D_{KL}(P \| Q) \neq D_{KL}(Q \| P)$.
    2. KL Divergence is always non-negative: $D_{KL}(P \| Q) \geq 0$ (and equals 0 if and only if $P = Q$).
  * **Likely Q**: "Which of the following is true for KL Divergence?" → *Answer: $D_{KL}(P \| Q) \neq D_{KL}(Q \| P)$*.
  * **Likely Q**: "KL Divergence measure result may be..." → *Answer: $\geq 0$*.

### Prediction 4: VAE log-variance Output
* **Concept**: Why we output log-variance instead of variance in VAE.
  * **Rule**: Variance $\sigma^2$ must always be positive (> 0), which is hard for a neural network to guarantee directly without constraints. log-variance $\log(\sigma^2)$ can be any real number ($-\infty$ to $+\infty$), which is natural for dense layers to output.
  * **Likely Q**: "Why does the VAE encoder output log-variance $\log(\sigma^2)$ instead of variance $\sigma^2$?" → *Answer: Because log-variance can be any real number, making it easier for the network to output without constraints.*

### Prediction 5: VAE Latent Vector Sampling
* **Concept**: The latent vector $z$ in a VAE is not a deterministic learned layer; it is sampled from a distribution: $z \sim \mathcal{N}(\mu, \sigma^2)$ using the reparameterization trick.
  * **Likely Q**: "The latent vector (Z) in VAE is..." → *Answer: Sampled from a distribution*.

### Prediction 6: GAN Detach & Gradients
* **Concept**: Preventing gradient updates to the Generator when training the Discriminator.
  * **Rule**: We use `fake_imgs.detach()` because we do not want backpropagation to modify G's weights during the D training step.
  * **Likely Q**: "Why do we use `.detach()` on fake images when training the Discriminator?" → *Answer: To block gradient flow and freeze the Generator's weights*.

### Prediction 7: CGAN Input Channels & Embeddings
* **Concept**: How conditional information is fed to G and D.
  * **Rule**: In G, class label is a small vector (e.g., size 10) concatenated with $z$. In D, class label is mapped to a large 2D matrix (same size as image, e.g. 28x28) and stacked as a second channel.
  * **Likely Q**: "How many input channels does a CGAN Discriminator take for a grayscale image?" → *Answer: 2 channels (1 for the image, 1 for the label embedding map)*.

### Prediction 8: Diffusion Model Target
* **Concept**: The objective of the neural network in Diffusion.
  * **Rule**: The model takes the noisy image $x_t$ and the timestep $t$, and predicts the **noise vector $\epsilon$** that was added (not the clean image).
  * **Likely Q**: "In Diffusion Models, what does the neural network predict at each step?" → *Answer: The noise vector ($\epsilon$)*.

---

## 📊 Part 2: Tracing Predictions [20 Marks]
The tracing section will have 3 questions requiring manual calculations and plotting.

### Tracing Q1: Forward Diffusion Step Tracing [5 Marks]
* **Prediction**: You will be given a small 1D original image array $x_0$, a noise schedule of $\alpha_t$ values, and random noise vectors $\epsilon_t$. You will be asked to compute the noisy image vectors $x_1$ and $x_2$ at timesteps $t=1$ and $t=2$ step-by-step.
* **Formula**:
  * $x_1 = \sqrt{\alpha_0} x_0 + \sqrt{1 - \alpha_0} \epsilon_0$
  * $x_2 = \sqrt{\alpha_1} x_1 + \sqrt{1 - \alpha_1} \epsilon_1$
* **Pro-Tip**: Calculate the square roots to 4 decimal places first. Perform element-wise scaling and addition carefully.

### Tracing Q2: Word2Vec Vector Addition & Plotting [5 Marks]
* **Prediction**: You will be given a table of 15–20 words with 2D embedding coordinates. You will be asked to:
  1. Plot these words on a 2D coordinate plane.
  2. Compute vector addition: $\text{newVec} = \text{word1} + \text{word2}$.
  3. Determine the nearest two words to the $\text{newVec}$ based on the plotted distance or direct calculations.
* **Pro-Tip**: Calculate the Euclidean distance squared: $d^2 = (x - x_{\text{new}})^2 + (y - y_{\text{new}})^2$ to confirm your visual answer mathematically!

### Tracing Q3: KL Divergence step-by-step [10 Marks]
* **Prediction**: You will be given two probability distributions $P(x)$ and $Q(x)$ over a small set of values $x \in \{-1, -0.5, 0.5, 1\}$ and asked to compute:
  1. $D_{KL}(P \| Q)$
  2. $D_{KL}(Q \| P)$
* **Pro-Tip**: Use the hint formula: $\ln(y) = 2.303 \log_{10}(y)$ carefully. Write down the term for each value of $x$ to show your steps and avoid calculation errors. Remember that the final sum cannot be negative.

---

## 💻 Part 3: Coding Predictions [12 Marks]
The coding section will require writing Python code iteratively. Expect exactly 2 questions (6 marks each).

### Coding Q1: AutoEncoder Latent Walking & Interpolation [6 Marks]
* **Prediction**: Write a search function that walks the latent space between pairs of test encodings to find a generation similar to a target image.
* **Key Code Pattern**:
  ```python
  import numpy as np
  
  # Results has shape (num_samples, latent_dim)
  Results = encoder.predict(x_test)
  those = []
  
  for i in range(len(Results)):
      for k in range(i + 1, len(Results)):
          # Variable step count loop (e.g., 10 to 20 steps)
          for steps in range(10, 21):
              # Interpolate between Z_i and Z_k
              interpolated = np.linspace(Results[i], Results[k], steps)
              # Decode the interpolated steps
              decoded_imgs = decoder.predict(interpolated)
              
              for a in range(steps):
                  if Dist(decoded_imgs[a], Targt) <= threshold:
                      those.append(decoded_imgs[a])
  ```

### Coding Q2: VAE Target Image Reconstruction & Mean Shifting [6 Marks]
* **Prediction**: Write code that manipulates an image (crop or flip), modifies encoder distribution parameters (shifting means to negative), and runs a search loop using the reparameterization trick to find the best match.
* **Key Code Pattern**:
  ```python
  import numpy as np
  
  # 1. Image crop or flip
  # Crop top half:
  cropped_target = TargetImage.reshape(28, 28)[0:14, :].flatten()
  # OR Flip vertically:
  flipped_target = np.flip(TargetImage.reshape(28, 28), axis=0).flatten()
  
  # 2. Mean modification logic
  # Example: make positive means negative
  modified_mus = all_mus.copy()
  modified_mus[modified_mus > 0] = -modified_mus[modified_mus > 0]
  
  # 3. Search Loop
  best_dist = float('inf')
  best_i = -1
  best_noise = None
  
  for i in range(len(x_train)):
      # Retrieve mean and log-variance
      mu = modified_mus[i:i+1] # pre-modified mean
      _, log_var = encoder.predict(x_train[i:i+1])
      
      # Try noise samples
      for _ in range(5):
          noise = np.random.normal(size=mu.shape)
          sigma = np.exp(0.5 * log_var)
          
          # Reparameterization: z = mu + sigma * noise
          z = mu + sigma * noise
          decoded = decoder.predict(z)
          
          # Compare target (complete or cropped)
          distance = Dist(decoded.flatten(), flipped_target)
          
          if distance < best_dist:
              best_dist = distance
              best_i = i
              best_noise = noise
  ```
