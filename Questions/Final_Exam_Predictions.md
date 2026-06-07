# 🔮 Final Exam Predictions — Generative AI (AI442)

> ⚠️ **IMPORTANT STRATEGIC NOTE (الأهم على الإطلاق):**
> CGAN (Conditional GAN) and Diffusion Models were **NOT** studied in previous terms (NMM exams), which is why they do not appear in the 2023 or 2024 old final exams. However, they are part of the curriculum this semester. Because they are **new topics**, the professor is highly likely to give them **substantial weight** in tomorrow's final exam. 
> 
> **بالعربي:** محاضرات الـ CGAN والـ Diffusion ما كانتش بتُدرس السنين اللي فاتت، وعشان كدا مش موجودين في الامتحانات القديمة. بما إنهم اتدرسوا الترم ده، فبنسبة كبيرة جداً هيكون عليهم جزء كبير ومهم من درجات الامتحان بكره! ركز عليهم كويس جداً.

---

## 🔥 Hot Topic Focus: CGAN & Diffusion (High Probability Questions)

### 1. CGAN (Conditional GAN) Key Concepts:
* **The Label Injection Mechanism**: 
  * In the **Generator**, we convert the class integer to a small embedding vector (`nn.Embedding(num_classes, embedding_dim)`), unsqueeze it to match $z$'s dimensions (shape `batch, emb_dim, 1, 1`), and concatenate it with the random noise vector $z$ along the channel dimension (`dim=1`).
  * In the **Discriminator**, we convert the class integer to a spatial embedding map (`nn.Embedding(num_classes, 28 * 28)`), reshape it to match the image dimensions (shape `batch, 1, 28, 28`), and concatenate it with the input image as a second channel along the channel dimension.
* **Architecture Difference**:
  * Generator input channels = `latent_dim + embedding_dim`.
  * Discriminator input channels = `1 + 1 = 2` channels (for grayscale MNIST).

### 2. Diffusion Models Key Concepts:
* **Noising (Forward Process)**: 
  * $x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{1 - \alpha_t} \epsilon$.
  * It can also jump directly from $x_0$ to any step $x_t$ using the cumulative product $\hat{\alpha}_t$: $$x_t = \sqrt{\hat{\alpha}_t} x_0 + \sqrt{1 - \hat{\alpha}_t} \epsilon$$.
* **Denoising (Reverse Process)**:
  * The neural network (typically a U-Net) takes the noisy image $x_t$ and the timestep $t$ (noise level), and predicts the **noise vector $\epsilon$** that was added (NOT the clean image).
  * Loss is the Mean Squared Error (MSE) between the predicted noise and the actual noise: `loss = MSE(predicted_noise, noise)`.

---

## 🎯 Part 1: MCQ Predictions [8 Marks]
Expect exactly 8 questions on the following concepts:

### Prediction 1: Autoencoder Architecture & Sizing
* **Rule**: In an Encoder-Decoder structure `[N1 - N2 - N3 - N4 - N5]`, the bottleneck `N3` is the compressed latent dimension (always less than the surrounding layers), and the final layer `N5` must equal the input layer `N1` to reconstruct the image.
* **Likely Q**: "Find the missed layer to make the model an Encoder-Decoder: `[80 - 15 - ? - 15 - 80]`" → *Answer: any number less than 15 (bottleneck)*.

### Prediction 2: UpSampling vs. Conv2DTranspose
* **Rule**: `UpSampling2D` has **zero** learnable parameters. `Conv2DTranspose` performs learned upscaling and has **learnable weights/parameters**.
* **Likely Q**: "What is the number of parameters for the UpSampling2D layer?" → *Answer: zero*.
* **Likely Q**: "Scaling up an image using learning weights is..." → *Answer: Conv2DTranspose*.

### Prediction 3: KL Divergence Properties
* **Rule**: KL Divergence is asymmetric ($D_{KL}(P \| Q) \neq D_{KL}(Q \| P)$) and always non-negative ($D_{KL}(P \| Q) \geq 0$).
* **Likely Q**: "Which of the following is true for KL Divergence?" → *Answer: $D_{KL}(P \| Q) \neq D_{KL}(Q \| P)$*.

### Prediction 4: VAE log-variance Output
* **Rule**: Variance $\sigma^2$ must always be positive (> 0), while log-variance $\log(\sigma^2)$ can take any real number value ($-\infty$ to $+\infty$), which is easier for standard dense layers to output without stability issues.
* **Likely Q**: "Why does the VAE encoder output log-variance $\log(\sigma^2)$ instead of variance $\sigma^2$ directly?" → *Answer: Because log-variance can be any real number, making it easier for the network to output without constraints.*

### Prediction 5: VAE Latent Vector Sampling
* **Rule**: The latent vector $z$ in a VAE is sampled from a distribution: $z \sim \mathcal{N}(\mu, \sigma^2)$ using the reparameterization trick: $z = \mu + \sigma \cdot \epsilon$, where $\epsilon \sim \mathcal{N}(0, 1)$.
* **Likely Q**: "The latent vector (Z) in VAE is..." → *Answer: Sampled from a distribution*.

### Prediction 6: GAN Detach & Gradients (Highly Probable!)
* **Rule**: We use `fake_imgs.detach()` because we do not want backpropagation to modify G's weights during the D training step.
* **Likely Q**: "Why do we use `.detach()` on fake images when training the Discriminator?" → *Answer: To block gradient flow and freeze the Generator's weights*.

### Prediction 7: CGAN Input Channels (Highly Probable!)
* **Rule**: In CGAN, the label is stacked as a second channel in the Discriminator.
* **Likely Q**: "How many input channels does a CGAN Discriminator take for a grayscale image?" → *Answer: 2 channels (1 for the image, 1 for the label embedding map)*.

### Prediction 8: Diffusion Model Target (Highly Probable!)
* **Rule**: The model takes the noisy image $x_t$ and the timestep $t$, and predicts the **noise vector $\epsilon$** that was added (not the clean image).
* **Likely Q**: "In Diffusion Models, what does the neural network predict at each step?" → *Answer: The noise vector ($\epsilon$)*.

---

## 📊 Part 2: Tracing Predictions [20 Marks]
The tracing section will have 3 questions requiring manual calculations and plotting.

### Tracing Q1: Forward Diffusion Step Tracing [5 Marks]
* **Prediction**: Since Diffusion is a new topic, a tracing question showing how noise is added step-by-step is highly likely. You will be given $x_0$, a noise schedule ($\alpha_t$ values), and random noise vectors $\epsilon_t$, and asked to compute $x_1$ and $x_2$.
* **Formula**:
  * $x_1 = \sqrt{\alpha_0} x_0 + \sqrt{1 - \alpha_0} \epsilon_0$
  * $x_2 = \sqrt{\alpha_1} x_1 + \sqrt{1 - \alpha_1} \epsilon_1$

### Tracing Q2: Word2Vec Vector Addition & Plotting [5 Marks]
* **Prediction**: You will be given a table of words with 2D embedding coordinates. You will be asked to:
  1. Plot these words on a 2D coordinate plane.
  2. Compute vector addition: $\text{newVec} = \text{word1} + \text{word2}$.
  3. Determine the nearest two words to the $\text{newVec}$ based on the plotted distance or direct calculations.

### Tracing Q3: KL Divergence step-by-step [10 Marks]
* **Prediction**: Calculate $D_{KL}(P \| Q)$ and $D_{KL}(Q \| P)$ step-by-step for a given 4-element probability distribution, using the hint: $\ln(y) = 2.303 \log_{10}(y)$. Show that they are asymmetric.

---

## 💻 Part 3: Coding Predictions [12 Marks]
The coding section will require writing Python code iteratively. Expect exactly 2 questions.

### Coding Q1: CGAN Embedding Injection & Generator Forward Step [6 Marks]
* **Prediction**: Since CGAN is a new topic, a coding question about label injection in PyTorch is highly probable. You will be asked to write the `forward` function of a CGAN Generator, showing how class labels are embedded, reshaped, and concatenated with the noise vector $z$ before passing through the transpose convolutions.
* **Key Code Pattern**:
  ```python
  def forward(self, z, labels):
      # 1. Embed class labels
      label_embedding = self.label_emb(labels)  # shape: (batch, emb_dim)
      # 2. Reshape/unsqueeze to 4D to match z
      label_embedding = label_embedding.unsqueeze(2).unsqueeze(3)  # shape: (batch, emb_dim, 1, 1)
      # 3. Concatenate along channel dimension (dim=1)
      x = torch.cat([z, label_embedding], dim=1)  # shape: (batch, latent_dim + emb_dim, 1, 1)
      # 4. Pass through transpose convolutions
      return self.model(x)
  ```

### Coding Q2: VAE Target Image Reconstruction & Mean Shifting [6 Marks]
* **Prediction**: Write code that manipulates an image (crop or flip), modifies encoder distribution parameters (shifting means), and runs a search loop using the reparameterization trick to find the best match.
* **Key Code Pattern**:
  ```python
  # 1. Image crop or flip
  flipped_target = np.flip(TargetImage.reshape(28, 28), axis=0).flatten()
  
  # 2. Mean modification logic
  modified_mus = all_mus.copy()
  modified_mus[modified_mus > 0] = -modified_mus[modified_mus > 0]
  
  # 3. Search Loop
  best_dist = float('inf')
  best_i = -1
  
  for i in range(len(x_train)):
      mu = modified_mus[i:i+1]
      _, log_var = encoder.predict(x_train[i:i+1])
      
      for _ in range(5):
          noise = np.random.normal(size=mu.shape)
          sigma = np.exp(0.5 * log_var)
          z = mu + sigma * noise
          decoded = decoder.predict(z)
          distance = Dist(decoded.flatten(), flipped_target)
          
          if distance < best_dist:
              best_dist = distance
              best_i = i
  ```
