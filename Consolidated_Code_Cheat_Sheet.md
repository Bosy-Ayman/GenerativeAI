# 🧠 Consolidated Code Cheat Sheet — Generative AI (AI442)

This document gathers **all Python code sequences** you need to memorize for the final exam. It is divided into three sections:
1. **Core Course Summary Architectures** (Word2Vec, AE, VAE, GAN/CGAN, Diffusion)
2. **Code Blocks from Sheet.pdf** (Odd/even latent shifting, GAN latent summation)
3. **Code Blocks from Past Exams** (Word2Vec filtering, AE latent walking, VAE target matching)

---

# 📂 Section 1: Core Course Summary Architectures

## 1. Word2Vec Network Architecture (Keras/TensorFlow)
* **Goal**: Build a simple bottleneck network to train and extract word embedding vectors.
* **The Code**:
```python
import tensorflow as tf
from tensorflow.keras import layers, Input, Model

# Input shape: unique words in dictionary (e.g., 21 words)
inputs = Input(shape=(21,))

# Hidden bottleneck layer (No activation function, linear projection)
# The size (units=2) is the output embedding size
bottleneck = layers.Dense(units=2, activation='linear')(inputs)

# Output layer: projects back to dictionary size to predict context word
outputs = layers.Dense(units=21, activation='softmax')(bottleneck)

# Compile and train
model = Model(inputs, outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit(x_train, y_train, epochs=100, batch_size=10)

# Extract embedding weights (shape: 21, 2)
weights = model.get_weights()[0]
```

---

## 2. Dense Autoencoder & Separate Decoder (Keras/TensorFlow)
* **Goal**: Build an AE, and split it into an encoder and a decoder. Copy weights manually to the decoder.
* **The Code**:
```python
import tensorflow as tf
from tensorflow.keras import layers, Input, Model

# 1. Complete AutoEncoder
encoding_dim = 32
input_img = Input(shape=(784,))
x = layers.Dense(100, activation='relu')(input_img)
encoded = layers.Dense(encoding_dim, activation='relu')(x)      # Bottleneck
x = layers.Dense(100, activation='relu')(encoded)
decoded = layers.Dense(784, activation='sigmoid')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# 2. Extract Encoder Model
encoder = Model(input_img, encoded)

# 3. Build Separate Decoder Model (requires weight copy!)
encoded_input = Input(shape=(encoding_dim,))
a = layers.Dense(100, activation='relu')(encoded_input)
d_out = layers.Dense(784, activation='sigmoid')(a)
decoder = Model(encoded_input, d_out)

# 4. Weight Transfer (Manual assignment)
# In autoencoder: layer index 4 (Dense 100) and layer index 5 (Dense 784) are decoder layers.
# Copy weights from autoencoder layers [4, 5] to decoder layers [1, 2].
decoder.layers[1].set_weights(autoencoder.layers[4].get_weights())
decoder.layers[2].set_weights(autoencoder.layers[5].get_weights())
```

---

## 3. Convolutional Autoencoder (Keras/TensorFlow)
* **Goal**: Shrink images spatially using convolutions + pooling, then expand them back using UpSampling or Conv2DTranspose.
* **The Code**:
```python
from tensorflow.keras import layers, Input

# ============ ENCODER ============
input_img = Input(shape=(28, 28, 1))
x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2,2), padding='same')(x) # Shape: (14, 14, 16)
x = layers.Conv2D(8, (3,3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2,2), padding='same')(x) # Shape: (7, 7, 8)

# ============ DECODER Option A (UpSampling2D - No weights) ============
x = layers.Conv2D(8, (3,3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2,2))(x)                       # Shape: (14, 14, 8)
x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2,2))(x)                       # Shape: (28, 28, 16)
decoded = layers.Conv2D(1, (3,3), activation='sigmoid', padding='same')(x)

# ============ DECODER Option B (Conv2DTranspose - Learned weights) ============
# Output size calculation: Out = (In - 1) * stride + kernel - 2*padding
x = layers.Conv2DTranspose(8, (3,3), strides=(2,2), activation='relu', padding='same')(encoded) # (14,14,8)
x = layers.Conv2DTranspose(16, (3,3), strides=(2,2), activation='relu', padding='same')(x)     # (28,28,16)
decoded = layers.Conv2D(1, (3,3), activation='sigmoid', padding='same')(x)
```

---

## 4. VAE Custom Sampling Layer & Model Loss (Keras/TensorFlow)
* **Goal**: Build a custom sampling layer for the reparameterization trick and calculate custom VAE loss.
* **The Code**:
```python
import tensorflow as tf
from tensorflow.keras import layers, Model

# Custom Sampling Layer: z = mean + exp(0.5 * log_var) * epsilon
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# VAE Model Wrapper to inject Reconstruction + KL Loss
class VAE(Model):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)

        # 1. Reconstruction Loss (binary cross entropy * dimensions)
        recon_loss = tf.keras.losses.binary_crossentropy(inputs, reconstructed)
        recon_loss = tf.reduce_mean(recon_loss) * 784

        # 2. KL Divergence Loss
        kl_loss = -0.5 * tf.reduce_sum(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1
        )
        kl_loss = tf.reduce_mean(kl_loss)

        self.add_loss(recon_loss + kl_loss)
        return reconstructed
```

---

## 5. PyTorch GAN Generator & Discriminator Models
* **Goal**: Define standard Generator and Discriminator classes in PyTorch.
* **The Code**:
```python
import torch
import torch.nn as nn

# Generator: Noise (z) -> Image
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.model = nn.Sequential(
            # Input: latent_dim -> 256
            nn.ConvTranspose2d(latent_dim, 256, kernel_size=7, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 256 -> 128
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 128 -> 1 (output grayscale channel)
            nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Output values range from [-1, 1]
        )

    def forward(self, z):
        return self.model(z)

# Discriminator: Image -> Probability (Real or Fake)
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # Input: 1 channel -> 64 channels
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),  # Standard in Discriminators
            # 64 -> 128 channels
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # Flatten to output single probability value
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid()  # Output probability [0, 1]
        )

    def forward(self, img):
        return self.model(img)
```

---

## 6. PyTorch GAN Training Step
* **Goal**: Train G and D alternatively, freezing weights and detaching fakes.
* **The Code**:
```python
criterion = nn.BCELoss()
opt_D = torch.optim.Adam(D.parameters())
opt_G = torch.optim.Adam(G.parameters())

# ============ 1. Train Discriminator ============
# Freeze Generator weights implicitly by not optimizing G
# Feed Real Images (target label: 1)
real_output = D(real_imgs)
real_loss = criterion(real_output, torch.ones(batch_size, 1, device=device))

# Feed Fake Images (target label: 0)
z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
fake_imgs = G(z)
# DETACH fake images to prevent gradients from updating G!
fake_output = D(fake_imgs.detach())
fake_loss = criterion(fake_output, torch.zeros(batch_size, 1, device=device))

d_loss = real_loss + fake_loss
opt_D.zero_grad()
d_loss.backward()
opt_D.step()

# ============ 2. Train Generator ============
# Freeze Discriminator weights implicitly by not optimizing D
# Feed new Fake Images to D (target label: 1 to FOOL D!)
z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
fake_imgs = G(z)
output = D(fake_imgs) # No detach here! We want gradients to flow to G.
g_loss = criterion(output, torch.ones(batch_size, 1, device=device))

opt_G.zero_grad()
g_loss.backward()
opt_G.step()
```

---

## 7. PyTorch CGAN label injection (Generator & Discriminator)
* **Goal**: Inject class labels as embeddings in both PyTorch networks.
* **The Code**:
```python
# ============ Generator Forward (Concatenate in 4D space) ============
# In G.__init__: self.label_emb = nn.Embedding(num_classes, embedding_dim)
def generator_forward(self, z, labels):
    # z shape: (batch, latent_dim, 1, 1) | labels shape: (batch)
    
    # 1. Embed class integer to vector
    label_embedding = self.label_emb(labels)  # (batch, embedding_dim)
    # 2. Reshape to 4D to match spatial dims of z
    label_embedding = label_embedding.unsqueeze(2).unsqueeze(3)  # (batch, embedding_dim, 1, 1)
    # 3. Concatenate along channels (dim=1)
    x = torch.cat([z, label_embedding], dim=1)  # (batch, latent_dim + embedding_dim, 1, 1)
    
    return self.model(x)

# ============ Discriminator Forward (Concatenate as image channel) ============
# In D.__init__: self.label_emb = nn.Embedding(num_classes, 28 * 28)
def discriminator_forward(self, img, labels):
    # img shape: (batch, 1, 28, 28) | labels shape: (batch)
    
    # 1. Embed class integer to a flattened image vector
    label_embedding = self.label_emb(labels)  # (batch, 784)
    # 2. Reshape to match the spatial image layout
    label_embedding = label_embedding.view(labels.size(0), 1, 28, 28)  # (batch, 1, 28, 28)
    # 3. Concatenate along channels (dim=1) as a second channel layer
    x = torch.cat([img, label_embedding], dim=1)  # (batch, 2, 28, 28)
    
    return self.model(x)
```

---

## 8. PyTorch Forward Diffusion Noise Injection
* **Goal**: Add noise directly to an image at any timestep $t$ using the cumulative schedule $\hat{\alpha}_t$.
* **The Code**:
```python
import torch

# Formula: noisy_x = sqrt(alpha_hat) * x + sqrt(1 - alpha_hat) * epsilon
def add_noise(x, t, alpha_hat_schedule):
    # 1. Extract cumulative alpha for timestep t
    alpha_t = alpha_hat_schedule[t]
    
    # 2. Reshape to match 4D image dimensions: (batch, channels, H, W)
    alpha_t = alpha_t.view(-1, 1, 1, 1)
    
    # 3. Calculate coefficients
    sqrt_alpha = torch.sqrt(alpha_t)
    sqrt_one_minus_alpha = torch.sqrt(1 - alpha_t)
    
    # 4. Generate random noise epsilon
    epsilon = torch.randn_like(x)
    
    # 5. Apply the linear combination
    noisy_x = sqrt_alpha * x + sqrt_one_minus_alpha * epsilon
    
    return noisy_x, epsilon
```

---
---

# 📂 Section 2: Code Blocks from Sheet.pdf

## 9. VAE Latent Dimension Odd/Even Shift (Sheet Page 5)
* **Goal**: Generate random vectors, shift even dimensions down and odd dimensions up by `val = 1.3`, and decode both.
* **The Code**:
```python
import numpy as np

n = 10
z_random = np.random.normal(size=(n, latent_dim))

# Initialize empty latent matrix of same size
ZZ = np.zeros((n, latent_dim))
val = 1.3

for j in range(n):
    for k in range(latent_dim):
        if k % 2 == 0:
            ZZ[j, k] = z_random[j, k] - val  # Even dimensions: subtract val
        else:
            ZZ[j, k] = z_random[j, k] + val  # Odd dimensions: add val

# Decode both original and modified
genImg = decoder.predict(z_random)
genImg2 = decoder.predict(ZZ)
```

---

## 10. GAN Latent Walk Morphing Loop (Sheet Page 6)
* **Goal**: Generate $z_1, z_2$ in PyTorch, sum them, modify `.data` elements directly, decode, and normalize output to $[0,1]$.
* **The Code**:
```python
import torch

G.eval()

# 1. Define two random noise vectors requiring gradient tracking
z1 = torch.randn(1, latent_dim, 1, 1, device=device, requires_grad=True)
z2 = torch.randn(1, latent_dim, 1, 1, device=device, requires_grad=True)

# 2. Generate and normalize to [0, 1] range (division by 2, shift by 0.5)
gen1 = ((G(z1).detach().cpu() + 1) / 2)
gen2 = ((G(z2).detach().cpu() + 1) / 2)

num_steps = 20
for step in range(num_steps):
    # Sum the latents
    z = z1 + z2
    generated = G(z).detach().cpu()
    
    # Normalize pixel values
    recon = ((generated + 1) / 2)
    
    # Modify dimension index 'i' directly on .data to bypass autograd graph tracking!
    z1.data[0, i, 0, 0] += 0.3
    z2.data[0, i, 0, 0] += 0.3
```

---
---

# 📂 Section 3: Code Blocks from Past Exams

## 11. Word2Vec Quadrant Weight Filtering (Final 2024)
* **Goal**: Find all vectors in `weights` that are above the target vector `i` and lie in the same quadrant.
* **The Code**:
```python
import numpy as np

weights = model.get_weights()[0]

def find_above_same_quarter(i):
    L = []
    # 1. Get signs (quadrant) of target vector
    x_i, y_i = weights[i][0], weights[i][1]
    sign_x_i = np.sign(x_i)
    sign_y_i = np.sign(y_i)

    # 2. Iterate through all weights
    for j in range(weights.shape[0]):
        x_j, y_j = weights[j][0], weights[j][1]

        # 3. Check quadrant matching
        if np.sign(x_j) == sign_x_i and np.sign(y_j) == sign_y_i:
            # Check if y-coordinate is greater (above)
            if y_j > y_i:
                L.append(weights[j])
    return L
```

---

## 12. Autoencoder Latent Space Walking Search Loop (Final 2024)
* **Goal**: Interpolate between each unique pair of latent vectors using variable steps and decode to match a target image.
* **The Code**:
```python
import numpy as np

Results = encoder.predict(x_test)
those = []

for i in range(len(Results)):
    for k in range(i + 1, len(Results)):  # i+1 avoids duplicates and self-pairs
        
        # Try different step counts (e.g., 10 to 20 steps)
        for steps in range(10, 21):
            # 1. Interpolate line between Results[i] and Results[k]
            interpolated = np.linspace(Results[i], Results[k], steps)
            
            # 2. Decode the whole array of steps at once
            decoded_imgs = decoder.predict(interpolated)
            
            # 3. Check distance to target (Targt) using Dist
            for a in range(steps):
                if Dist(decoded_imgs[a], Targt) <= threshold:
                    those.append(decoded_imgs[a])
                    break  # Found match, break out of steps loop for this pair
```

---

## 13. VAE Image Flipping & Mean Modification Search (Final 2024)
* **Goal**: Flip target image, modify positive means in training set to be negative, sample via reparameterization trick, decode, and find best match.
* **The Code**:
```python
import numpy as np

# 1. Flip Target Image vertically (axis=0) and flatten
flipped_target = np.flip(Trgt.reshape(28, 28), axis=0).flatten()

# 2. Modify training means (all_mus) - make positive values negative
modified_mus = all_mus.copy()
modified_mus[modified_mus > 0] = -modified_mus[modified_mus > 0]

best_dist = float('inf')
best_i = -1
best_noise = None

# 3. Search Loop
for i in range(len(x_train)):
    mu_mod = modified_mus[i:i+1]
    
    # Get log-variance of image i from encoder
    _, log_var_i = encoder.predict(x_train[i:i+1])
    sigma_i = np.exp(0.5 * log_var_i)

    # Sample 5 times with random noise
    for _ in range(5):
        noise = np.random.normal(size=mu_mod.shape)
        
        # Reparameterization: z = mu + sigma * epsilon
        z = mu_mod + sigma_i * noise
        
        # Decode and compute distance
        DCode = decoder.predict(z)
        distance = Dist(DCode.flatten(), flipped_target)
        
        if distance < best_dist:
            best_dist = distance
            best_i = i
            best_noise = noise
```
