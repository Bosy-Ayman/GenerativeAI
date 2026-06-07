# 🧠 Code Memorization Guide — Generative AI (AI442)

This guide provides simple **mental models (طرق ربط ذهني)**, **mnemonics**, and **simplified code skeletons** to help you write all core code sequences from memory under exam pressure.

---

## 🗺️ The Map of 5 Code Blocks You Must Memorize

```
1. Word2Vec Filtering  ──▶  2. AE Latent Walk  ──▶  3. VAE Custom Layer  ──▶  4. CGAN Forward  ──▶  5. VAE Search Loop
(Find words above index)    (linspace morphing)       (Sampling z = μ + σε)      (Embedding cat)       (mu/noise search)
```

---

## 🔑 Block 1: Word2Vec Weight Filtering

### 💡 The Mental Model:
The embedding matrix `weights = model.get_weights()[0]` is a 2D array of shape `(vocab_size, 2)`. 
* `weights[j][0]` is the **X coordinate**.
* `weights[j][1]` is the **Y coordinate** (which determines if a vector is "above" another).

### 📝 The Code Sequence:
```python
weights = model.get_weights()[0]

# Rule 1: Find all words ABOVE given i-th word
def find_above(i):
    L = []
    for j in range(weights.shape[0]):
        if weights[j][1] > weights[i][1]:  # Compare Y-coordinates
            L.append(weights[j])
    return L

# Rule 2: Find all words ABOVE and in the SAME QUARTER (Quadrant)
def find_above_same_quarter(i):
    L = []
    x_i, y_i = weights[i][0], weights[i][1]
    
    for j in range(weights.shape[0]):
        x_j, y_j = weights[j][0], weights[j][1]
        
        # Check quadrant symmetry (same signs!)
        if np.sign(x_j) == np.sign(x_i) and np.sign(y_j) == np.sign(y_i):
            if y_j > y_i:  # Check if above
                L.append(weights[j])
    return L
```

### 🧠 Memorization Trick (بالعربي):
* **فوق الكلمة $i$** = بنقارن الـ Coordinate التاني (y-coordinate) اللي هو `[1]`.
* **في نفس الربع (Quadrant)** = إشاراتهم متطابقة. بنستخدم `np.sign` عشان نتأكد إن إشارة $x$ وإشارة $y$ متطابقين للكلمتين.

---

## 🔑 Block 2: Autoencoder Latent Space Walking

### 💡 The Mental Model:
Take every possible pair of latent vectors $(Z_i, Z_k)$, create a straight line of $N$ steps between them, decode each step, and check distance to a target.

### 📝 The Code Sequence:
```python
Zs = encoder.predict(x_test)
those = []

# 1. Nest loops to pair every image with every other image
for i in range(len(Zs)):
    for k in range(i + 1, len(Zs)):  # i+1 avoids duplicates and self-pairs
        
        # 2. Try steps (e.g., variable steps 12 to 18)
        for steps in range(12, 19):
            # 3. Walk the line in latent space
            interpolated = np.linspace(Zs[i], Zs[k], steps)
            
            # 4. Decode all steps at once
            decoded = decoder.predict(interpolated)
            
            # 5. Check distance to target (Targt)
            for a in range(steps):
                if Dist(decoded[a], Targt) <= threshold:
                    those.append(decoded[a])
                    break # pair matched, stop this walk
```

### 🧠 Memorization Trick (بالعربي):
* **linspace (الخط المتصل)**: `np.linspace(A, B, steps)` هو اللي بيخلق النقط اللي في النص.
* **Predict & Loop**: الأسرع نمرر الـ `interpolated` بالكامل للـ `decoder.predict` مرة واحدة، وبعدين نعمل loop نقيس المسافات بالـ `Dist`.

---

## 🔑 Block 3: VAE Reparameterization & KL Loss

### 💡 The Mental Model:
To compile and train a VAE, you need to define how to sample $z$ using log-variance and compute the KL divergence loss.

### 📝 The Code Sequence:

#### **1. VAE Sampling Layer:**
$$z = \mu + \sigma \cdot \epsilon \implies z = \mu + \exp(0.5 \cdot \log(\sigma^2)) \cdot \epsilon$$
```python
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))  # Random ε ~ N(0,1)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon    # μ + exp(0.5 * log_var) * ε
```

#### **2. KL Divergence Loss in VAE:**
$$D_{KL} = -0.5 \sum \left(1 + \log(\sigma^2) - \mu^2 - \sigma^2\right)$$
```python
# In TensorFlow/Keras:
kl_loss = -0.5 * tf.reduce_sum(
    1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1
)
```

### 🧠 Memorization Trick (بالعربي):
* **المعادلة الذهبية للـ Sampling**: `mu + exp(0.5 * log_var) * epsilon`.
  * *ليه 0.5؟* لأن $e^{0.5 \cdot \log(\sigma^2)} = e^{\log(\sigma)} = \sigma$.
* **ترتيب معادلة الـ KL**: `1 + log_var - mu^2 - exp(log_var)`.
  * *روابط الحفظ*: تبدأ بـ `1` زائد الـ `log_var` ناقص مربع المتوسط `mu` ناقص الـ `variance` الأصلية اللي هي `exp(log_var)`. وكل ده مضروب في `-0.5`.

---

## 🔑 Block 4: CGAN Conditional Embedding & Injection

### 💡 The Mental Model:
In PyTorch, class labels are integers. We convert them to embeddings. 
* **Generator**: Label embedding is small (e.g. 10). Concat with 4D noise vector $z$ (`batch, latent_dim, 1, 1`).
* **Discriminator**: Label embedding is spatial size (e.g. 28*28). Concat with 4D image tensor (`batch, 1, 28, 28`).

### 📝 The Code Sequence:

#### **1. CGAN Generator Forward:**
```python
def forward(self, z, labels):
    # labels shape: (batch) | z shape: (batch, latent_dim, 1, 1)
    
    # 1. Embed and unsqueeze twice to reach 4D
    label_emb = self.label_emb(labels)              # (batch, emb_dim)
    label_emb = label_emb.unsqueeze(2).unsqueeze(3)  # (batch, emb_dim, 1, 1)
    
    # 2. Concatenate along channel dimension (dim=1)
    x = torch.cat([z, label_emb], dim=1)            # (batch, latent_dim + emb_dim, 1, 1)
    
    # 3. Pass through ConvTranspose layers
    return self.model(x)
```

#### **2. CGAN Discriminator Forward:**
```python
def forward(self, img, labels):
    # labels shape: (batch) | img shape: (batch, 1, 28, 28)
    
    # 1. Embed and reshape to image spatial dimensions
    label_emb = self.label_emb(labels)              # (batch, 784)
    label_emb = label_emb.view(labels.size(0), 1, 28, 28)  # (batch, 1, 28, 28)
    
    # 2. Stack as second channel (dim=1)
    x = torch.cat([img, label_emb], dim=1)          # (batch, 2, 28, 28)
    
    # 3. Pass through Conv layers
    return self.model(x)
```

### 🧠 Memorization Trick (بالعربي):
* **Concatenation Channel**: الإضافة دايماً بتتلزق في الـ channel dimension اللي هو `dim=1` في PyTorch.
* **Generator**: بنلزق Embedding صغير في الـ $z$ بعد ما نعمل له `unsqueeze` مرتين.
* **Discriminator**: بنلزق Embedding بحجم الصورة بالكامل (`28*28`) في الصورة بعد ما نغير أبعاد الـ label لـ `(batch, 1, 28, 28)` عشان يبقى stacked channel.

---

## 🔑 Block 5: VAE Target Reconstruction Search Loop

### 💡 The Mental Model:
Given a target image, crop/flip it. Then loop over the training set, encode each image, apply the modified mean logic, sample $z$ using the reparameterization formula, decode it, and find the minimum distance.

### 📝 The Code Sequence:
```python
# 1. Prep target (e.g., flip vertically)
flipped_target = np.flip(Trgt.reshape(28, 28), axis=0).flatten()

# 2. Search loop
best_dist = float('inf')
best_i = -1
best_noise = None

for i in range(len(x_train)):
    # Encode current training image i
    mu, log_var = encoder.predict(x_train[i:i+1])
    
    # Modify means: e.g., shift positive values to negative
    mu_mod = mu.copy()
    mu_mod[mu_mod > 0] = -mu_mod[mu_mod > 0]
    
    # Try random noise samples
    for _ in range(3):
        noise = np.random.normal(size=mu.shape)
        sigma = np.exp(0.5 * log_var)
        
        # Reparameterize: z = mu_mod + sigma * noise
        z = mu_mod + sigma * noise
        
        # Decode and compare distance
        decoded = decoder.predict(z)
        distance = Dist(decoded.flatten(), flipped_target)
        
        if distance < best_dist:
            best_dist = distance
            best_i = i
            best_noise = noise
```

### 🧠 Memorization Trick (بالعربي):
* **خطوات البحث**:
  1. جهّز الهدف (اقلب الصورة بالـ `np.flip`).
  2. اعمل loop على صور التدريب.
  3. استخرج الـ `mu` والـ `log_var` بالـ `encoder.predict`.
  4. عدّل الـ `mu` بناء على كلام السؤال.
  5. اسحب تشويش عشوائي `noise` وطبق قانون الـ Sampling: `z = mu_mod + sigma * noise`.
  6. فك التشفير بالـ `decoder.predict` وقارن المسافة بالـ `Dist`.

---

## 🔑 Block 6: GAN Training Step & Detaching

### 💡 The Mental Model:
* **Train Discriminator**: Freeze Generator. Label real images as `1`, fake images as `0`. Compute losses and update D.
* **Train Generator**: Freeze Discriminator. Pass fake images to D, but target `1` (fool D). Update G.

### 📝 The Code Sequence:
```python
# ================= Train Discriminator =================
z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
fake_imgs = G(z)  # Generate fakes

real_output = D(real_imgs)
fake_output = D(fake_imgs.detach())  # Detach prevents backprop to G!

d_loss = criterion(real_output, real_labels) + criterion(fake_output, fake_labels)
# Update D weights (opt_D.step())

# ================= Train Generator =================
z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
fake_imgs = G(z)
output = D(fake_imgs)  # No detach here, we want gradients to G!

g_loss = criterion(output, real_labels)  # Target is real_labels (1) to fool D!
# Update G weights (opt_G.step())
```

### 🧠 Memorization Trick (بالعربي):
* **Discriminator**: بنستخدم `fake_imgs.detach()` عشان نوقف الـ gradients للـ Generator. والـ labels هي `real_labels` للحقيقي و `fake_labels` للمزيف.
* **Generator**: مش بنستخدم detach، وبنقارن مخرجات الـ Discriminator للصور المزيفة بـ `real_labels` (الهدف 1) لأن المولد عايز يخدع المميّز.
* **صور الـ GAN** دايماً بنعمل ليها Normalize لـ `[-1, 1]` عشان الـ Generator بينتهي بـ `Tanh`. لما بنعرضها بنرجعها لـ `[0, 1]` بالقانون: `(img + 1) / 2`.
