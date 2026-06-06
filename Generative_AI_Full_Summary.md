# 📚 Generative AI — Full Course Summary for Final Exam

> **ملخص شامل لمادة الذكاء الاصطناعي التوليدي — جاهز للمراجعة النهائية**
>
> This summary connects **all lectures** into one coherent story. Each concept builds on the previous one. Arabic explanations (بالعربي) and real-life examples are included throughout.

---

## 🗺️ The Big Picture — How Everything Connects

```
┌─────────────┐     ┌───────────────────┐     ┌────────────────┐     ┌──────────────┐
│  Word2Vec   │────▶│   AutoEncoders    │────▶│      VAE       │────▶│  Diffusion   │
│ (Lecture 2) │     │ (Lectures 3-5)    │     │ (Lectures 6-9) │     │ (Lecture 13) │
│             │     │                   │     │                │     │              │
│ "Represent  │     │ "Compress &       │     │ "Compress &    │     │ "Add noise   │
│  words as   │     │  reconstruct      │     │  GENERATE new  │     │  then learn  │
│  vectors"   │     │  data"            │     │  data"         │     │  to remove   │
│             │     │                   │     │                │     │  it"         │
└─────────────┘     └───────────────────┘     └────────────────┘     └──────────────┘
```

**The journey / الرحلة:**
1. **Word2Vec**: How to represent words as small, meaningful vectors (embeddings) — تمثيل الكلمات كأرقام
2. **AutoEncoders**: How to compress ANY data (images) into a small vector and reconstruct it — ضغط البيانات وإعادة بناءها
3. **VAE**: How to make that compressed space *organized* so we can **generate NEW data** — توليد بيانات جديدة
4. **Diffusion Models**: A completely different approach — destroy data with noise, then learn to reverse the destruction — تدمير البيانات بالتشويش ثم تعلم إعادة بنائها

---

# 📖 Part 1: Word2Vec (Lecture 2a & 2b)

## 1.1 What is the Problem?

Computers don't understand words — they only understand numbers. We need to convert words to numbers.
> **بالعربي:** الكمبيوتر مش فاهم كلمات، فلازم نحول الكلمات لأرقام

### Old Approaches (What Word2Vec Replaces):

| Method | How It Works | Problem |
|--------|-------------|---------|
| **One-Hot Encoding** | Each word = a vector of all zeros except one 1 | Vectors are HUGE & no similarity info (كل الكلمات بعيدة عن بعض بنفس المسافة) |
| **Bag of Words** | Count how many times each word appears | Loses word order (مفيش ترتيب) |
| **TF-IDF** | Count words + reduce importance of common ones | Still no semantic meaning |

### 🏠 Real-Life Analogy:
> Imagine you have 10,000 students. One-hot encoding is like giving each student a locker number — Student #5432 tells you nothing about the student. Word2Vec is like describing each student with 2-3 features (height, GPA, age) — now similar students have similar descriptions!
>
> **بالعربي:** الـ One-Hot زي رقم اللوكر — مجرد رقم مالوش معنى. لكن Word2Vec زي ما توصف كل طالب بطوله ودرجاته — الطلاب المتشابهين هيكون وصفهم قريب من بعض

## 1.2 Word2Vec Core Idea

**Distributional Hypothesis**: "A word is known by the company it keeps" — الكلمة تُعرف من الكلمات اللي حواليها

- "The **prince** is a strong **man**"
- "Only a **man** can be a **king**"
- → `prince` and `king` and `man` appear in similar contexts → they should have similar vectors!

## 1.3 Word2Vec Pipeline

### Step 1: Preprocessing (تحضير البيانات)
```
Raw Text → Remove stop words (a, the, is, be...) → Remove punctuation → Lowercase → Tokenize
```

### Step 2: Build Word Pairs with Context Window (بناء أزواج الكلمات)

With **window = 2**, for each word, pair it with the 2 words before and 2 words after:

```
Sentence: "future king prince"
Window = 2:
    ['future', 'king']   → future appears near king
    ['future', 'prince'] → future appears near prince
    ['king', 'future']   → king appears near future
    ['king', 'prince']   → king appears near prince
    ...
```

> **بالعربي:** بناخد كل كلمة ونشوف الكلمات اللي جنبها (في نطاق 2 كلمات) ونعمل أزواج. ده بيقول للموديل "الكلمات دي بتيجي جنب بعض كتير"

### Step 3: Create Dictionary (إنشاء القاموس)
Each unique word gets an index:
```python
{'beautiful': 0, 'boy': 1, 'can': 2, ..., 'king': 7, ..., 'queen': 13, ...}
```

### Step 4: One-Hot Encode (ترميز One-Hot)
Convert each word pair to one-hot vectors:
```
['future', 'king'] → X: [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]  (index 6)
                     Y: [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]  (index 7)
```

### Step 5: Training — The Neural Network (تدريب الشبكة)
```
Input [1 × 21]  →  Hidden Layer [2 neurons]  →  Output [21 neurons + softmax]
(one-hot)          (THIS IS THE EMBEDDING!)       (predict context word)
```

**The magic**: The hidden layer with only **2 neurons** forces the network to compress 21 dimensions into 2. After training, these 2 numbers ARE the word embedding!

> **بالعربي:** الشبكة فيها طبقة وسطى صغيرة (2 خلايا بس). الشبكة مجبورة تضغط كل المعلومات في رقمين. الرقمين دول هما الـ embedding بتاع الكلمة

```python
# Architecture
inp = Input(shape=(21,))                              # 21 unique words
x = Dense(units=2, activation='linear')(inp)           # Embedding size = 2
x = Dense(units=21, activation='softmax')(x)           # Predict context word
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(x=XX, y=YY, batch_size=10, epochs=100)
```

### Step 6: Extract Embeddings (استخراج الـ Embeddings)
```python
weights = model.get_weights()[0]  # Shape: (21, 2) — each word → 2D vector
# Now 'king' = [0.8, 1.2], 'queen' = [0.7, 1.1], etc.
```

### Step 7: Visualize — Plot words in 2D
After training, you can plot words on a 2D chart. Words with similar meanings will be close together!

> **النتيجة:** king و queen قريبين من بعض، prince و princess قريبين من بعض، man و woman قريبين من بعض

## 1.4 Key Takeaway
Word2Vec trains an autoencoder-like network: `word → small vector → predict context word`. The small vector is the **embedding** — a meaningful numeric representation of the word.

> This is the **SAME IDEA** as autoencoders (next topic) — compress to a small representation, then reconstruct!

---

# 📖 Part 2: AutoEncoders (Lectures 3, 4, 5)

## 2.1 What is an AutoEncoder? (ما هو الأوتو إنكودر؟)

An AutoEncoder is a neural network that learns to **compress data** into a small representation (encoding) and then **reconstruct** the original data from it.

```
Input (784 pixels) → [Encoder] → Latent Space (32 numbers) → [Decoder] → Output (784 pixels)
         x                              z (bottleneck)                         x̂
```

> **بالعربي:** الأوتو إنكودر هو شبكة عصبية بتتعلم تضغط الصورة لأرقام قليلة (الـ bottleneck) وبعدين تعيد بناء الصورة منهم. زي ما تاخد صورة 28×28 بكسل وتوصفها في 32 رقم بس

### 🏠 Real-Life Analogy:
> Imagine describing a person's face over the phone to a sketch artist. You can't send the actual photo (784 pixels), so you describe it: "round face, big eyes, small nose, dark hair" — just 4-5 features. The sketch artist reconstructs a face from your description.
> - **You** = Encoder (compressor)
> - **Your description** = Latent Space / Bottleneck
> - **Sketch artist** = Decoder (reconstructor)
>
> **بالعربي:** تخيل إنك بتوصف وش حد بالتليفون لرسام. مش هتبعت الصورة (784 بكسل)، هتقول "وشه مدور، عيونه كبيرة، مناخيره صغيرة". الـ 4-5 صفات دول هما الـ Latent Space. الرسام بيرسم الوش من الوصف = Decoder

**Key terms:**
- **Encoder (المُشفِّر)**: Compresses input → latent vector `z`
- **Decoder (فك التشفير)**: Reconstructs output from `z`
- **Latent Space (الفضاء الخفي)**: The compressed representation — the bottleneck
- **Reconstruction Loss**: How different is the output from the input? (`binary_crossentropy`)

## 2.2 Dense (Fully-Connected) AutoEncoder (Lecture 3)

### Architecture:
```
Input (784) → Dense(100, relu) → Dense(32, relu) → Dense(100, relu) → Dense(784, sigmoid)
                  ENCODER              LATENT           DECODER
```

### Implementation:
```python
# Complete AutoEncoder
encoding_dim = 32
input_img = keras.Input(shape=(784,))
x = layers.Dense(100, activation='relu')(input_img)
encoded = layers.Dense(encoding_dim, activation='relu')(x)
x = layers.Dense(100, activation='relu')(encoded)
decoded = layers.Dense(784, activation='sigmoid')(x)

autoencoder = keras.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
```

### Extracting Encoder and Decoder as Separate Models:
```python
# Encoder: input → encoded
encoder = keras.Model(input_img, encoded)

# Decoder: separate model (needs weight transfer!)
encoded_input = keras.Input(shape=(encoding_dim,))
a = layers.Dense(100, activation='relu')(encoded_input)
decoded = layers.Dense(784, activation='sigmoid')(a)
decoder = keras.Model(encoded_input, decoded)
```

### ⚠️ Important Note on Weight Transfer (نقل الأوزان):
When you create a **separate decoder model**, you must **copy weights** from the autoencoder:
```python
k = 0
for i in range(4, 8):  # Decoder layers are at indices 4-7
    decoder.weights[k].assign(autoencoder.weights[i])
    k += 1
```

> **بالعربي:** لما تعمل موديل decoder لوحده، لازم تنقل الأوزان من الأوتو إنكودر الأصلي. لأن الـ decoder الجديد مدربش لوحده — هو بياخد الأوزان من الموديل الكبير

### Easier Way (Using Layer Reference):
```python
encoder = keras.Model(input_img, encoded)
decoder_layer = autoencoder.layers[-1]  # Get last layer reference
decoder = keras.Model(encoded_input, decoder_layer(encoded_input))
# No need to transfer weights — it's the SAME layer object!
```

### Training & Results:
```python
# Train only the autoencoder — encoder & decoder train automatically
autoencoder.fit(x_train, x_train, epochs=30, batch_size=256, 
                validation_data=(x_test, x_test))

# Predict
encoded_imgs = encoder.predict(x_test)    # Compress
decoded_imgs = decoder.predict(encoded_imgs)  # Reconstruct
```

### Modifying Encoded Images (تعديل الصور المشفرة):
You can modify values in the latent space to create variations:
```python
imgsToBe = encoded_imgs[0:10]
for i in range(10):
    imgsToBe[i, 0:16] -= random.random() * 5  # Modify first 16 dims
decoded_imgs = decoder.predict(imgsToBe)  # Generates modified images
```

> **بالعربي:** لو عدلت الأرقام في الـ Latent Space، هتتغير الصورة اللي الـ Decoder بينتجها. ده أول خطوة في اتجاه التوليد!

## 2.3 Convolutional AutoEncoder (Lecture 4)

For images, convolutions are better than dense layers because they understand **spatial structure** (nearby pixels are related).
> **بالعربي:** للصور، الـ Conv أحسن من Dense لأنه بيفهم العلاقة بين البكسلات اللي جنب بعض

### Key Building Blocks:

#### 1. Conv2D + MaxPooling (Encoder — shrinks spatial size)
```
28×28 → Conv2D(16) → MaxPool → 14×14 → Conv2D(8) → MaxPool → 7×7 → Conv2D(1) → MaxPool → 4×4
```

#### 2. UpSampling2D (Decoder — grows spatial size back)
Repeats each pixel to make the image bigger:
```python
X = [[1, 7],         UpSampling2D((2,2))     [[1, 1, 7, 7],
     [3, 2]]         ──────────────────▶       [1, 1, 7, 7],
                                               [3, 3, 2, 2],
                                               [3, 3, 2, 2]]
```
With `interpolation='bilinear'`, it smoothly interpolates instead of just repeating.

#### 3. Conv2DTranspose (Alternative to UpSampling — learned upscaling)
The formula for output size: **OutSize = (n − 1) × s + k** (where n = input size, s = stride, k = kernel size)

### Implementation — UpSampling Approach:
```python
# ENCODER
input_img = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2,2), padding='same')(x)
x = layers.Conv2D(8, (3,3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2,2), padding='same')(x)
x = layers.Conv2D(1, (3,3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2,2), padding='same')(x)

# DECODER
x = layers.Conv2D(8, (3,3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2,2))(x)
x = layers.Conv2D(8, (3,3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2,2))(x)
x = layers.Conv2D(16, (3,3), activation='relu')(x)
x = layers.UpSampling2D((2,2))(x)
decoded = layers.Conv2D(1, (3,3), activation='sigmoid', padding='same')(x)
```

### Implementation — Conv2DTranspose Approach:
```python
# ENCODER (same as above but fewer layers)
input_img = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2,2), padding='same')(x)
x = layers.Conv2D(8, (3,3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2,2), padding='same')(x)
encoded = x

# DECODER with Conv2DTranspose
x = layers.Conv2DTranspose(8, (3,3), strides=(2,2), activation='relu', padding='same')(encoded)
x = layers.Conv2DTranspose(8, (3,3), strides=(2,2), activation='relu', padding='same')(x)
decoded = layers.Conv2D(1, (3,3), activation='sigmoid', padding='same')(x)
```

### UpSampling vs Conv2DTranspose — Comparison:

| Feature | UpSampling2D | Conv2DTranspose |
|---------|-------------|-----------------|
| **How it works** | Repeats pixels (or bilinear interpolation) | Learned upscaling with weights |
| **Parameters** | No learnable parameters | Has learnable weights |
| **Quality** | Simpler, can cause artifacts | Generally better quality |
| **Speed** | Faster | Slower (more computation) |

> **بالعربي:** UpSampling بتكرر البكسلات ببساطة (بدون تعلم). Conv2DTranspose بتتعلم إزاي تكبر الصورة — عندها أوزان بتتدرب. عادةً Conv2DTranspose بتدي نتايج أحسن

## 2.4 Latent Space Walking (Lecture 5)

### The Concept
If we have two images encoded in the latent space as vectors A and B, we can **walk (interpolate)** between them to generate intermediate images.

> **بالعربي:** لو عندنا صورتين وضغطناهم في الـ Latent Space، نقدر نمشي بينهم خطوة خطوة ونشوف إيه الصور اللي في النص. ده بيبين إن الـ Latent Space فيه معلومات مستمرة

### 🏠 Real-Life Analogy:
> Imagine you have a description of a cat face (A = [round ears, small eyes]) and a dog face (B = [pointy ears, big eyes]). Walking between them means: step 1: [slightly pointy ears, slightly bigger eyes], step 2: [more pointy, bigger], etc. — you'd see faces morphing from cat to dog!

### Implementation:
```python
# Pick two test images
b = [9, 11]
imgs = x_test[b]
encoded_imgs = encoder.predict(imgs)  # Get their latent vectors

# Interpolate between them
interpolation_steps = 20
interpolated = np.linspace(encoded_imgs[0], encoded_imgs[1], interpolation_steps)
# np.linspace creates 20 evenly-spaced points between vector A and vector B

# Decode all interpolated vectors
decoded_imgs = decoder.predict(interpolated)
# Result: 20 images that smoothly morph from image A to image B
```

### ⚠️ Problem with Standard AutoEncoders for Generation:

The latent space of a standard autoencoder is **not organized** — there are "holes" and "dead zones" where no training data exists. If you sample a random point, you might get garbage!

> **بالعربي:** مشكلة الأوتو إنكودر العادي إن الـ Latent Space مش منظم. في مناطق فاضية ملهاش معنى. لو أخذت نقطة عشوائية ممكن تطلع صورة مش مفهومة. عشان كده محتاجين VAE!

**This is exactly why we need VAE** → it forces the latent space to be smooth and organized!

---

# 📖 Part 3: Variational AutoEncoder — VAE (Lectures 6, 7, 8, 9)

<iframe width="560" height="315" src="https://www.youtube.com/embed/qJeaCHQ1k2w?si=BiC9vFGUJfUSsQkA" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

## 3.1 Mathematical Foundations (Lecture 6) — Probability Review

Before understanding VAE, we need these probability concepts:

### Conditional Probability (الاحتمال الشرطي)
**P(A|B)** = probability of A given that B has already happened.

> **بالعربي:** احتمال حدوث A لو عارفين إن B حصل. مثلاً: احتمال إن شخص عنده سكر لو عارفين إنه بدين

### Joint Probability (الاحتمال المشترك)
**P(A, B)** = probability of A AND B happening together.
> **بالعربي:** احتمال حدوث A و B مع بعض. مثلاً: احتمال إن شخص بدين وعنده سكر

### Marginal Probability (الاحتمال الهامشي)
**P(A)** = probability of A regardless of anything else.
> **بالعربي:** احتمال A بغض النظر عن أي حاجة تانية. مثلاً: احتمال إن شخص عشوائي عنده سكر

### Bayes' Rule (قاعدة بايز)
$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

> **بالعربي:** لو عارف احتمال B لو A حصل، تقدر تحسب احتمال A لو B حصل

### Why Do We Need This for VAE?
In VAE, we want to find **P(z|x)** — given an image x, what is the distribution of latent codes z?

But this is **impossible to compute directly** (intractable). So we use an approximation **q(z|x)** and measure how close it is to the real P(z|x) using **KL Divergence**.

## 3.2 Expected Value (القيمة المتوقعة) — Lecture 7

**Expected Value = Long-run weighted average (المتوسط المرجح على المدى الطويل)**

$$E[f(X)] = \sum_{x} P(x) \cdot f(x)$$

### 🏠 Real-Life Example — Real Estate Investment:

| Market (X) | Probability P(X) | Profit/Loss f(X) |
|------------|-------------------|-------------------|
| Strong market (+1) | 30% | +$300,000 |
| Weak market (-1) | 70% | −$100,000 |

$$E[f(X)] = 0.3 \times 300{,}000 + 0.7 \times (-100{,}000) = 90{,}000 - 70{,}000 = \$20{,}000$$

> **بالعربي:** كل مليون دولار هتستثمره، المتوقع على المدى الطويل إنك تكسب 20,000 دولار. مش معناه إنك هتكسب كده كل مرة — ده المتوسط على مرات كتير

## 3.3 KL Divergence (Lecture 7) — قياس الفرق بين توزيعين

**Kullback-Leibler (KL) Divergence** measures how different one probability distribution P is from another Q.

$$D_{KL}(P \| Q) = \sum_{x} P(x) \cdot \ln\frac{P(x)}{Q(x)}$$

### 🏠 Real-Life Analogy:
> Imagine a weather forecast: P = actual weather (50% sunny, 30% rainy, 20% cloudy) and Q = your friend's guess (40% sunny, 40% rainy, 20% cloudy). KL Divergence tells you how wrong your friend's guess is.
>
> **بالعربي:** KL Divergence بيقيس قد إيه توزيع احتمالي Q مختلف عن توزيع P. لو P و Q نفس الشكل، KL = 0. كل ما يكونوا مختلفين أكتر، KL بيكبر

### Important Properties:
1. **KL ≥ 0** always (never negative)
2. **KL = 0** only when P = Q (identical distributions)
3. **KL is NOT symmetric**: D_KL(P‖Q) ≠ D_KL(Q‖P)

### Worked Example (from the Sheet):

| X | P(X) | Q(X) |
|---|------|------|
| 1 | 0.5 | 0.4 |
| 2 | 0.3 | 0.4 |
| 3 | 0.2 | 0.2 |

$$D_{KL}(P \| Q) = 0.5 \ln\frac{0.5}{0.4} + 0.3 \ln\frac{0.3}{0.4} + 0.2 \ln\frac{0.2}{0.2}$$

$$= 0.5(0.223) + 0.3(-0.288) + 0.2(0) = 0.1115 - 0.0864 + 0 ≈ 0.025$$

### Another Worked Example (Sheet Q2):

| x | P(x) | Q(x) |
|---|------|------|
| -1 | 2/6 ≈ 0.33 | 7/18 ≈ 0.38 |
| -0.5 | 2/6 ≈ 0.33 | 1/18 ≈ 0.05 |
| 0.5 | 1/6 ≈ 0.16 | 9/18 ≈ 0.5 |
| 1 | 1/6 ≈ 0.16 | 1/18 ≈ 0.05 |

**D_KL(P‖Q) = 0.58** and **D_KL(Q‖P) = 0.47** → Notice: NOT symmetric!

### Why KL Divergence Matters for VAE:
In VAE, we want the encoder's output distribution **q(z|x)** to be close to a standard normal distribution **N(0,1)**. KL Divergence measures this gap → we minimize it during training!

## 3.4 VAE — The Key Innovation (Lectures 6, 8)

### Problem with Standard AutoEncoder:
- Encoder outputs a **single fixed point** z for each input x
- The latent space has gaps/holes → random sampling gives garbage
- **Cannot generate new data reliably**

### VAE Solution:
Instead of encoding to a **single point**, encode to a **distribution** (mean μ and variance σ²):

```
Standard AE:  Image → Encoder → z (single point) → Decoder → Reconstructed Image
VAE:          Image → Encoder → (μ, σ²) → Sample z ~ N(μ, σ²) → Decoder → Reconstructed Image
```

> **بالعربي:**
> - الأوتو إنكودر العادي: الصورة → الإنكودر → نقطة واحدة z → الديكودر → الصورة
> - الـ VAE: الصورة → الإنكودر → (متوسط μ + تباين σ²) → نسحب عينة عشوائية z → الديكودر → الصورة
>
> الفرق إن الـ VAE مش بيقول "الصورة دي = النقطة 3.5"، بل بيقول "الصورة دي = توزيع حوالين 3.5 بتباين 0.8". ده بيخلي الفضاء الخفي سلس ومتصل

### 🏠 Real-Life Analogy:
> Standard AE is like saying "My house is at exactly GPS coordinates (30.0444, 31.2357)". VAE is like saying "My house is SOMEWHERE in the Zamalek neighborhood" (a distribution). This fuzzy encoding forces the network to make the entire Zamalek area represent similar houses → smooth, continuous latent space!

### VAE from the Sheet — Mickey & Donald Example:
```
q(z | x="Mickey")     = N(z; μ=1, σ=1)    → Mickey's peak at z=1
q(z | x="DonaldDuck") = N(z; μ=4, σ=1)    → Donald's peak at z=4
```

The normal distribution formula:
$$\mathcal{N}(z; \mu, \sigma) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(z-\mu)^2}{2\sigma^2}}$$

Computing values for Donald (μ=4, σ=1):
```
N(z=1; 4,1) = 0.004    ← Very unlikely (far from mean)
N(z=2; 4,1) = 0.053
N(z=3; 4,1) = 0.241
N(z=4; 4,1) = 0.398    ← Most likely (AT the mean) ← PEAK
N(z=5; 4,1) = 0.241
N(z=6; 4,1) = 0.053
```

> **بالعربي:** صورة ميكي بتتمثل بتوزيع طبيعي مركزه عند z=1. صورة دونالد بتتمثل بتوزيع طبيعي مركزه عند z=4. كل ما تبعد عن المركز، الاحتمال بيقل

## 3.5 VAE Loss Function (Lecture 8)

The VAE loss has **two parts**:

$$\mathcal{L}_{VAE} = \underbrace{\text{Reconstruction Loss}}_{\text{How well does it reconstruct?}} + \underbrace{D_{KL}(q(z|x) \| \mathcal{N}(0,1))}_{\text{How close is the latent distribution to N(0,1)?}}$$

### Part 1: Reconstruction Loss (خسارة إعادة البناء)
- Measures how similar the output is to the input
- Uses **binary cross-entropy** (for pixel values between 0-1)
- Forces the VAE to learn good representations

> **بالعربي:** الجزء ده بيضمن إن الصورة اللي طالعة شبه الصورة اللي داخلة. زي ما تقول للموديل "لازم تعرف تعيد بناء الصورة كويس"

### Part 2: KL Divergence Loss / Regularization Loss (خسارة التنظيم)
- Measures how far the encoder's distribution q(z|x) is from N(0,1)
- Forces the latent space to be organized, smooth, continuous
- Prevents the encoder from just memorizing (placing each image at a fixed distant point)

> **بالعربي:** الجزء ده بيجبر الإنكودر إن التوزيع بتاعه يكون قريب من التوزيع الطبيعي القياسي N(0,1). ده اللي بيخلي الـ Latent Space منظم وسلس — بدونه الموديل هيحفظ كل صورة في مكان بعيد عن التاني ومش هنقدر نولد حاجة جديدة

### KL Divergence for VAE (Closed-Form):
When q(z|x) = N(μ, σ²) and p(z) = N(0, 1):

$$D_{KL} = -\frac{1}{2} \sum_{j=1}^{J} \left(1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2\right)$$

In code (using log-variance for numerical stability):
```python
kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
```

### 🏠 Real-Life Analogy for the Two Losses:
> Imagine a student taking an exam:
> - **Reconstruction Loss** = "Did you get the right answer?" → Accuracy
> - **KL Loss** = "Did you show your work in an organized way?" → Organization
>
> Both matter! Getting right answers chaotically is bad. Being organized but wrong is also bad. You need both.

## 3.6 The Reparameterization Trick (Lecture 9) — حيلة إعادة المعاملة

### The Problem:
We need to **sample** z from N(μ, σ²), but sampling is a **random** operation that we can't backpropagate through!

### The Solution:
Instead of sampling z directly from N(μ, σ²), we:
1. Sample ε from N(0, 1) — standard normal
2. Compute: **z = μ + σ × ε**

This way, the randomness is in ε (which doesn't depend on model parameters), and z is a deterministic function of μ and σ (which we CAN backpropagate through).

```python
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))  # Random noise
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon   # z = μ + σ * ε
```

> **بالعربي:** مشكلة الـ sampling إنها عملية عشوائية ومش نقدر نعمل backpropagation من خلالها. الحل: بدل ما نسحب z من N(μ,σ²)، نسحب ε من N(0,1) ونحسب z = μ + σ×ε. كده العشوائية في ε بس، و z بيعتمد على μ و σ بشكل رياضي واضح نقدر نعمل backprop عليه

### 🏠 Real-Life Analogy:
> You want to pick a random point near your house. Instead of directly picking a random GPS coordinate (can't track how you did it), you:
> 1. Pick a random direction and distance (ε)
> 2. Start from your house (μ) and walk that direction/distance scaled by (σ)
> 3. Point = house_location + neighborhood_size × random_walk
>
> Now you can trace back: the point depends on house_location and neighborhood_size!

### Why log-variance instead of variance? (ليه بنستخدم log-variance؟)
- Variance σ² must always be **positive** (> 0)
- log(σ²) can be **any real number** (-∞ to +∞) — easier for neural networks to output
- To get σ from log_var: `σ = exp(0.5 * log_var)` because `exp(0.5 * log(σ²)) = exp(log(σ)) = σ`

## 3.7 VAE Implementation (Lecture 9)

### Complete Architecture:

```python
# ============ ENCODER ============
encoder_inputs = layers.Input(shape=(original_dim,))  # 784 for MNIST
x = layers.Dense(256, activation="relu")(encoder_inputs)
x = layers.Dense(128, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)         # Output μ
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)   # Output log(σ²)
z = Sampling()([z_mean, z_log_var])                          # Sample z = μ + σ*ε
encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
```

```python
# ============ DECODER ============
latent_inputs = layers.Input(shape=(latent_dim,))
x = layers.Dense(128, activation="relu")(latent_inputs)
x = layers.Dense(256, activation="relu")(x)
decoder_outputs = layers.Dense(original_dim, activation="sigmoid")(x)
decoder = Model(latent_inputs, decoder_outputs, name="decoder")
```

```python
# ============ VAE MODEL ============
class VAE(Model):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)

        # Reconstruction Loss
        reconstruction_loss = tf.keras.losses.binary_crossentropy(inputs, reconstructed)
        reconstruction_loss = reconstruction_loss * original_dim

        # KL Loss
        kl_loss = -0.5 * tf.reduce_sum(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1
        )

        # Total Loss
        total_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
        self.add_loss(total_loss)
        return reconstructed
```

### Training:
```python
vae = VAE(encoder, decoder)
vae.compile(optimizer="adam")

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train.reshape(-1, original_dim)  # Flatten 28x28 → 784
x_test = x_test.reshape(-1, original_dim)

vae.fit(x_train, x_train, epochs=30, batch_size=32,
        validation_data=(x_test, x_test))
```

### Generation — Creating NEW Images (الأهم! — توليد صور جديدة):
```python
# Method 1: Random Generation — Sample random z from N(0,1) and decode
n = 10
z_random = np.random.normal(size=(n, latent_dim))
generated = decoder.predict(z_random)
# Because KL loss forced the latent space to be N(0,1), random samples work!
```

```python
# Method 2: Reconstruction with Variation — Encode an image, then sample multiple z's
x = x_test[11:12]
z_mean, z_log_var, _ = encoder.predict(x)
n = 20
epsilon = np.random.normal(size=(n, latent_dim))
z_samples = z_mean + np.exp(0.5 * z_log_var) * epsilon
generated = decoder.predict(z_samples)
# Result: 20 slightly different versions of the same digit!
```

> **بالعربي:**
> - **طريقة 1:** ناخد أرقام عشوائية من N(0,1) وندخلها للـ Decoder → صور جديدة تماماً!
> - **طريقة 2:** ناخد صورة موجودة، نشفرها (نطلع μ و σ)، ونسحب 20 عينة مختلفة → 20 نسخة مختلفة شوية من نفس الرقم

### Sheet Exam Question — Modifying Z Values vs Distribution Parameters:
```python
# From the sheet:
z_random = np.random.normal(size=(n, latent_dim))  # Random z vectors
ZZ = np.zeros((n, latent_dim))
val = 1.3
for j in range(n):
    for k in range(latent_dim):
        if k % 2 == 0:
            ZZ[j,k] = z_random[j,k] - val   # Even dims: subtract
        else:
            ZZ[j,k] = z_random[j,k] + val   # Odd dims: add

genImg  = decoder.predict(z_random)   # Original images
genImg2 = decoder.predict(ZZ)         # Modified images
```

> ⚠️ **Important exam answer:** This modifies the **latent values (z)** directly, NOT the distribution parameters (μ, σ). We didn't touch μ or σ — we just shifted the z values after sampling.
>
> **بالعربي:** الكود بيعدل قيم z مباشرة (بيزود أو بينقص 1.3 من كل بُعد). ده مش بيأثر على μ أو σ — ده بيغير النقطة اللي الديكودر هيفك تشفيرها

---

# 📖 Part 4: Diffusion Models (Lecture 13)

## 4.1 The Big Idea

Diffusion models take a **completely different approach** from VAEs:

1. **Forward Process (تشويش)**: Gradually add noise to an image until it becomes pure random noise
2. **Reverse Process (إزالة التشويش)**: Train a neural network to reverse the noise — step by step, remove noise to recover the image

> **بالعربي:** الـ Diffusion Model بيشتغل بطريقة مختلفة تماماً. بياخد صورة ويضيف عليها تشويش تدريجياً لحد ما تبقى تشويش بالكامل. بعدين بيدرب شبكة عصبية تتعلم تشيل التشويش خطوة بخطوة. لما عايز يولد صورة جديدة، بيبدأ من تشويش عشوائي وبيشيل التشويش!

### 🏠 Real-Life Analogy:
> Imagine you have a beautiful sand painting. The forward process is like slowly blowing wind on it — each gust scrambles it more until it's just random sand. The reverse process is like watching that video in reverse — learning how to un-scramble sand into a painting. Once you learn this, you can take ANY random pile of sand and turn it into art!
>
> **بالعربي:** تخيل لوحة رملية جميلة. العملية الأمامية زي ما تنفخ عليها — كل نفخة بتشوهها أكتر لحد ما تبقى رمل عشوائي. العملية العكسية = تعلم إزاي ترجع الرمل لوحة. لما تتعلم ده، تقدر تاخد أي رمل عشوائي وتحوله للوحة!

## 4.2 Forward Process — Adding Noise (العملية الأمامية)

At each timestep t, add a bit of noise:

$$x_t = \sqrt{\alpha_t} \cdot x_{t-1} + \sqrt{1 - \alpha_t} \cdot \epsilon$$

Where:
- `x_{t-1}` = image at previous step
- `α_t` = how much of the original to keep (gets smaller over time)
- `ε` = random noise from N(0,1)
- `x_t` = noisier image

Using cumulative α (alpha_hat) to jump directly to any timestep:

$$x_t = \sqrt{\hat{\alpha}_t} \cdot x_0 + \sqrt{1 - \hat{\alpha}_t} \cdot \epsilon$$

Where $\hat{\alpha}_t = \prod_{i=1}^{t} \alpha_i$

```python
def add_noise(x, t, alpha_hat):
    alpha_t = alpha_hat[t]
    alpha_t = alpha_t.view(-1, 1, 1, 1)
    sqrt_alpha_t = torch.sqrt(alpha_t)
    sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
    noise = torch.randn_like(x)                               # ε ~ N(0,1)
    noisy_x = sqrt_alpha_t * x + sqrt_one_minus_alpha_t * noise  # The formula
    return noisy_x, noise
```

### Sheet Example — Tracing Forward Diffusion:

Given α₀=0.9, α₁=0.8, α₂=0.6, trace x₀ to x₃:

```
x₁ = √0.9 × x₀ + √(1-0.9) × ε₀  =  0.949 × x₀ + 0.316 × ε₀
x₂ = √0.8 × x₁ + √(1-0.8) × ε₁  =  0.894 × x₁ + 0.447 × ε₁
x₃ = √0.6 × x₂ + √(1-0.6) × ε₂  =  0.774 × x₂ + 0.632 × ε₂
```

> **بالعربي:** في كل خطوة، بناخد جزء من الصورة السابقة (الجذر التربيعي لألفا) ونضيف عليه تشويش (الجذر التربيعي لـ 1 ناقص ألفا ضرب تشويش عشوائي). كل ما ألفا تقل، الصورة بتتشوه أكتر

## 4.3 Training — Learning to Denoise (التدريب)

The model learns to **predict the noise** that was added, not to directly predict the clean image:

```python
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for imgs, _ in loader:
        imgs = imgs.to(device)
        
        # 1. Pick random timestep for each image
        t = torch.randint(0, timesteps, (imgs.size(0),), device=device)
        
        # 2. Add noise to get noisy image
        noisy_imgs, noise = add_noise(imgs, t)
        
        # 3. Model predicts the noise
        predicted_noise = model(noisy_imgs, t)
        
        # 4. Loss = how different is predicted noise from actual noise
        loss = criterion(predicted_noise, noise)
        
        # 5. Update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Key Points:
1. **Random timestep t** is chosen for each image — the model learns to denoise at ANY noise level
2. The model takes both the **noisy image** AND the **timestep t** as input (it needs to know HOW noisy the image is)
3. Loss = **MSE between predicted noise and actual noise** that was added
4. The model architecture is typically a **U-Net**

> **بالعربي:**
> - بنختار timestep عشوائي لكل صورة
> - بنضيف تشويش للصورة
> - الموديل بيحاول يتوقع التشويش اللي اتضاف
> - الـ Loss = الفرق بين التشويش الحقيقي والتشويش اللي الموديل توقعه
> - الموديل بياخد الصورة المشوشة + رقم الـ timestep كمدخلات (لازم يعرف قد إيه الصورة متشوشة)

## 4.4 Generation — Creating New Images (التوليد)

To generate a new image:
1. Start with pure random noise x_T ~ N(0, 1)
2. For t = T, T-1, ..., 1, 0:
   - Predict the noise in the current image
   - Remove some of that noise
   - Result is a slightly cleaner image
3. Final result x_0 is the generated image!

## 4.5 Using Pre-trained Diffusion Models

```python
import torch
from diffusers import DDPMPipeline

model_id = "google/ddpm-cifar10-32"
pipe = DDPMPipeline.from_pretrained(model_id)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# Set random seed
generator = torch.Generator(device=pipe.device).manual_seed(0)

# Generate image from pure noise
image = pipe(
    batch_size=1,
    generator=generator,
    num_inference_steps=100    # 100 denoising steps
).images[0]
image.save("image.png")
# Takes about ~1 minute with pre-trained model
```

---

# 🔗 How Everything Connects — The Full Story

## The Evolution of Generative Models:

```
Word2Vec           →  AutoEncoder        →  VAE                →  Diffusion
(words → vectors)     (images → vectors     (images → organized   (noise → images)
                       → images)             distributions
                                             → NEW images)
```

| Concept | Word2Vec | AutoEncoder | VAE | Diffusion |
|---------|----------|-------------|-----|-----------|
| **Goal** | Meaningful word vectors | Compress & reconstruct | Compress & GENERATE | Generate from noise |
| **Input** | Word pairs | Images | Images | Images + noise level |
| **Bottleneck** | Embedding layer (2D) | Latent vector z | Distribution (μ, σ²) | No bottleneck |
| **Loss** | Cross-entropy | Reconstruction only | Reconstruction + KL | MSE (noise prediction) |
| **Can Generate?** | No (not designed for it) | Poorly (holes in latent space) | Yes! (smooth latent space) | Yes! (high quality) |
| **Framework** | TensorFlow/Keras | TensorFlow/Keras | TensorFlow/Keras | PyTorch |

## Key Connections:

### 1. Encoding ↔ Compression
- Word2Vec: 21-dim one-hot → 2-dim embedding (compression of word identity)
- AutoEncoder: 784-dim image → 32-dim latent (compression of visual features)
- VAE: 784-dim image → distribution over latent space (probabilistic compression)

### 2. The "Bottleneck" Principle
All these models use the same core idea: force information through a narrow channel to learn the most important features.

> **بالعربي:** كل النماذج دي بتستخدم نفس الفكرة الأساسية: اجبر المعلومات تمر من قناة ضيقة عشان تتعلم أهم الخصائص

### 3. From Reconstruction → Generation
- AE: Can reconstruct, but can't generate well (latent space has holes)
- VAE: Adds KL loss → smooth latent space → CAN generate by sampling from N(0,1)
- Diffusion: Different approach entirely — learns to reverse noise → generates by denoising random noise

### 4. The Role of Randomness
- AE: **No randomness** — deterministic encoding
- VAE: **Controlled randomness** — reparameterization trick (z = μ + σε)
- Diffusion: **Structured randomness** — noise schedule (α values)

---

# 📝 Exam Quick Reference — الأشياء المهمة للامتحان

## Must-Know Formulas:

### 1. Normal Distribution
$$\mathcal{N}(z; \mu, \sigma) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(z-\mu)^2}{2\sigma^2}}$$

### 2. KL Divergence
$$D_{KL}(P \| Q) = \sum_{x} P(x) \ln\frac{P(x)}{Q(x)}$$

### 3. KL for VAE (Closed-Form)
$$D_{KL} = -\frac{1}{2} \sum (1 + \log\sigma^2 - \mu^2 - \sigma^2)$$

### 4. VAE Loss
$$\mathcal{L} = \text{Reconstruction Loss} + D_{KL}(q(z|x) \| \mathcal{N}(0,1))$$

### 5. Reparameterization Trick
$$z = \mu + \sigma \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, 1)$$

### 6. Forward Diffusion
$$x_t = \sqrt{\alpha_t} \cdot x_0 + \sqrt{1-\alpha_t} \cdot \epsilon$$

### 7. Conv2DTranspose Output Size
$$\text{OutSize} = (n-1) \times s + k$$

## Must-Know Code Patterns:

### Sampling Layer (VAE)
```python
z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
```

### KL Loss (VAE)
```python
kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
```

### Forward Diffusion
```python
noisy_x = sqrt_alpha_t * x + sqrt_one_minus_alpha_t * noise
```

### Generating from VAE
```python
z_random = np.random.normal(size=(n, latent_dim))  # Sample from N(0,1)
generated = decoder.predict(z_random)                # Decode to image
```

## Common Exam Traps:

1. **KL Divergence is NOT symmetric**: D_KL(P‖Q) ≠ D_KL(Q‖P)
2. **Modifying z values ≠ modifying distribution parameters (μ, σ)**
3. **The decoder in a separate model needs weight transfer** (unless using layer reference)
4. **VAE uses log-variance** (not variance) for numerical stability
5. **Diffusion models predict NOISE**, not the clean image
6. **In diffusion, the model takes BOTH the noisy image AND the timestep t**

> **بالعربي — ملخص النقاط الخطيرة:**
> 1. الـ KL مش متماثل — D(P‖Q) ≠ D(Q‖P)
> 2. تعديل قيم z مش نفس تعديل μ و σ
> 3. الـ Decoder لوحده محتاج نقل أوزان
> 4. الـ VAE بيستخدم log-variance مش variance عادي
> 5. الـ Diffusion بيتنبأ بالتشويش مش بالصورة النظيفة
> 6. موديل الـ Diffusion بياخد الصورة المشوشة + رقم الخطوة الزمنية كمدخلات

---

> **بالتوفيق في الامتحان! 🎯** Remember: understand the WHY behind each model, not just the WHAT. Each model was created to solve a specific limitation of the previous one.
