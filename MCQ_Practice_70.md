# 📝 Generative AI — 70 MCQ Practice Questions for Final Exam

> **70 سؤال اختيار من متعدد يغطوا كل المنهج — جاهز للمراجعة النهائية**
>
> Questions are ordered by topic. Answers with explanations are at the bottom.

---

## 🔤 Word2Vec (Questions 1–10)

**Q1.** What is the main goal of Word2Vec?
- A) Classify text into categories
- B) Map words from a higher-dimensional vector to a lower-dimensional vector
- C) Translate words from one language to another
- D) Remove stop words from sentences

**Q2.** Which hypothesis does Word2Vec rely on?
- A) Bayesian Hypothesis
- B) Central Limit Theorem
- C) Distributional Hypothesis — "a word is known by the company it keeps"
- D) Universal Approximation Theorem

**Q3.** In the Word2Vec implementation, with a vocabulary of 21 unique words and an embedding size of 2, what is the shape of the hidden layer weight matrix?
- A) (2, 21)
- B) (21, 2)
- C) (21, 21)
- D) (2, 2)

**Q4.** What is the activation function used in the hidden (embedding) layer of the Word2Vec model?
- A) ReLU
- B) Sigmoid
- C) Softmax
- D) Linear

**Q5.** With a context window of 2, for the sentence "future king prince", which of the following is a valid word pair?
- A) ['future', 'prince']
- B) ['future', 'future']
- C) ['king', 'the']
- D) ['prince', 'is']

**Q6.** What loss function is used in the Word2Vec model?
- A) Mean Squared Error
- B) Binary Cross-Entropy
- C) Categorical Cross-Entropy
- D) KL Divergence

**Q7.** After training Word2Vec, how are the word embeddings extracted?
- A) From the output layer weights
- B) From the input layer weights
- C) From the hidden layer weights (`model.get_weights()[0]`)
- D) From the bias terms

**Q8.** What does the output layer of the Word2Vec model predict?
- A) The input word itself
- B) The context (neighboring) word
- C) The part of speech
- D) The sentence sentiment

**Q9.** In the preprocessing step of Word2Vec, what are stop words?
- A) Words with special characters
- B) Common words like "the", "is", "a" that are removed
- C) Words that appear only once in the corpus
- D) Words at the end of sentences

**Q10.** If two words have similar embeddings after Word2Vec training, what does this mean?
- A) They are spelled similarly
- B) They appear in similar contexts
- C) They have the same number of letters
- D) They belong to the same sentence

---

## 🔧 AutoEncoders — Dense (Questions 11–18)

**Q11.** In an AutoEncoder, what is the "bottleneck" (latent space)?
- A) The input layer
- B) The loss function
- C) The compressed representation between encoder and decoder
- D) The output layer

**Q12.** What is the loss function used in the Dense AutoEncoder for MNIST images (pixels normalized to [0,1])?
- A) Categorical Cross-Entropy
- B) Mean Squared Error
- C) Binary Cross-Entropy
- D) Hinge Loss

**Q13.** In the course's Dense AutoEncoder, the input is 784 dimensions and the encoding is 32 dimensions. What is the compression ratio?
- A) 784:1
- B) 32:784
- C) 784:32 (≈ 24.5:1)
- D) 1:1

**Q14.** When creating a separate decoder model from a trained autoencoder, what must you do?
- A) Retrain the decoder from scratch
- B) Transfer (copy) the weights from the autoencoder's decoder layers
- C) Freeze the encoder layers
- D) Use a different loss function

**Q15.** What is the easier alternative to manual weight transfer for the decoder?
- A) Use `model.save()` and `model.load()`
- B) Use `autoencoder.layers[-1]` to get a reference to the same layer object
- C) Train encoder and decoder separately from the start
- D) Use transfer learning from a pre-trained model

**Q16.** When training the autoencoder, what is used as the target (y)?
- A) The class labels (0-9)
- B) Random noise
- C) The input itself (x_train, x_train)
- D) Zero vectors

**Q17.** After training the autoencoder, if you modify some values in the encoded representation and then decode, what happens?
- A) The decoder crashes
- B) You get a modified version of the original image
- C) You get the exact same image
- D) You get a blank image

**Q18.** In the course's code, the autoencoder, encoder, and decoder are separate model objects. How many models are trained explicitly?
- A) 3 — all three are trained separately
- B) 2 — encoder and decoder are trained separately
- C) 1 — only the autoencoder is trained; encoder and decoder share its weights
- D) 0 — no training is needed

---

## 🔧 AutoEncoders — Convolutional (Questions 19–25)

**Q19.** Why are convolutional autoencoders better than dense autoencoders for images?
- A) They use fewer parameters
- B) They understand spatial structure — nearby pixels are related
- C) They train faster on CPUs
- D) They don't need activation functions

**Q20.** What does `UpSampling2D((2,2))` do?
- A) Applies a convolution with stride 2
- B) Doubles the spatial dimensions by repeating pixels
- C) Reduces the image size by half
- D) Adds padding to the image

**Q21.** What is the formula for `Conv2DTranspose` output size?
- A) OutSize = n × s + k
- B) OutSize = (n + 1) × s - k
- C) OutSize = (n - 1) × s + k
- D) OutSize = n / s + k

**Q22.** Given an input size n=10, stride s=1, and kernel k=3, what is the Conv2DTranspose output size?
- A) 10
- B) 11
- C) 12
- D) 13

**Q23.** What is the key difference between UpSampling2D and Conv2DTranspose?
- A) UpSampling2D is slower
- B) Conv2DTranspose has no learnable parameters
- C) UpSampling2D has no learnable parameters; Conv2DTranspose has learnable weights
- D) They produce identical results

**Q24.** In the convolutional encoder, which layer combination is used to reduce spatial size?
- A) Conv2D + UpSampling2D
- B) Conv2D + MaxPooling2D
- C) Conv2DTranspose + MaxPooling2D
- D) Dense + Flatten

**Q25.** What is the final activation function in the convolutional decoder?
- A) ReLU
- B) Tanh
- C) Sigmoid
- D) Softmax

---

## 🚶 Latent Space Walking (Questions 26–28)

**Q26.** What does `np.linspace(A, B, 20)` do in the context of latent walking?
- A) Creates 20 random points between A and B
- B) Creates 20 evenly-spaced points between vectors A and B
- C) Creates 20 copies of A
- D) Trains the model for 20 epochs

**Q27.** What is the main problem revealed by latent space walking in standard autoencoders?
- A) The decoder is too slow
- B) The encoder loses information
- C) The latent space has holes/dead zones where random points produce garbage
- D) The images are too small

**Q28.** Latent space walking works by:
- A) Retraining the encoder between each step
- B) Interpolating between two latent vectors and decoding each interpolated point
- C) Adding noise to the original image
- D) Changing the decoder architecture

---

## 📊 Probability & KL Divergence (Questions 29–38)

**Q29.** What does conditional probability P(A|B) represent?
- A) Probability of A and B happening together
- B) Probability of A happening given B has already occurred
- C) Probability of A regardless of B
- D) Probability of B given A

**Q30.** What does joint probability P(A, B) represent?
- A) Probability of A or B
- B) Probability of A given B
- C) Probability of A and B happening together
- D) Probability of neither A nor B

**Q31.** The Expected Value E[f(X)] is best described as:
- A) The most frequently occurring value
- B) The median of all values
- C) The long-run weighted average
- D) The maximum possible value

**Q32.** Given: Strong market (probability 30%, profit $300,000) and Weak market (probability 70%, loss $100,000). What is the expected profit?
- A) $100,000
- B) $200,000
- C) $20,000
- D) -$70,000

**Q33.** KL Divergence D_KL(P ‖ Q) measures:
- A) The correlation between P and Q
- B) How much information is lost when Q is used to approximate P
- C) The average of P and Q
- D) Whether P and Q are independent

**Q34.** Is KL Divergence symmetric?
- A) Yes, D_KL(P‖Q) = D_KL(Q‖P) always
- B) No, D_KL(P‖Q) ≠ D_KL(Q‖P) in general
- C) Only when P = Q
- D) Only for normal distributions

**Q35.** What is D_KL(P ‖ Q) when P = Q?
- A) 1
- B) Infinity
- C) 0
- D) -1

**Q36.** The KL Divergence formula is:
- A) D_KL = Σ P(x) × Q(x)
- B) D_KL = Σ P(x) × ln(P(x) / Q(x))
- C) D_KL = Σ (P(x) - Q(x))²
- D) D_KL = Σ P(x) × Q(x) × ln(P(x))

**Q37.** Given P(1)=0.5, P(2)=0.3, P(3)=0.2 and Q(1)=0.4, Q(2)=0.4, Q(3)=0.2, what can we say about D_KL(P‖Q)?
- A) It equals zero because the distributions are similar
- B) It is positive because P ≠ Q
- C) It is negative
- D) It is undefined

**Q38.** From the exam sheet, D_KL(P‖Q) = 0.58 and D_KL(Q‖P) = 0.47. What does this prove?
- A) KL Divergence is symmetric
- B) KL Divergence is NOT symmetric
- C) The distributions are identical
- D) The formula is wrong

---

## 🔮 VAE — Concepts & Loss (Questions 39–52)

**Q39.** What is the main difference between a standard AutoEncoder and a VAE?
- A) VAE uses convolutional layers
- B) VAE encodes to a probability distribution (μ, σ²) instead of a single point
- C) VAE uses a different optimizer
- D) VAE doesn't have a decoder

**Q40.** In VAE, the encoder outputs:
- A) A single latent vector z
- B) Mean (μ) and log-variance (log σ²)
- C) The reconstructed image
- D) The class label

**Q41.** Why does VAE use log-variance instead of variance?
- A) It's faster to compute
- B) Variance must be positive, but log-variance can be any real number — easier for networks to output
- C) Log-variance gives better images
- D) It's required by the Keras API

**Q42.** The VAE loss function consists of:
- A) Only Reconstruction Loss
- B) Only KL Divergence Loss
- C) Reconstruction Loss + KL Divergence Loss
- D) Adversarial Loss + Reconstruction Loss

**Q43.** What does the Reconstruction Loss in VAE ensure?
- A) The latent space is organized
- B) The output looks similar to the input
- C) The distribution matches N(0,1)
- D) The model converges quickly

**Q44.** What does the KL Divergence Loss in VAE ensure?
- A) The output looks similar to the input
- B) The encoder's distribution q(z|x) stays close to N(0,1)
- C) The decoder produces sharp images
- D) The learning rate is appropriate

**Q45.** If the KL loss is too weak, what happens?
- A) All images look the same
- B) The latent space becomes disorganized with holes — like a standard autoencoder
- C) The model doesn't converge
- D) The images become too sharp

**Q46.** If the KL loss is too strong, what happens?
- A) The latent space has holes
- B) Everything collapses to the center — all images look like blurry averages
- C) The model trains faster
- D) The decoder fails

**Q47.** The Reparameterization Trick computes z as:
- A) z = μ × σ
- B) z = μ + σ × ε, where ε ~ N(0,1)
- C) z = ε × μ + σ
- D) z = μ - σ × ε

**Q48.** Why is the Reparameterization Trick needed?
- A) To speed up training
- B) Because sampling is a random operation that blocks backpropagation; the trick separates randomness from learnable parameters
- C) To reduce memory usage
- D) To normalize the latent space

**Q49.** In the VAE Sampling layer, `tf.exp(0.5 * z_log_var)` computes:
- A) The variance σ²
- B) The standard deviation σ
- C) The mean μ
- D) The log-variance

**Q50.** From the exam sheet: modifying z values directly (e.g., `ZZ[j,k] = z_random[j,k] - 1.3`) changes:
- A) The distribution parameters μ and σ
- B) The latent values z only — NOT the distribution parameters
- C) The decoder weights
- D) The encoder architecture

**Q51.** In the VAE, what does the Normal Distribution formula N(z; μ=4, σ=1) tell us about z=4?
- A) z=4 is impossible
- B) z=4 has the highest probability (it's the peak / mean)
- C) z=4 is the variance
- D) z=4 is the standard deviation

**Q52.** To generate NEW images using a trained VAE, you:
- A) Sample z ~ N(0,1) and pass it to the decoder
- B) Sample z ~ N(0,1) and pass it to the encoder
- C) Pass a real image to the decoder
- D) Retrain the model

---

## ⚔️ GAN (Questions 53–62)

**Q53.** A GAN consists of which two components?
- A) Encoder and Decoder
- B) Generator and Discriminator
- C) Encoder and Generator
- D) Decoder and Discriminator

**Q54.** What does the Generator do in a GAN?
- A) Classifies images as real or fake
- B) Compresses images into latent vectors
- C) Creates fake images from random noise z
- D) Trains the discriminator

**Q55.** What does the Discriminator output?
- A) A reconstructed image
- B) A latent vector
- C) A probability between 0 (fake) and 1 (real)
- D) A class label

**Q56.** During Discriminator training, why is `.detach()` used on fake images?
- A) To save memory
- B) To stop gradients from flowing back to the Generator — we only want to update D
- C) To normalize the images
- D) To convert images from GPU to CPU

**Q57.** What is the Generator's loss target?
- A) `fake_labels` (zeros) — G wants D to say fake
- B) `real_labels` (ones) — G wants D to think fakes are real
- C) The reconstruction error
- D) The KL divergence

**Q58.** What loss function does GAN use?
- A) Mean Squared Error
- B) Categorical Cross-Entropy
- C) Binary Cross-Entropy (BCELoss)
- D) KL Divergence

**Q59.** Why does the Generator use `nn.Tanh()` as its final activation?
- A) To output probabilities between 0 and 1
- B) To output pixel values in the range [-1, 1], matching the normalized training data
- C) To prevent vanishing gradients
- D) To classify the output

**Q60.** Why does the Discriminator use `nn.LeakyReLU(0.2)` instead of `nn.ReLU()`?
- A) LeakyReLU is faster
- B) LeakyReLU allows a small gradient for negative inputs (0.2×x), preventing dead neurons and stabilizing training
- C) ReLU is not compatible with Conv2d
- D) LeakyReLU produces better images

**Q61.** In GAN, the noise vector z has shape `(batch_size, latent_dim, 1, 1)`. Why the extra `1, 1` dimensions?
- A) For compatibility with fully connected layers
- B) Because ConvTranspose2d expects 4D input (batch, channels, height, width)
- C) To add padding
- D) For data augmentation

**Q62.** To convert a GAN-generated image from [-1,1] to [0,1] for display, you use:
- A) `img * 2 - 1`
- B) `(img + 1) / 2`
- C) `img / 255`
- D) `torch.sigmoid(img)`

---

## 🎯 CGAN (Questions 63–67)

**Q63.** What is the main advantage of CGAN over regular GAN?
- A) Faster training
- B) Better image quality
- C) You can control WHAT class the Generator produces (e.g., generate a specific digit)
- D) It uses fewer parameters

**Q64.** How are class labels injected in CGAN?
- A) As one-hot vectors
- B) As pixel values in the image
- C) Using `nn.Embedding` to convert labels to dense vectors
- D) By modifying the loss function

**Q65.** In the CGAN Generator, the label embedding size is `embedding_dim` (e.g., 10). In the Discriminator, the label embedding size is:
- A) Also `embedding_dim` (10)
- B) `28 × 28` (784) — same as the image size
- C) `latent_dim` (100)
- D) 1

**Q66.** In the CGAN Discriminator, the input has how many channels?
- A) 1 (image only)
- B) 2 (image channel + label map channel)
- C) 3 (RGB)
- D) 10 (one per class)

**Q67.** To generate specifically a digit "7" with a trained CGAN:
- A) `G(z)` — no label needed
- B) `G(z, torch.tensor([7]))` — pass label 7 to the Generator
- C) `D(img, torch.tensor([7]))` — pass label 7 to the Discriminator
- D) Retrain the model on only digit 7

---

## 🌊 Diffusion Models (Questions 68–70)

**Q68.** In the forward diffusion process, the formula `x_t = √α_t × x₀ + √(1-α_t) × ε` — as α_t decreases, what happens?
- A) The image becomes cleaner
- B) The image becomes noisier — more noise, less original image
- C) The image stays the same
- D) The model trains faster

**Q69.** What does the diffusion model predict during training?
- A) The clean original image
- B) The noise that was added to the image
- C) The class label
- D) The timestep t

**Q70.** In the diffusion training loop, the model receives two inputs. What are they?
- A) The clean image and its label
- B) The noisy image and the timestep t
- C) Two different images
- D) The noise and the clean image

---
---

# ✅ Answer Key with Explanations

## Word2Vec (1–10)

| # | Answer | Explanation |
|---|--------|-------------|
| 1 | **B** | Word2Vec's objective is mapping words from higher-dimensional (one-hot) to lower-dimensional (embedding) vectors |
| 2 | **C** | "A word is known by the company it keeps" — words in similar contexts get similar embeddings |
| 3 | **B** | 21 input words × 2 embedding dimensions = weight matrix shape (21, 2) |
| 4 | **D** | The embedding layer uses `activation='linear'` — no transformation, just raw weights |
| 5 | **A** | With window=2, 'future' and 'prince' are within 2 positions of each other |
| 6 | **C** | `categorical_crossentropy` — predicting which context word (one of 21 classes) |
| 7 | **C** | `model.get_weights()[0]` — the first layer's weights ARE the embeddings |
| 8 | **B** | The output layer predicts the context (neighboring) word using softmax |
| 9 | **B** | Stop words are common words (the, is, a, be, will) removed during preprocessing |
| 10 | **B** | Similar embeddings = words appear in similar contexts (distributional hypothesis) |

## AutoEncoders — Dense (11–18)

| # | Answer | Explanation |
|---|--------|-------------|
| 11 | **C** | The bottleneck is the compressed representation (latent space) between encoder and decoder |
| 12 | **C** | `binary_crossentropy` — pixels are normalized to [0,1], treated as probabilities |
| 13 | **C** | 784 input dims compressed to 32 latent dims = 784:32 ≈ 24.5:1 compression |
| 14 | **B** | You must copy/transfer weights from the autoencoder to the separate decoder model |
| 15 | **B** | Using `autoencoder.layers[-1]` gives a reference to the same layer object — shared weights |
| 16 | **C** | AutoEncoder target = input itself. `autoencoder.fit(x_train, x_train)` |
| 17 | **B** | Modifying encoded values produces a variation — a modified version of the image |
| 18 | **C** | Only the autoencoder is trained; encoder and decoder share its weights automatically |

## AutoEncoders — Convolutional (19–25)

| # | Answer | Explanation |
|---|--------|-------------|
| 19 | **B** | Conv layers understand spatial relationships — nearby pixels matter for images |
| 20 | **B** | UpSampling2D((2,2)) doubles height and width by repeating each pixel |
| 21 | **C** | Conv2DTranspose output = (n-1) × s + k |
| 22 | **C** | (10-1) × 1 + 3 = 9 + 3 = 12 |
| 23 | **C** | UpSampling2D just repeats pixels (no learning). Conv2DTranspose has trainable weights |
| 24 | **B** | Conv2D extracts features, MaxPooling2D reduces spatial size |
| 25 | **C** | Sigmoid — outputs pixel values between 0 and 1 |

## Latent Space Walking (26–28)

| # | Answer | Explanation |
|---|--------|-------------|
| 26 | **B** | `np.linspace` creates evenly-spaced points between two endpoints |
| 27 | **C** | Standard AE latent space has holes — random points can decode to garbage |
| 28 | **B** | Encode two images, interpolate between their latent vectors, decode each point |

## Probability & KL Divergence (29–38)

| # | Answer | Explanation |
|---|--------|-------------|
| 29 | **B** | P(A\|B) = probability of A given B has occurred |
| 30 | **C** | P(A,B) = probability of both A and B happening together |
| 31 | **C** | Expected value = Σ P(x)·f(x) = weighted average over the long run |
| 32 | **C** | E = 0.3×300,000 + 0.7×(-100,000) = 90,000 - 70,000 = $20,000 |
| 33 | **B** | KL measures information lost when using Q to approximate P |
| 34 | **B** | KL is NOT symmetric — D_KL(P‖Q) ≠ D_KL(Q‖P) in general |
| 35 | **C** | When P = Q, ln(P/Q) = ln(1) = 0, so D_KL = 0 |
| 36 | **B** | D_KL(P‖Q) = Σ P(x) × ln(P(x)/Q(x)) |
| 37 | **B** | P ≠ Q, so KL > 0 (KL is always ≥ 0, and equals 0 only when P = Q) |
| 38 | **B** | 0.58 ≠ 0.47 proves KL Divergence is NOT symmetric |

## VAE (39–52)

| # | Answer | Explanation |
|---|--------|-------------|
| 39 | **B** | VAE encodes to a distribution (μ, σ²) not a single point — this fills latent space holes |
| 40 | **B** | Encoder outputs μ (mean) and log(σ²) (log-variance), then z is sampled |
| 41 | **B** | σ² must be > 0, but log(σ²) can be any real number — unconstrained output for neural nets |
| 42 | **C** | VAE Loss = Reconstruction Loss + KL Divergence Loss |
| 43 | **B** | Reconstruction loss ensures output ≈ input |
| 44 | **B** | KL loss forces q(z\|x) close to N(0,1) — organizes the latent space |
| 45 | **B** | Weak KL → distributions spread out → holes appear → back to standard AE problems |
| 46 | **B** | Strong KL → everything collapses to center → all outputs look like blurry averages |
| 47 | **B** | z = μ + σ × ε (reparameterization trick), ε ~ N(0,1) |
| 48 | **B** | Sampling is non-differentiable. The trick makes z a differentiable function of μ, σ |
| 49 | **B** | exp(0.5 × log(σ²)) = exp(log(σ)) = σ (standard deviation) |
| 50 | **B** | Modifying z values directly doesn't change μ or σ — only the latent point is shifted |
| 51 | **B** | z=4 is the mean (μ=4), which is the peak of the normal distribution — highest probability |
| 52 | **A** | Sample z from N(0,1) → pass to decoder → new image. KL loss ensures N(0,1) works |

## GAN (53–62)

| # | Answer | Explanation |
|---|--------|-------------|
| 53 | **B** | GAN = Generator (creates fakes) + Discriminator (detects fakes) |
| 54 | **C** | Generator takes random noise z and produces fake images |
| 55 | **C** | Discriminator outputs a probability: 0 = fake, 1 = real (via Sigmoid) |
| 56 | **B** | `.detach()` prevents gradients from reaching G — only D's weights should update |
| 57 | **B** | G wants D to output 1 for fakes (think they're real) → target = `real_labels` |
| 58 | **C** | `nn.BCELoss()` — binary classification (real vs fake) |
| 59 | **B** | Tanh outputs [-1,1], matching training data normalized with `Normalize((0.5,),(0.5,))` |
| 60 | **B** | LeakyReLU passes small negative gradients (0.2×x for x<0), preventing dead neurons |
| 61 | **B** | ConvTranspose2d requires 4D tensors: (batch, channels, height, width). The 1×1 is spatial dims |
| 62 | **B** | `(img + 1) / 2` maps [-1,1] → [0,1] |

## CGAN (63–67)

| # | Answer | Explanation |
|---|--------|-------------|
| 63 | **C** | CGAN adds class labels → you choose which class to generate (e.g., digit "7") |
| 64 | **C** | `nn.Embedding` converts integer labels to dense vectors (same idea as Word2Vec) |
| 65 | **B** | G uses small embedding (10-dim), D uses image-sized embedding (28×28 = 784) to match spatial dims |
| 66 | **B** | 2 channels: 1 for the image + 1 for the label map (reshaped to 28×28) |
| 67 | **B** | `G(z, torch.tensor([7]))` — pass both noise z and the desired label to the Generator |

## Diffusion (68–70)

| # | Answer | Explanation |
|---|--------|-------------|
| 68 | **B** | Smaller α → √α shrinks (less original image kept), √(1-α) grows (more noise added) |
| 69 | **B** | The model predicts the NOISE (ε) that was added — not the clean image |
| 70 | **B** | Model receives: (1) the noisy image, and (2) the timestep t — it needs to know the noise level |

---

## 📊 Score Guide

| Score | Level | Advice |
|-------|-------|--------|
| **60-70** | 🌟 Excellent | أنت جاهز للامتحان — You're ready! |
| **50-59** | ✅ Good | Review the topics you got wrong |
| **40-49** | ⚠️ Needs Work | Re-read the summary, focus on concepts |
| **< 40** | 🔴 Study More | Go through the full summary carefully, then retry |

> **بالتوفيق! 🎯🤲**
