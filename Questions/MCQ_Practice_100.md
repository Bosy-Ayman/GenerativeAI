# 📚 Generative AI — 100 MCQ Practice Questions for Final Exam

> **100 سؤال اختيار من متعدد يغطي كامل منهج الذكاء الاصطناعي التوليدي — جاهز للمراجعة النهائية**
>
> This comprehensive practice set is designed to help you prepare for tomorrow's final exam. It covers all course topics, coding exercises, sheet relationships, and mathematical formulations, while **strictly excluding PCA and Huffman coding**.
>
> Questions are ordered by topic. Detailed answers with explanations (in English & Arabic) are provided at the bottom.

---

## 🗺️ Part 1: Word2Vec (Questions 1–15)

**Q1.** What is the main objective of the Word2Vec model?
- A) To perform machine translation directly on raw strings
- B) To map words from a high-dimensional one-hot representation to a lower-dimensional dense vector space
- C) To count the frequency of words in a document and normalize them using TF-IDF
- D) To compress images of words using deep autoencoders

**Q2.** Which hypothesis serves as the foundation for the Word2Vec model?
- A) Central Limit Hypothesis
- B) Bayesian Inference Hypothesis
- C) Distributional Hypothesis — "A word is known by the company it keeps"
- D) Universal Approximation Hypothesis

**Q3.** Given a vocabulary of 21 unique words and an embedding size (hidden layer size) of 2, what is the shape of the weight matrix of the first Dense layer (input to hidden)?
- A) (2, 21)
- B) (21, 2)
- C) (21, 21)
- D) (2, 2)

**Q4.** What activation function is applied to the hidden embedding layer in a standard Word2Vec neural network?
- A) ReLU
- B) Sigmoid
- C) Softmax
- D) Linear (no activation function)

**Q5.** In Word2Vec preprocessing, which of the following describes "stop words"?
- A) Rare words that appear only once in the corpus
- B) Common grammar words (like "the", "is", "a", "be") that are removed to focus on meaningful words
- C) Punctuation marks at the end of sentences
- D) Out-of-vocabulary words replaced with an `<unk>` token

**Q6.** With a context window of 2, for the processed phrase `["future", "king", "prince"]`, which of the following is a valid training word pair `[input, target]`?
- A) `["future", "prince"]`
- B) `["future", "future"]`
- C) `["king", "the"]`
- D) `["prince", "is"]`

**Q7.** What loss function is typically optimized when training a classification-based Word2Vec network predicting a context word out of $V$ vocabulary classes?
- A) Mean Squared Error (MSE)
- B) Binary Cross-Entropy
- C) Categorical Cross-Entropy
- D) Kullback-Leibler (KL) Divergence

**Q8.** After training is completed, how are the word embeddings extracted from the model?
- A) From the output classification layer weights
- B) From the hidden layer weights (`model.get_weights()[0]`)
- C) From the softmax probability outputs
- D) From the training loss gradients

**Q9.** In a 2D embedding space, if the word vector for "available" is `(-0.18, -0.17)` and the word vector for "good" is `(-0.40, -0.12)`, what is the coordinate of the composite vector `newVec = "available" + "good"`?
- A) `(-0.58, -0.17)`
- B) `(-0.22, -0.05)`
- C) `(-0.58, -0.29)`
- D) `(-0.30, -0.14)`

**Q10.** If the vector `newVec = "available" + "good"` is computed as `(-0.58, -0.29)`, which of the following candidate words is closest to it in Euclidean distance?
- A) `talk` with vector `(-0.62, -0.14)`
- B) `one` with vector `(-0.38, -0.29)`
- C) `good` with vector `(-0.40, -0.12)`
- D) `hi` with vector `(-0.23, -0.28)`

**Q11.** In the Continuous Bag of Words (CBOW) architecture of Word2Vec, what does the network predict?
- A) It predicts the surrounding context words given a single target word
- B) It predicts the single target word given the surrounding context words
- C) It predicts if two words are synonyms or antonyms
- D) It predicts the sentiment of the context window

**Q12.** In the Skip-Gram architecture of Word2Vec, what is the input to the network?
- A) A bag-of-words vector representing the whole sentence
- B) A single target word (in one-hot format)
- C) Multiple context words combined by averaging
- D) A sequence of character embeddings

**Q13.** Why is Hierarchical Softmax or Negative Sampling used in large-scale Word2Vec training?
- A) To allow the network to handle continuous non-integer values
- B) To resolve the bottleneck of computing a full Softmax over a huge vocabulary ($V$) at each training step
- C) To make the embeddings symmetric
- D) To force the embeddings to follow a Gaussian distribution

**Q14.** In Word2Vec coding questions, if we check whether a vector is in the "same quarter" (quadrant) as target vector $i$, what mathematical check are we performing?
- A) Checking if they have the same Euclidean length
- B) Checking if their dot product is exactly 1
- C) Checking if the signs of their $x$ and $y$ coordinates match (`np.sign(x_j) == np.sign(x_i)` and `np.sign(y_j) == np.sign(y_i)`)
- D) Checking if their slopes are perpendicular

**Q15.** If we filter Word2Vec weights to find all vectors "above" the target vector $i$, which coordinate are we comparing?
- A) The first coordinate (x-coordinate, weight index `0`)
- B) The second coordinate (y-coordinate, weight index `1`)
- C) The absolute magnitude of the vector
- D) The angle of the vector relative to the origin

---

## 🔍 Part 2: Autoencoders — Dense (Questions 16–30)

**Q16.** What is the primary architecture layout of a standard Autoencoder?
- A) Input → Encoder → Latent Space (Bottleneck) → Decoder → Output
- B) Input → Generator → Discriminator → Output
- C) Noise → Latent Space → Decoder → Output
- D) Input → Classifier → Regressor → Output

**Q17.** What is the target output ($y$) of an Autoencoder during model training?
- A) Class labels representing the categories
- B) The input features themselves ($y = x$)
- C) A random noise vector
- D) A zero-filled array

**Q18.** An autoencoder where the dimension of the latent space ($z$) is smaller than the input dimension ($x$) is called:
- A) Overcomplete Autoencoder
- B) Variational Autoencoder
- C) Undercomplete Autoencoder
- D) Denoising Autoencoder

**Q19.** What is the main risk of an Overcomplete Autoencoder (where latent space dimension is larger than input)?
- A) The model fails to compile
- B) The model simply copies the input to output without learning any meaningful features
- C) The latent space has too many holes
- D) The reconstruction loss becomes infinite

**Q20.** In Keras, how is the training call for an Autoencoder set up?
- A) `autoencoder.fit(x_train, y_train)`
- B) `autoencoder.fit(x_train, x_train)`
- C) `autoencoder.fit(x_train, noise)`
- D) `autoencoder.fit(z, x_train)`

**Q21.** Which loss function is most suitable for an Autoencoder reconstructing images where pixel values are normalized between 0 and 1?
- A) Categorical Cross-Entropy
- B) Binary Cross-Entropy
- C) Hinge Loss
- D) KL Divergence

**Q22.** If the input to a Dense Autoencoder is an MNIST image ($28 \times 28 = 784$ pixels) and the latent bottleneck layer has 32 units, what is the compression ratio?
- A) 32:1
- B) 784:32 (approximately 24.5:1)
- C) 28:1
- D) 784:1

**Q23.** In Keras, if we define the encoder and decoder as separate `Model` objects sharing layers with the main `autoencoder` model, how many models need to be trained explicitly?
- A) All 3 must be trained separately
- B) Only the encoder and decoder are trained
- C) Only the main `autoencoder` model is trained; the others share its updated weights
- D) The models do not need training; they use random initialization

**Q24.** When building a separate `decoder` model using manual weight transfer from a trained `autoencoder` model, which of the following is correct?
- A) The decoder weights must be initialized to zero
- B) The decoder layer weights must be copied from the corresponding decoder layers of the trained `autoencoder`
- C) Only bias weights need to be transferred
- D) Decoder weights can be randomly sampled from a standard normal distribution

**Q25.** In Keras, what is the advantage of using a layer reference (e.g., `decoder_layer = autoencoder.layers[-1]`) to construct a separate decoder model?
- A) It makes the decoder run faster
- B) It automatically shares the weights of that layer, removing the need for manual weight copying/transfer
- C) It reduces the spatial size of the output
- D) It bypasses the activation function

**Q26.** Why is a standard Autoencoder poorly suited for generating new data by random sampling?
- A) It does not have a decoder layer
- B) The latent space is unregularized, resulting in "holes" and "dead zones" where random points decode into garbage images
- C) It can only reconstruct black-and-white images
- D) The loss function is not differentiable

**Q27.** If you modify a few dimensions of a test image's latent vector $z$ (e.g. subtracting a value from the first 16 dimensions) and pass it to the decoder, what do you expect?
- A) The decoder will crash due to shape mismatch
- B) You will get a modified version of the original image showing controlled semantic variations
- C) You will get a completely random set of noise pixels
- D) You will get the exact same original image

**Q28.** In an Autoencoder, what activation function is typically used in the final layer of the decoder if the input images are normalized to the range $[0, 1]$?
- A) ReLU
- B) Sigmoid
- C) Tanh
- D) Softmax

**Q29.** In an Autoencoder, what activation function is used in the intermediate hidden layers of both the encoder and decoder?
- A) Softmax
- B) Linear
- C) ReLU
- D) Sigmoid

**Q30.** What is the main purpose of the Decoder component in an Autoencoder?
- A) To compress the high-dimensional input to a bottleneck representation
- B) To map the compressed latent vector back to the original input space (reconstruction)
- C) To calculate the loss between original and generated data
- D) To add Gaussian noise to the input images

---

## 🖼️ Part 3: Autoencoders — Convolutional & Latent Walk (Questions 31–45)

**Q31.** Why are Convolutional Autoencoders preferred over Dense Autoencoders for image processing?
- A) They are much simpler to implement
- B) They preserve spatial structural relationships (local pixel dependencies) using shared convolutional kernels
- C) They do not require an activation function
- D) They output discrete classification labels

**Q32.** In a Convolutional Encoder, which layers are typically stacked to reduce the spatial size of the inputs?
- A) Conv2DTranspose and UpSampling2D
- B) Conv2D and MaxPooling2D
- C) Dense and Flatten
- D) Conv2DTranspose and MaxPooling2D

**Q33.** What is the behavior of the `UpSampling2D((2,2))` layer in a decoder?
- A) It scales the image using trainable weights
- B) It doubles the spatial dimensions (height and width) of the feature map by repeating rows and columns of pixels
- C) It halves the dimensions of the feature map by taking the maximum value
- D) It adds padding channels around the borders

**Q34.** What is the key functional difference between `UpSampling2D` and `Conv2DTranspose` layers?
- A) UpSampling2D has trainable weights; Conv2DTranspose does not
- B) UpSampling2D is a simple non-learnable pixel repetition; Conv2DTranspose performs a learnable upscaling operation with trainable parameters
- C) UpSampling2D reduces spatial dimensions; Conv2DTranspose increases them
- D) Conv2DTranspose is only used in the encoder

**Q35.** What is the formula to calculate the output size of a `Conv2DTranspose` layer (assuming no padding)?
- A) $\text{OutSize} = n \cdot s + k$
- B) $\text{OutSize} = (n - 1) \cdot s + k$
- C) $\text{OutSize} = (n + 1) \cdot s - k$
- D) $\text{OutSize} = n / s + k$
*(where $n$ is input size, $s$ is stride, and $k$ is kernel size)*

**Q36.** Using the formula $\text{OutSize} = (n - 1) \cdot s + k$, if a feature map has size $n = 14$, and we apply `Conv2DTranspose` with stride $s = 2$ and kernel size $k = 2$, what is the output size?
- A) 30
- B) 28
- C) 26
- D) 32

**Q37.** In Keras, what does setting `interpolation='bilinear'` inside `UpSampling2D` do?
- A) It adds learnable weights to the upsampling layer
- B) It performs smooth bilinear interpolation to resize the image instead of simple pixel replication
- C) It converts the image to grayscale
- D) It applies a max pooling step

**Q40.** What does a "Latent Space Walk" (Interpolation) involve?
- A) Randomly changing the weights of the trained encoder network
- B) Encoding two distinct images to vectors $A$ and $B$, generating intermediate points between them, and passing each to the decoder
- C) Passing noise vectors of increasing dimensions directly to the encoder
- D) Moving the filter kernels across the image pixel grid

**Q39.** In Python, which function is commonly used to generate intermediate, evenly-spaced latent vectors between two encoded representations $A$ and $B$?
- A) `np.random.normal`
- B) `np.linspace`
- C) `np.arange`
- D) `np.logspace`

**Q40.** If a Latent Space Walk is performed between an image of a "cat" and an image of a "dog", what do you expect to see in the decoded output of a smooth latent space?
- A) A sharp transition where the image instantly jumps from a cat to a dog at the midpoint
- B) A series of intermediate images where cat features smoothly morph into dog features
- C) A sequence of random static noise blocks
- D) A blank screen at all intermediate steps

**Q41.** In standard Autoencoders, why does a Latent Walk sometimes produce blurry or unrealistic garbage images at intermediate steps?
- A) Because the decoder weights are frozen
- B) Because the latent space has empty regions (holes) where the model was never trained, meaning the decoder doesn't know how to reconstruct them
- C) Because the learning rate was too small during training
- D) Because the latent dimensions are too large

**Q42.** If an MNIST input image has shape `(28, 28, 1)`, what is the shape of the output of the final layer of a Convolutional Autoencoder?
- A) `(784,)`
- B) `(28, 28, 1)`
- C) `(14, 14, 8)`
- D) `(32,)`

**Q43.** In a Convolutional Decoder, what is the typical role of a final `Conv2D(1, (3,3), padding='same')` layer placed right after upsampling blocks?
- A) To compress the feature map down to the bottleneck representation
- B) To project the multi-channel feature maps back to a single-channel (or 3-channel for RGB) image with a sigmoid activation
- C) To perform spatial pooling to reduce size
- D) To add class label information

**Q44.** When training a Convolutional Autoencoder, why do we use `padding='same'` in convolutional layers?
- A) To double the size of the output feature maps
- B) To keep the spatial dimensions constant after convolution, preventing the borders of the image from shrinking
- C) To add trainable parameters to the network
- D) To regularize the weights of the kernels

**Q45.** If the encoder has three MaxPool layers that downsample a `28x28` image to `14x14`, then `7x7`, then `4x4` (with padding), how many upsampling operations are needed in the decoder to reconstruct the `28x28` shape?
- A) 1
- B) 2
- C) 3
- D) 4

---

## 📈 Part 4: Probability & KL Divergence (Questions 46–60)

**Q46.** What does conditional probability $P(\text{Class} | \text{Features})$ represent?
- A) The probability of the class regardless of the features
- B) The probability of observing the class given that the features are already known
- C) The joint probability of both class and features occurring together
- D) The sum of class and feature probabilities

**Q47.** What is the relationship between joint probability $P(A, B)$ and conditional probability $P(A|B)$?
- A) $P(A, B) = P(A|B) + P(B)$
- B) $P(A, B) = P(A|B) \cdot P(B)$
- C) $P(A, B) = P(A|B) / P(B)$
- D) $P(A, B) = P(B|A) \cdot P(B)$

**Q48.** What is the mathematical definition of Bayes' Rule?
- A) $P(A|B) = \frac{P(B|A) \cdot P(B)}{P(A)}$
- B) $P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$
- C) $P(A|B) = P(B|A) \cdot P(A)$
- D) $P(A|B) = P(A, B) \cdot P(B)$

**Q49.** What does the Expected Value $E[f(X)]$ of a discrete random variable represent?
- A) The value of $f(X)$ at the maximum probability point (mode)
- B) The long-run weighted average of $f(X)$, calculated as $\sum_x P(x) \cdot f(x)$
- C) The absolute difference between the maximum and minimum values of $f(X)$
- D) The standard deviation of the variable $X$

**Q50.** Calculate the expected value $E[X]$ for a game where you have a 20% chance of winning \$500, a 50% chance of winning \$100, and a 30% chance of losing \$200:
- A) \$150
- B) \$90
- C) \$200
- D) \$120

**Q51.** What does the Kullback-Leibler (KL) Divergence $D_{KL}(P \| Q)$ measure?
- A) The similarity of the means of two normal distributions
- B) The distance between two points in a Euclidean space
- C) The amount of information lost (or divergence) when probability distribution $Q$ is used to approximate the true distribution $P$
- D) The correlation coefficient of two variables

**Q52.** What is the discrete formula for Kullback-Leibler Divergence $D_{KL}(P \| Q)$?
- A) $\sum_x P(x) \cdot Q(x)$
- B) $\sum_x P(x) \cdot \ln\left(\frac{P(x)}{Q(x)}\right)$
- C) $\sum_x Q(x) \cdot \ln\left(\frac{Q(x)}{P(x)}\right)$
- D) $\sum_x (P(x) - Q(x))^2$

**Q53.** Which of the following is a key mathematical property of KL Divergence?
- A) It is symmetric, meaning $D_{KL}(P \| Q) = D_{KL}(Q \| P)$
- B) It is non-negative, meaning $D_{KL}(P \| Q) \geq 0$, and equals 0 if and only if $P = Q$
- C) It is always less than or equal to 1
- D) It can take negative values if $Q(x) > P(x)$

**Q54.** Given $P = [0.5, 0.5]$ and $Q = [0.5, 0.5]$, what is $D_{KL}(P \| Q)$?
- A) 1
- B) 0
- C) Infinity
- D) -1

**Q55.** If $D_{KL}(P \| Q) = 0.58$ and $D_{KL}(Q \| P) = 0.47$ for two distributions $P$ and $Q$, what fundamental property of KL Divergence is demonstrated?
- A) Non-negativity
- B) Asymmetry (it is not a true distance metric)
- C) Shift invariance
- D) Variance preservation

**Q56.** In the term $\ln\left(\frac{P(x)}{Q(x)}\right)$ inside the KL Divergence formula, what happens if $Q(x) = 0$ for a point where $P(x) > 0$?
- A) The term becomes 0
- B) The divergence goes to positive infinity ($\infty$)
- C) The term becomes negative
- D) The equation ignores that point

**Q57.** When calculating KL Divergence in exams, how do you handle base-10 logarithms if the formula uses the natural logarithm ($\ln$)?
- A) $\ln(y) = \log_{10}(y) / 2.303$
- B) $\ln(y) = 2.303 \cdot \log_{10}(y)$
- C) $\ln(y) = \log_{10}(y)$
- D) $\ln(y) = e \cdot \log_{10}(y)$

**Q58.** Why is the KL Divergence term included in the loss function of a Variational Autoencoder (VAE)?
- A) To force the output image to match the input image pixel-by-pixel
- B) To minimize the number of parameters in the encoder network
- C) To regularize the encoder's output distribution $q(z|x)$, forcing it to approximate a standard normal distribution $\mathcal{N}(0, 1)$
- D) To prevent the gradients from vanishing during backpropagation

**Q59.** If a distribution $P(x)$ has a peak at $x = 2$, and $Q(x)$ has a peak at $x = 5$, what will happen to the KL Divergence $D_{KL}(P \| Q)$ as the two peaks move further apart?
- A) It will decrease towards 0
- B) It will increase
- C) It will remain constant
- D) It will become negative

**Q60.** The expectation term $E_{z \sim q(z|x)}[\ln p(z)]$ represents:
- A) The reconstruction loss of the VAE
- B) The average log-likelihood of the latent code under the prior distribution
- C) The entropy of the encoder distribution
- D) The learning rate of the optimization step

---

## 🔮 Part 5: Variational Autoencoders — VAE (Questions 61–80)

**Q61.** Why is a Variational Autoencoder (VAE) considered a "Generative" model, while a standard Autoencoder is not?
- A) VAE is trained on larger datasets
- B) VAE maps inputs to a continuous probability distribution in latent space, allowing new samples to be generated by sampling from this distribution
- C) VAE has more layers in its decoder
- D) VAE uses reinforcement learning

**Q62.** In a VAE, what two vectors does the encoder network output for a given input $x$?
- A) The reconstructed image $\hat{x}$ and the loss value
- B) The mean vector ($\mu$) and the log-variance vector ($\log \sigma^2$) of the latent distribution
- C) The latent vector $z$ and the noise vector $\epsilon$
- D) The class probabilities and the input pixel values

**Q63.** What is the purpose of the Reparameterization Trick in VAEs?
- A) To reduce the size of the latent dimension
- B) To allow backpropagation to flow through the sampling step by expressing the random variable $z$ as a deterministic function of parameters $\mu$ and $\sigma$ plus independent noise $\epsilon$
- C) To speed up the encoder network
- D) To convert the output image to grayscale

**Q64.** According to the Reparameterization Trick, how is the latent code $z$ computed?
- A) $z = \mu \cdot \sigma + \epsilon$
- B) $z = \mu + \sigma \cdot \epsilon$, where $\epsilon \sim \mathcal{N}(0, I)$
- C) $z = \epsilon + \sigma / \mu$
- D) $z = \mu - \sigma \cdot \epsilon$

**Q65.** Why does the VAE encoder output $\log \sigma^2$ (log-variance) instead of the variance $\sigma^2$ or standard deviation $\sigma$ directly?
- A) Logarithms make the weights smaller
- B) Variance must be strictly positive ($> 0$), which is hard to enforce on raw neural network outputs. $\log \sigma^2$ can be any real number ($-\infty$ to $+\infty$), which a Dense layer can output without constraints
- C) The Keras framework only supports logarithmic calculations
- D) It prevents the decoder from overfitting

**Q66.** In Keras, if the encoder outputs `z_log_var`, how do we obtain the standard deviation $\sigma$ in the sampling layer?
- A) `sigma = tf.exp(z_log_var)`
- B) `sigma = tf.exp(0.5 * z_log_var)`
- C) `sigma = tf.sqrt(z_log_var)`
- D) `sigma = tf.log(z_log_var) / 2`

**Q67.** The loss function of a VAE is the sum of which two components?
- A) Adversarial Loss and Mean Squared Error
- B) Reconstruction Loss and KL Divergence Loss
- C) Categorical Cross-Entropy and L2 Regularization
- D) Contrastive Loss and Binary Cross-Entropy

**Q68.** What is the consequence of having a VAE training loss with a very weak (or zero) weight on the KL Divergence term?
- A) The model fails to reconstruct images
- B) The latent space reverts to a disorganized state with holes (like a standard AE), and random sampling fails to generate realistic new images
- C) All generated images collapse to identical blurry averages
- D) The learning rate drops to zero

**Q69.** What is the consequence of having a VAE training loss with an excessively strong weight on the KL Divergence term?
- A) The reconstruction quality degrades completely, and the model outputs blurry average shapes as all latent distributions collapse onto the standard normal prior $\mathcal{N}(0, I)$
- B) The model overfits the training images and generates perfect copies
- C) The latent space develops large gaps
- D) The model trains extremely fast

**Q70.** What is the target distribution that the KL Divergence term forces the encoder's latent distributions to match?
- A) A uniform distribution $U(0, 1)$
- B) A standard normal distribution $\mathcal{N}(0, I)$ with mean 0 and identity covariance (variance 1)
- C) A binomial distribution
- D) The empirical distribution of the training labels

**Q71.** From the course slides/sheet, given the VAE latent distributions of Mickey: $q(z|x=\text{Mickey}) = \mathcal{N}(z; 1, 1)$ and Donald: $q(z|x=\text{Donald}) = \mathcal{N}(z; 4, 1)$. At what point does Donald's distribution reach its peak probability density?
- A) $z = 1$
- B) $z = 4$
- C) $z = 0$
- D) $z = 2.5$

**Q72.** For Donald's distribution $q(z|x=\text{Donald}) = \mathcal{N}(z; 4, 1)$, what is the relative probability density at $z = 1$ ($3\sigma$ away from the mean)?
- A) It is at its peak (highest value)
- B) It is extremely low (close to 0, approximately 0.004)
- C) It is exactly equal to the density at $z = 4$
- D) It is negative

**Q73.** Why do Mickey ($q(z|x=\text{Mickey}) = \mathcal{N}(z; 1, 1)$) and Donald ($q(z|x=\text{Donald}) = \mathcal{N}(z; 4, 1)$) have exactly the same peak probability density value ($\approx 0.398$)?
- A) Because they have the same mean ($\mu$)
- B) Because they both have a standard deviation $\sigma = 1$, and peak density is determined solely by $1 / (\sigma \sqrt{2\pi})$
- C) Because they represent the same class
- D) Because the latent space has no holes

**Q74.** In VAE coding, if we modify sampled latent vectors directly: `z[z < 0] = z[z < 0] * -0.8`, what are we changing?
- A) The encoder's distribution parameters ($\mu$ and $\sigma$)
- B) The sampled latent values ($z$) directly, leaving the distribution parameters unchanged
- C) The decoder's weights and biases
- D) The prior standard normal distribution

**Q75.** If the VAE encoder outputs $\mu_i$ for image $i$, and we modify the positive values of all mean vectors to be negative before sampling, what is the effect on the decoder?
- A) The decoder will output completely black images
- B) The decoder will generate images corresponding to the shifted latent coordinates, producing variations that might differ from the normal reconstruction
- C) The decoder layers will freeze and stop training
- D) The network will crash with a division-by-zero error

**Q76.** What does the mathematical expression $D_{KL}(q(z|x) \| p(z))$ represent in VAE?
- A) The reconstruction loss of the autoencoder
- B) The regularization term measuring how much information is lost when approximating the prior $p(z) = \mathcal{N}(0, I)$ using the encoder's distribution $q(z|x)$
- C) The classification error of the generated images
- D) The variance of the output pixel intensities

**Q77.** If we want to flip an image `Trgt` of shape `(28, 28)` vertically in Python, which numpy function should we use?
- A) `np.reshape(Trgt, (14, 56))`
- B) `np.flip(Trgt, axis=0)`
- C) `np.transpose(Trgt)`
- D) `np.rot90(Trgt)`

**Q78.** In the closed-form KL Divergence loss formula for VAE: $-\frac{1}{2} \sum \left(1 + \log(\sigma^2) - \mu^2 - \sigma^2\right)$, what value represents the loss if $\mu = 0$ and $\sigma^2 = 1$?
- A) 1
- B) 0
- C) $-0.5$
- D) Infinity

**Q80.** To generate a completely new, random image from a trained VAE model, what input is fed to the decoder?
- A) A test image from the dataset
- B) A latent vector $z$ sampled from a standard normal distribution $\mathcal{N}(0, I)$
- C) A vector of zeros
- D) The mean vector $\mu$ obtained by encoding a target image

**Q80.** In the reparameterization trick code `z_mean + tf.exp(0.5 * z_log_var) * epsilon`, what role does `epsilon` play?
- A) It is a learnable scaling parameter
- B) It is a random noise vector sampled from a standard normal distribution $\mathcal{N}(0, I)$
- C) It is the reconstruction error threshold
- D) It represents the learning rate of the optimizer

---

## ⚔️ Part 6: GANs & Conditional GANs — GAN/CGAN (Questions 81–92)

**Q81.** What is the core training dynamic of a Generative Adversarial Network (GAN)?
- A) An encoder compresses data while a decoder reconstructs it
- B) Two neural networks (Generator and Discriminator) compete in a zero-sum game
- C) A single model predicts the next pixel in a sequence
- D) A model learns to add noise to images and then remove it

**Q82.** What does the Generator network in a GAN take as input?
- A) A real training image
- B) A random noise vector $z$ from a latent space
- C) The target class labels only
- D) The output of the Discriminator network

**Q83.** What is the output of the Discriminator network in a standard GAN?
- A) A generated fake image
- B) A single scalar value between 0 and 1 representing the probability that the input image is real
- C) A multi-class probability distribution over all classes
- D) A latent vector $z$

**Q84.** During Discriminator training, why do we use `fake_images.detach()` in PyTorch before feeding them to the Discriminator?
- A) To convert the images to numpy arrays
- B) To freeze the Discriminator's weights
- C) To detach the fake images from the computation graph of the Generator, preventing gradients from flowing back to update the Generator's weights while training the Discriminator
- D) To speed up the GPU execution

**Q85.** What is the target label value used when calculating the loss for the Generator?
- A) All zeros (`fake_labels`), because the Generator wants to make fake images
- B) All ones (`real_labels`), because the Generator wants the Discriminator to classify its fake images as real
- C) A random distribution of label values
- D) The structural similarity index of the images

**Q86.** Which activation function is typically used in the final layer of a GAN Generator to output pixel values in the range $[-1, 1]$?
- A) Sigmoid
- B) Tanh
- C) ReLU
- D) Softmax

**Q87.** Why is LeakyReLU preferred over standard ReLU in the Discriminator network?
- A) It is faster to compute
- B) It allows a small non-zero gradient for negative values ($0.2 \cdot x$), preventing "dead neurons" and stabilizing the adversarial training process
- C) It restricts the output to positive values only
- D) It is required for convolutional operations

**Q88.** What is the main difference between a Conditional GAN (CGAN) and a standard GAN?
- A) CGAN is trained on multiple GPUs
- B) CGAN includes class labels (conditioning information) as inputs to both the Generator and Discriminator, allowing controlled generation of specific classes
- C) CGAN does not use a Discriminator
- D) CGAN uses mean squared error loss instead of cross-entropy

**Q89.** In a PyTorch CGAN implementation, how are the integer class labels processed before being fed into the networks?
- A) They are normalized between -1 and 1
- B) They are converted to dense vector embeddings using an `nn.Embedding` layer
- C) They are passed as raw floats
- D) They are converted to binary strings

**Q90.** In the CGAN Discriminator, if we concatenate the class label information with the input image, how does the channel dimension of the input change?
- A) It remains 1 (grayscale)
- B) It increases (e.g. from 1 channel to 2 channels, where the second channel is a filled spatial map of the label embedding)
- C) It is multiplied by the batch size
- D) It drops to 0

**Q91.** In PyTorch, what is the purpose of modifying a latent vector using `.data` (e.g., `z.data[0, i, 0, 0] += 0.3`) during a latent walk?
- A) To convert the tensor to a CPU tensor
- B) To perform an in-place modification directly on the underlying tensor data without tracking the operation in the autograd computation graph
- C) To calculate the gradients of the latent vector
- D) To save the tensor to disk

**Q92.** If a GAN Generator's output images are in the range $[-1, 1]$ due to a Tanh activation, how must they be normalized in PyTorch before displaying them?
- A) `img * 255.0`
- B) `(img + 1.0) / 2.0`
- C) `torch.clamp(img, 0, 1)`
- D) `img / 2.0`

---

## 🌊 Part 7: Diffusion Models (Questions 93–100)

**Q93.** How do Diffusion Models differ conceptually from GANs and VAEs?
- A) They do not use neural networks
- B) They generate images in a single forward pass
- C) They set up a double process: a forward process that systematically destroys data by adding Gaussian noise, and a reverse process where a network learns to denoise the data step-by-step
- D) They are only trained on text data

**Q94.** In the forward diffusion process, the transition at step $t$ is defined as $x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{1 - \alpha_t} \epsilon$. As the noise scheduler parameter $\alpha_t$ decreases over time, what is the effect on the image $x_t$?
- A) The image becomes cleaner and sharper
- B) The image contains less information from the original image ($x_0$) and becomes noisier
- C) The image size increases
- D) The variance of the image decreases to zero

**Q95.** What is the mathematical significance of the relation $(\sqrt{\alpha_t})^2 + (\sqrt{1 - \alpha_t})^2 = 1$ in the forward diffusion step?
- A) It is the minimax constraint
- B) It is the Variance Preservation relation, ensuring that the scale of the variance of the noised image remains stable (constant) at each step
- C) It is the normalization factor for the sigmoid function
- D) It proves that the model is symmetric

**Q96.** Given the single-step diffusion parameters $\alpha_0 = 0.9$ and $\alpha_1 = 0.8$, if we start with a clean image $x_0$ and sample noise vectors $\epsilon_0, \epsilon_1 \sim \mathcal{N}(0, I)$, what is the formula to calculate the noised image $x_2$ step-by-step?
- A) $x_2 = \sqrt{\alpha_1}(\sqrt{\alpha_0}x_0 + \sqrt{1-\alpha_0}\epsilon_0) + \sqrt{1-\alpha_1}\epsilon_1$
- B) $x_2 = \alpha_1 \alpha_0 x_0 + \epsilon_1$
- C) $x_2 = (\sqrt{\alpha_0} + \sqrt{\alpha_1})x_0 + (\sqrt{1-\alpha_0} + \sqrt{1-\alpha_1})\epsilon_1$
- D) $x_2 = \sqrt{\alpha_1 \alpha_0}x_0 + \epsilon_0 + \epsilon_1$

**Q97.** What is the core advantage of the cumulative noising formula $x_t = \sqrt{\hat{\alpha}_t} x_0 + \sqrt{1 - \hat{\alpha}_t} \epsilon$ (where $\hat{\alpha}_t = \prod_{i=1}^t \alpha_i$)?
- A) It allows the network to train without using a GPU
- B) It allows us to sample the noised image $x_t$ at ANY arbitrary timestep $t$ directly from the clean image $x_0$ in a single step, without iterating through all intermediate steps $1, \dots, t-1$
- C) It removes the need for the reverse denoising process
- D) It ensures the loss is calculated using binary cross-entropy

**Q98.** What does the neural network (usually a U-Net) predict during the training loop of a Diffusion Model?
- A) The clean original image $x_0$ directly
- B) The random noise vector $\epsilon$ that was added to the image at that specific timestep $t$
- C) The classification label of the image
- D) The value of the scheduler parameter $\alpha_t$

**Q99.** During the training of a Diffusion Model, what are the two inputs fed into the U-Net denoiser network?
- A) The clean image $x_0$ and the noise vector $\epsilon$
- B) The noisy image $x_t$ and the scalar timestep $t$ (time embedding)
- C) The noisy image $x_t$ and the class label
- D) Two different noisy images from different timesteps

**Q100.** What is the role of the "Noise Scheduler" ($\beta_t$ or $\alpha_t$) in a Diffusion Model?
- A) To control the learning rate of the optimizer
- B) To define the specific mathematical schedule for how much noise is added to the image at each timestep $t$
- C) To sort the training images by category
- D) To stop the training when the loss is minimized

---
---

# ✅ Answer Key with Explanations

## 🗺️ Word2Vec (1–15)

| # | Answer | Explanation |
|---|--------|-------------|
| 1 | **B** | Word2Vec's objective is mapping words from higher-dimensional (one-hot) to lower-dimensional (embedding) vectors. |
| 2 | **C** | "A word is known by the company it keeps" — words in similar contexts get similar embeddings. |
| 3 | **B** | 21 input words × 2 embedding dimensions = weight matrix shape (21, 2). |
| 4 | **D** | The embedding layer uses `activation='linear'` — no transformation, just raw weights. |
| 5 | **B** | Stop words are common words (the, is, a, be, will) removed during preprocessing. |
| 6 | **A** | With window=2, 'future' and 'prince' are within 2 positions of each other. |
| 7 | **C** | `categorical_crossentropy` — predicting which context word (one of 21 classes). |
| 8 | **B** | `model.get_weights()[0]` — the first layer's weights ARE the embeddings. |
| 9 | **C** | Vector addition coordinate-wise: $(-0.18 + (-0.40), -0.17 + (-0.12)) = (-0.58, -0.29)$. |
| 10 | **A** | The distance squared is $(-0.62 - (-0.58))^2 + (-0.14 - (-0.29))^2 = 0.0241$, which is the smallest. |
| 11 | **B** | CBOW predicts target word from context. Skip-Gram does the opposite. |
| 12 | **B** | Skip-Gram takes a single target word as input and outputs context predictions. |
| 13 | **B** | It resolves the computational bottleneck of computing full Softmax over large vocabularies ($V$). |
| 14 | **C** | Same quadrant means they have matching signs for both x and y coordinates. |
| 15 | **B** | "Above" refers to the y-coordinate (vertical axis), which is index 1 of the 2D vector. |

---

## 🔍 Autoencoders — Dense (16–30)

| # | Answer | Explanation |
|---|--------|-------------|
| 16 | **A** | Autoencoders compress inputs via an encoder to latent space and reconstruct via a decoder. |
| 17 | **B** | The network is trained unsupervised to make output as close as possible to the input ($y = x$). |
| 18 | **C** | An undercomplete autoencoder has a latent layer smaller than the input, forcing compression. |
| 19 | **B** | Having a latent space equal to or larger than input allows copying without feature extraction. |
| 20 | **B** | The model is trained with `x_train` as both inputs and targets. |
| 21 | **B** | Pixels normalized in $[0,1]$ are treated as probabilities; binary cross-entropy is the standard loss. |
| 22 | **B** | The input features (784) are divided by bottleneck size (32), which is $\approx 24.5$. |
| 23 | **C** | Since they share layers with the main model, training the autoencoder updates all weights. |
| 24 | **B** | Decoders require copying weights from the trained decoder layers of the main autoencoder. |
| 25 | **B** | Layer references automatically link the same layer in memory, sharing weights without copy steps. |
| 26 | **B** | The unconstrained latent space has holes that do not map to meaningful outputs. |
| 27 | **B** | Shifting coordinates moves the vector along continuous dimensions, causing output variations. |
| 28 | **B** | Sigmoid maps outputs to $[0,1]$ to match normalized pixel intensities. |
| 29 | **C** | ReLU is standard for hidden layers to ensure stable gradients and fast convergence. |
| 30 | **B** | The decoder's function is mapping the compressed latent representation back to input space. |

---

## 🖼️ Autoencoders — Convolutional & Latent Walk (31–45)

| # | Answer | Explanation |
|---|--------|-------------|
| 31 | **B** | Convolutional layers preserve spatial structures using shared kernels across local regions. |
| 32 | **B** | Conv2D extracts features, and MaxPooling2D downsamples spatial height and width. |
| 33 | **B** | UpSampling2D is a non-learnable layer that doubles features by copying pixel values. |
| 34 | **B** | UpSampling2D is deterministic; Conv2DTranspose uses trainable weights to learn upscaling. |
| 35 | **B** | Transposed convolution output formula: $(n-1) \cdot s + k$. |
| 36 | **B** | OutSize = $(14-1) \cdot 2 + 2 = 13 \cdot 2 + 2 = 28$. |
| 37 | **B** | Bilinear upsampling uses linear interpolation to create smoother borders instead of blocky pixels. |
| 38 | **B** | Latent walking interpolates between two latent vectors and decodes each intermediate step. |
| 39 | **B** | `np.linspace` generates linearly spaced vectors between starting and ending vectors. |
| 40 | **B** | Continual paths in latent space map to continuous, gradual transitions in decoded images. |
| 41 | **B** | Latent space holes correspond to regions that map to garbage outputs because no training data lay there. |
| 42 | **B** | The autoencoder reconstructs the input, maintaining the spatial shape `(28, 28, 1)`. |
| 43 | **B** | The final Conv2D layer projects channels down to the target color channel count with Sigmoid activation. |
| 44 | **B** | Same padding pads boundaries with zeros so that convolutions do not reduce spatial dimensions. |
| 45 | **C** | Three downsamplings require three corresponding upsampling steps to restore the original size. |

---

## 📈 Probability & KL Divergence (46–60)

| # | Answer | Explanation |
|---|--------|-------------|
| 46 | **B** | Conditional probability measures the likelihood of an event given that another event is known. |
| 47 | **B** | From conditional probability definition: $P(A, B) = P(A|B) \cdot P(B)$. |
| 48 | **B** | Bayes' rule calculates inverse conditional probability: $P(A|B) = P(B|A)P(A)/P(B)$. |
| 49 | **B** | The expected value is the probability-weighted average of all possible values of a random variable. |
| 50 | **B** | Expected value = $(0.2 \cdot 500) + (0.5 \cdot 100) + (0.3 \cdot -200) = 100 + 50 - 60 = 90$. |
| 51 | **C** | KL Divergence measures the asymmetric information difference between two distributions. |
| 52 | **B** | $D_{KL}(P \| Q) = \sum P(x) \ln(P(x) / Q(x))$ is the standard Shannon entropy-based definition. |
| 53 | **B** | KL Divergence is always non-negative ($D_{KL} \geq 0$) and equals 0 if and only if $P = Q$. |
| 54 | **B** | When $P = Q$, then $\ln(P(x)/Q(x)) = \ln(1) = 0$, so the divergence sum is 0. |
| 55 | **B** | $D_{KL}(P \| Q) \neq D_{KL}(Q \| P)$ proves KL is asymmetric (not a true distance metric). |
| 56 | **B** | If $Q(x) = 0$ and $P(x) > 0$, the ratio division by zero yields infinite divergence. |
| 57 | **B** | Natural log base conversion: $\ln(y) = \ln(10) \cdot \log_{10}(y) \approx 2.303 \cdot \log_{10}(y)$. |
| 58 | **C** | The KL term forces encoder distributions to approximate $\mathcal{N}(0, I)$, preventing disorganized space. |
| 59 | **B** | As distributions overlap less, the ratio $P(x)/Q(x)$ grows at the main density regions of $P$, increasing KL. |
| 60 | **B** | The expectation term measures the log-likelihood of the sampled latent vectors under the prior prior. |

---

## 🔮 Variational Autoencoders — VAE (61–80)

| # | Answer | Explanation |
|---|--------|-------------|
| 61 | **B** | VAE enforces a continuous latent space distribution, enabling generation of new samples. |
| 62 | **B** | The encoder outputs mean ($\mu$) and log-variance ($\log \sigma^2$) to parameterize the Gaussian distribution. |
| 63 | **B** | sampling is a random non-differentiable operation; the trick shifts the randomness to an independent noise term $\epsilon$. |
| 64 | **B** | Reparameterization: $z = \mu + \sigma \cdot \epsilon$ where $\epsilon \sim \mathcal{N}(0, I)$. |
| 65 | **B** | Variance must be positive, but log-variance can take any real value, making it easier for Dense layers to output. |
| 66 | **B** | $\sigma = \sqrt{\sigma^2} = \sqrt{\exp(\log \sigma^2)} = \exp(0.5 \cdot \log \sigma^2)$. |
| 67 | **B** | VAE loss consists of reconstruction loss (fidelity) and KL divergence loss (distribution regularization). |
| 68 | **B** | Without KL, VAE acts like a standard autoencoder, creating latent space holes. |
| 69 | **A** | Over-regularization forces the reconstruction term to be ignored, collapsing outputs to average blurs. |
| 70 | **B** | The target distribution is a standard normal distribution with mean 0 and variance 1 ($\mathcal{N}(0, I)$). |
| 71 | **B** | Normal distributions reach their maximum probability density at their mean: $z = \mu = 4$. |
| 72 | **B** | With $\mu=4, \sigma=1$, $z=1$ is $3\sigma$ away from the mean, yielding an extremely low density ($\approx 0.004$). |
| 73 | **B** | Both distributions have $\sigma=1$. Peak value depends only on the standard deviation: $1/(\sigma \sqrt{2\pi})$. |
| 74 | **B** | Modifying $z$ directly changes the sampled latent vector coordinates, not the encoder's distribution parameters. |
| 75 | **B** | Modifying means changes the center of the sampled space, generating semantic variations. |
| 76 | **B** | The KL term regularizes the VAE by checking how much the encoder distribution deviates from standard normal prior. |
| 77 | **B** | `np.flip(Trgt, axis=0)` flips the rows of a 2D array, which flips the image vertically. |
| 78 | **B** | If $\mu = 0$ and $\sigma^2 = 1$, the terms inside the summation cancel out, resulting in a loss of 0. |
| 79 | **B** | Since VAE latent space matches $\mathcal{N}(0, I)$, we sample $z \sim \mathcal{N}(0, I)$ and decode it to generate new images. |
| 80 | **B** | `epsilon` provides the random noise needed for sampling, keeping the parameter operations differentiable. |

---

## ⚔️ GANs & Conditional GANs — GAN/CGAN (81–92)

| # | Answer | Explanation |
|---|--------|-------------|
| 81 | **B** | GAN training is formulated as a minimax game between Generator (G) and Discriminator (D). |
| 82 | **B** | The Generator takes a random noise vector $z$ as input to produce fake images. |
| 83 | **B** | The Discriminator is a binary classifier outputting the probability that the input is real. |
| 84 | **C** | `.detach()` stops autograd tracking, preventing D's training step from updating G's weights. |
| 85 | **B** | The Generator wants the Discriminator to classify fake images as real, so its target is 1. |
| 86 | **B** | Tanh outputs values in $[-1, 1]$, matching training images normalized to the same range. |
| 87 | **B** | LeakyReLU allows small gradients for negative inputs, preventing dead neurons in the Discriminator. |
| 88 | **B** | CGAN conditions generation on class labels, allowing class-targeted image generation. |
| 89 | **B** | Embedding layers map discrete class IDs to continuous dense vectors. |
| 90 | **B** | The label embedding is expanded and added as an additional channel to the image input. |
| 91 | **B** | `.data` accesses the raw tensor directly, bypassing autograd tracking for in-place modifications. |
| 92 | **B** | To map $[-1, 1]$ back to $[0, 1]$, we apply the transformation: $(x + 1)/2$. |

---

## 🌊 Diffusion Models (93–100)

| # | Answer | Explanation |
|---|--------|-------------|
| 93 | **C** | Diffusion models use a forward noising process and a reverse denoising process. |
| 94 | **B** | As $\alpha_t$ decreases, the scaling factor of the original image shrinks, and the noise component grows. |
| 95 | **B** | Sum of squares equals 1, preserving the variance scale of the image at each step. |
| 96 | **A** | Step-by-step substitution: $x_1 = \sqrt{\alpha_0}x_0 + \sqrt{1-\alpha_0}\epsilon_0$, then substitute into $x_2$. |
| 97 | **B** | The cumulative formula allows jumping directly to step $t$ without iterating intermediate steps. |
| 98 | **B** | The network is trained to predict the noise $\epsilon$ added at timestep $t$. |
| 99 | **B** | The U-Net takes the noisy image $x_t$ and the timestep $t$ as inputs to predict the noise. |
| 100 | **B** | The scheduler parameters define how much noise is added at each step of the process. |

---

## 📊 Score Guide

| Score | Level | Advice |
|-------|-------|--------|
| **90–100** | 🌟 Master | You are 100% ready for the exam! |
| **80–89** | ✅ Great | Double-check the questions you missed |
| **70–79** | ⚠️ Review | Re-study the weak sections |
| **< 70** | 🔴 Study More | Re-read the summary files carefully |

> **بالتوفيق والنجاح في الامتحان! 🎯🤲**
