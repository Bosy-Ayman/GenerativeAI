# AutoEncoder Word2Vec Case Study: Full Summary

## 1. Objective and Overview

The primary goal of this case study is to implement a **Word2Vec algorithm** from scratch using deep learning in Python. The objective is to map words from a high-dimensional vector space (one-hot encoding) to a lower-dimensional, dense continuous vector space (word embeddings) where semantically similar words are positioned closer to each other.

The methodology follows five main phases: **Preprocessing, Generating Data Pairs, Training, Evaluation, and Post-processing (Visualization)**.

## 2. Step-by-Step Implementation Details

### Phase 1: Data Loading & Preprocessing

The model starts with a tiny corpus of text focused on a "royal family" theme (e.g., "The future king is the prince", "Only a woman can be a queen"). A robust `text_preprocessing` function is defined to clean the input strings:

- **Punctuation Removal:** Strips out characters like `!()-[]{};:'"\,<>./?@#$%^&*_“~`.
    
- **Number/Digit Removal:** Uses Regular Expressions (`re.sub`) to remove digits and words containing numbers.
    
- **Whitespace Cleaning:** Replaces multiple spaces with a single space and strips trailing whitespaces.
    
- **Lowercasing:** Converts all text to lowercase to maintain uniformity.
    
- **Tokenization:** Splits the text into a list of individual words.
    
- **Stop Word Removal:** Filters out common, uninformative words using a predefined list (e.g., _'and', 'a', 'is', 'the', 'in', 'be', 'will', 'was', 'but'_, etc.).
    

**Code Implementation:**

```python
import pandas as pd
import re

# Load Data
texts = pd.read_csv('TextNLP_5.csv')
texts = [x for x in texts['text']]

def text_preprocessing(
    text:list,
    punctuations = r'''!()-[]{};:'"\,<>./?@#$%^&*_“~''',
    stop_words=['and', 'a', 'is', 'the', 'in', 'be', 'will', 'was', 'but' , 'this', 'were', 'with', 'of' ,'also', 'on', '.' , 'for', 'any', 'its', 'and', 'are', 'from', 'both', 'as']
    )->list:
    """
    A method to preproces text
    """
    for x in text.lower():
        if x in punctuations:
            text = text.replace(x, "")
    # Removing words that have numbers in them
    text = re.sub(r'\w*\d\w*', '', text)
    # Removing digits
    text = re.sub(r'[0-9]+', '', text)
    # Cleaning the whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Setting every word to lower
    text = text.lower()
    # Converting all our text to a list
    text = text.split(' ')
    # Droping empty strings
    text = [x for x in text if x!='']
    # Droping stop words
    text = [x for x in text if x not in stop_words]
    return text
```

### Phase 2: Building Word Pairs (Context Window)

To train the network, the algorithm establishes context by looking at neighboring words.

- **Window Size:** Set to `2`. This means for every target word, the model looks 2 words ahead and 2 words behind.
    
- **Data Pairs Generation:** Iterates through the cleaned text to generate pairs of `[main_word, context_word]`. For example, from "future king prince", it generates pairs like `['future', 'king']`, `['future', 'prince']`, etc.
    

**Code Implementation:**

```python
# Defining the window for context
window = 2
# Creating a placeholder for the scanning of the word list
word_lists = []
all_text = []

for text in texts:
    # Cleaning the text
    text = text_preprocessing(text)
    # Appending to the all text list
    all_text += text
     # Creating a context dictionary
    for i, word in enumerate(text):
        for w in range(window):
            # Getting the context that is ahead by *window* words
            if i + 1 + w < len(text):
                word_lists.append([word] + [text[(i + 1 + w)]])
            # Getting the context that is behind by *window* words
            if i - w - 1 >= 0:
                word_lists.append([word] + [text[(i - w - 1)]])
```

### Phase 3: Dictionary Creation & One-Hot Encoding

To feed the text into a neural network, words must be converted to numbers.

- **Vocabulary Dictionary:** Identifies all unique words (features) in the corpus, sorts them alphabetically, and assigns each a unique integer index.
    
- **One-Hot Vectors:** Creates two matrices, `X` (input) and `Y` (target context). For every word pair, it creates two arrays filled with zeros, placing a `1` at the respective indices for the main and context words.
    
- **Tensor Conversion:** Converts the `X` and `Y` lists into TensorFlow tensors (`tf.float32`) for optimized deep learning operations.
    

**Code Implementation:**

```python
import numpy as np
import tensorflow as tf
from tqdm import tqdm

def create_unique_word_dict(text:list) -> dict:
    # Getting all the unique words from our text and sorting them alphabetically
    words = list(set(text))
    words.sort()
    # Creating the dictionary for the unique words
    unique_word_dict = {}
    for i, word in enumerate(words):
        unique_word_dict.update({word: i})
    return unique_word_dict

unique_word_dict = create_unique_word_dict(all_text)
n_words = len(unique_word_dict)
words = list(unique_word_dict.keys())

X = []
Y = []
for i, word_list in tqdm(enumerate(word_lists)):
    # Getting the indices
    main_word_index = unique_word_dict.get(word_list[0])
    context_word_index = unique_word_dict.get(word_list[1])

    # Creating the placeholders
    X_row = np.zeros(n_words)
    Y_row = np.zeros(n_words)

    # One hot encoding the main word & context word
    X_row[main_word_index] = 1
    Y_row[context_word_index] = 1

    # Appending to the main matrices
    X.append(X_row)
    Y.append(Y_row)

# Convert to Tensors
XX  = tf.convert_to_tensor(X, dtype=tf.float32)
YY  = tf.convert_to_tensor(Y, dtype=tf.float32)
```

### Phase 4: Model Architecture

A shallow neural network (functioning as an AutoEncoder/Skip-gram model) is built using Keras:

- **Input Layer:** Expects an input of shape matching the vocabulary size (representing the unique one-hot encoded words).
    
- **Hidden Layer (Embedding Layer):** A `Dense` layer with `2` units and a `linear` activation function. It uses a `GlorotUniform` initializer. This layer acts as the bottleneck, compressing the input down to a 2-dimensional vector.
    
- **Output Layer:** A `Dense` layer with units matching vocabulary size and a `softmax` activation function to output a probability distribution over the vocabulary.
    

**Code Implementation:**

```python
import keras
from keras.layers import Input, Dense
from keras.models import Model

def CreateModel():
  # Defining the size of the embedding
  embed_size = 2
  initializer = keras.initializers.GlorotUniform()

  inp = Input(shape=XX[0].shape)  
  x = Dense(units=embed_size, activation='linear', kernel_initializer=initializer)(inp)
  x = Dense(units=YY.shape[1], activation='softmax')(x)

  model = Model(inputs=inp, outputs=x)
  model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')
  model.summary()
  return model

model = CreateModel()
```

### Phase 5: Training Configuration

- **Loss Function:** `categorical_crossentropy` (standard for multi-class classification/probability outputs).
    
- **Optimizer:** `adam`.
    
- **Hyperparameters:** The model is trained using a `batch_size` of 10 for `200` epochs.
    

**Code Implementation:**

```python
# Optimizing the network weights
model.fit(
    x=XX,
    y=YY,
    batch_size=10,
    epochs=200
)
```

### Phase 6: Post-Processing (Extracting Embeddings)

The true goal of this neural network is not the actual predictions, but the **weights of the hidden layer**.

- The weights of the first hidden layer are extracted using `model.get_weights()[0]`.
    
- Each row in this weight matrix corresponds to a word in the vocabulary, and the two columns represent the X and Y coordinates in the new 2D embedding space.
    
- An `embedding_dict` is created, mapping each unique word to its newly learned 2-dimensional continuous array.
    

**Code Implementation:**

```python
# The input layer weights (Embeddings)
weights = model.get_weights()[0]

embedding_dict = {}
for word in words:
    embedding_dict.update({
        word: weights[unique_word_dict.get(word)]
    })
```

### Phase 7: Visualization

- Using `matplotlib.pyplot`, the 2D coordinates of every word are plotted on a scatter plot.
    
- `plt.annotate` is used to label each point with its corresponding word.
    
- Axes lines (x=0, y=0) are drawn for better spatial perspective.
    
- **Result:** Words with similar contexts (e.g., royal titles, genders) naturally group together in the 2D space, demonstrating the successful capture of semantic relationships.
    

**Code Implementation:**

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
i = 0
for word in list(unique_word_dict.keys()):
  coord = embedding_dict.get(word)
  plt.scatter(coord[0], coord[1])
  plt.annotate(word, (coord[0], coord[1]))
  i = i+1

# Drawing axes lines for reference
ax = [-1, 1]
ay = [0 , 0]
plt.plot(ax, ay, label = "line 1")
ax = [0  , 0]
ay = [-1 , 1]
plt.plot(ax, ay, label = "line 1")

plt.show()
```