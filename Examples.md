![[Pasted image 20260401081025.png]]

# حساب الـ variance لكل eigenvector
```python
variances = []

for j in range(eigen_vectors.shape[1]):   # نكرر لكل eigenvector

    eigen_vector = eigen_vectors[:, j]

    tot = 0
    # حساب الـ Mean
    for i in range(X_meaned.shape[0]):
        a = np.dot(X_meaned[i], eigen_vector)
        tot = tot + a

    Mean = tot / X_meaned.shape[0]

    # حساب الـ Variance
    tot = 0
    for i in range(X_meaned.shape[0]):
        a = np.dot(X_meaned[i], eigen_vector)   # تصحيح e → eigen_vector
        tot = tot + (a - Mean) * (a - Mean)

    Variance = tot / (X_meaned.shape[0] - 1)

    variances.append(Variance)

# ترتيب الـ eigenvectors من الأكبر للأصغر
sorted_indices = np.argsort(variances)[::-1]

sorted_eigenvectors = eigen_vectors[:, sorted_indices]
sorted_variances = np.array(variances)[sorted_indices]

print("Variances:", variances)
print("Sorted Variances:", sorted_variances)
print("Sorted Eigenvectors:\n", sorted_eigenvectors)
---
```
# 💻 + 🧠 الكود مع الشرح الرياضي

---

## 🔹 1) نلف على كل EigenVector

```python
for j in range(eigen_vectors.shape[1]):
    eigen_vector = eigen_vectors[:, j]
```

📌 هنا بنختار كل اتجاه ( v_j )

رياضيًا:  
[  
v_j = \text{EigenVector رقم } j  
]

---

## 🔹 2) حساب الـ Projection + Mean

```python
tot = 0
for i in range(X_meaned.shape[0]):
    a = np.dot(X_meaned[i], eigen_vector)
    tot = tot + a
```

📌 السطر ده:

```python
a = np.dot(X_meaned[i], eigen_vector)
```

يمثل:

[  
a_i = X_i \cdot v_j  
]

👉 ده **إسقاط (projection)** النقطة على الاتجاه

---

### ثم نحسب المتوسط:

```python
Mean = tot / X_meaned.shape[0]
```

رياضيًا:

[  
\mu = \frac{1}{n} \sum_{i=1}^{n} a_i  
]

---

## 🔹 3) حساب الـ Variance

```python
tot = 0
for i in range(X_meaned.shape[0]):
    a = np.dot(X_meaned[i], eigen_vector)
    tot = tot + (a - Mean) * (a - Mean)
```

📌 الجزء ده:

```python
(a - Mean) * (a - Mean)
```

يمثل:

[  
(a_i - \mu)^2  
]

---

### ثم نحسب الـ Variance:

```python
Variance = tot / (X_meaned.shape[0] - 1)
```

رياضيًا:

[  
\sigma^2 = \frac{1}{n-1} \sum_{i=1}^{n} (a_i - \mu)^2  
]

---

## 🔹 4) تخزين الـ Variance

```python
variances.append(Variance)
```

📌 ده معناه:

[  
\lambda_j = \sigma^2  
]

👉 كل variance = eigenvalue

---

## 🔹 5) ترتيب القيم

```python
sorted_indices = np.argsort(variances)[::-1]
```

📌 رياضيًا:

[  
\lambda_1 \ge \lambda_2 \ge \lambda_3 \ge \dots  
]

---

## 🔹 6) ترتيب الـ EigenVectors

```python
sorted_eigenvectors = eigen_vectors[:, sorted_indices]
```

📌 بنرتب الاتجاهات حسب أهميتها

---

# 🔥 أهم فكرة في السؤال

بدل ما يكون عندك:

[  
\lambda_j \text{ (Eigenvalue)}  
]

إنت حسبتها بنفسك:

[  
\lambda_j = \text{Var}(X \cdot v_j)  
]

---

# 🎯 الخلاصة

- بنسقط البيانات:  
    [  
    a_i = X_i \cdot v  
    ]
    
- نحسب المتوسط:  
    [  
    \mu = \frac{1}{n} \sum a_i  
    ]
    
- نحسب التباين:  
    [  
    \sigma^2 = \frac{1}{n-1} \sum (a_i - \mu)^2  
    ]
    
- ده = **eigenvalue**
    
- نرتب:  
    👉 الأكبر = أهم اتجاه (Principal Component)
    

---

```python

for each eigen_vector:

    # projection + mean
    tot = 0
    for each sample:
        a = dot(X, eigen_vector)
        tot += a
    Mean = tot / n

    # variance
    tot = 0
    for each sample:
        a = dot(X, eigen_vector)
        tot += (a - Mean)^2

    Variance = tot / (n - 1)
```

---
![[Pasted image 20260401082317.png]]
```python
Result = encoder.predict(x_train)  
  
Mu = Result[0]  
Sigma = Result[1]  
Z = Result[2]  
  
Samples, LatDim = Z.shape  
  
for i in range(Samples):  
for j in range(LatDim):  
if Z[i, j] < 0.1:  
Z[i, j] += 0.3  
  
Decoded = decoder.predict(Z)
```

---

![[Pasted image 20260401083910.png]]
```python
import numpy as np

Result = encoder.predict(x_train)

Mu = Result[0]      # المتوسطات
Sgma = Result[1]    # الانحرافات
Z = Result[2]       # latent vectors

Samples, LatDim = Z.shape

for i in range(Samples):
    # نبدأ بافتراض كل الانحرافات كبيرة
    isAllGreaterThan = True
    
    for s in Sgma[i]:
        if s < 1.5:
            isAllGreaterThan = False
            break  # كفاية نعرف إنه فيه واحدة أقل من 1.5
    
    if isAllGreaterThan:
        # لو كل sigma > 1.5، نضيف noise = 0.3
        Sgma[i] *= 0.3
        
        # نعيد sample من الـ latent space
        Z[i] = Mu[i] + Sgma[i] * np.random.randn(LatDim)

# نعمل decode للـ Z بعد التعديل
Decoded = decoder.predict(Z)
```

- **Loop على كل sample**
- **Check: all σ > 1.5?**
- **لو صح → noise controlled**
- **Re-sample Z = Mu + Sgma * randn**
- **Decode**:

---


# 🔹 Steps for the exam (Word2Vec Q2 modification + distance)

1. **Copy the weights**
    

```python
Modified = copy.deepcopy(model.get_weights()[0])
```

2. **Modify Q2 words (x<0, y>0)**
    

```python
for k in range(Modified.shape[0]):
    if Modified[k][0] < 0 and Modified[k][1] > 0:
        Modified[k][0] = 0
```

3. **Pick the words**
    

```python
Word_11 = original[11]
Word_3  = original[3]
Word_11_after = Modified[11]
Word_3_after = Modified[3]
```

4. **Compute Euclidean distance**
    

```python
DistBefore = np.linalg.norm(Word_11 - Word_3)
DistAfter  = np.linalg.norm(Word_11_after - Word_3_after)
```

5. **Interpret**
    

- DistBefore → المسافة الأصلية
    
- DistAfter → المسافة بعد التعديل على البعد الثاني فقط
    

---

### 🔹 خلاصتها في جملة واحدة:

> Copy weights → Set x=0 for Q2 words → Pick words 11 & 3 → Compute distance before & after → Compare. ✅

---

![[Pasted image 20260401094637.png]]

---

![[Pasted image 20260401094730.png]]

---

![[Pasted image 20260401104915.png]]
# نفترض Mu جاي من الـ encoder
```python
Result = encoder.predict(x_train)
Mu = Result[0]

# Loop على كل عنصر في Mu
for i in range(Mu.shape[0]):        # عدد الصور / العينات
    for k in range(Mu.shape[1]):    # عدد أبعاد الـ latent vector
        if Mu[i][k] > 0:            # لو القيمة موجبة
            Mu[i][k] = -Mu[i][k]    # نخليها سالبة

```

Mu الآن كل القيم الموجبة أصبحت سالبة

---

![[Pasted image 20260401094837.png]]

![[Pasted image 20260401094851.png]]

---
![[Pasted image 20260401111000.png]]
```python
import numpy as np

def print_nearest_left_vector(weights, i):
    # Get the target i-th vector
    target_vector = weights[i]
    target_x = target_vector[0] 
    
    min_distance = float('inf')
    nearest_vector = None
    
    # Iteratively check each vector in the weights matrix
    for j in range(len(weights)):
        current_vector = weights[j]
        current_x = current_vector[0]
        
        # Condition 1: Check if the vector is to the LEFT of the i-th vector
        if current_x < target_x:
            
            # Calculate the Euclidean distance between the current vector and target vector
            distance = np.linalg.norm(current_vector - target_vector)
            
            # Condition 2: Check if this is the shortest distance found so far
            if distance < min_distance:
                min_distance = distance
                nearest_vector = current_vector
                
    # Print the result
    if nearest_vector is not None:
        print("Nearest vector at the left:", nearest_vector)
    else:
        print("No vectors found to the left of the given vector.")
        
    return nearest_vector

# Example usage (assuming 'weights' is already extracted as per the prompt):
# weights = model.get_weights()[0]
# print_nearest_left_vector(weights, i)
```