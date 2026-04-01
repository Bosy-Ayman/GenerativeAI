# Comprehensive Summary: Principal Component Analysis (PCA)

This document provides a detailed summary of the PCA lecture, combining the theoretical foundations (from Part A), the mathematical intuition, and the step-by-step Python implementation using NumPy and Matplotlib (from Part B and the provided code).

## Part 1: Theoretical Foundations

==Principal Component Analysis (PCA) is a dimensionality reduction technique.== The primary objective is to project high-dimensional data onto a lower-dimensional subspace while retaining as much of the data's original variation as possible.

### 1. Variance and Covariance

To understand how data is distributed, we look at two main metrics:

- **Variance (**$\sigma_x^2$**):** Measures the spread of the data along a single axis.
    
    $$\sigma_x^2 = \frac{1}{N-1} \sum_{i=1}^N (x_i - \mu)^2 = \sigma(x, x)$$
- **Covariance (**$\sigma(x,y)$**):** Measures the joint variability (orientation) between two variables. If positive, they increase together; if negative, one increases while the other decreases.
    
    $$\sigma(x,y) = \frac{1}{n-1} \sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})$$

### 2. The Covariance Matrix ($\Sigma$)

For a multi-dimensional dataset, the variances and covariances are organized into a square matrix. For 2D data ($x, y$), the covariance matrix is:

$$\Sigma = \begin{bmatrix} \sigma(x,x) & \sigma(x,y) \\ \sigma(y,x) & \sigma(y,y) \end{bmatrix}$$

This matrix encapsulates both the spread and the orientation of the entire dataset.

### 3. Eigen Decomposition

The goal of PCA is to find a new set of axes (vectors) that best describe the shape of the data.

- **Objective:** Find a vector $v$ such that the projection of our data onto $v$ gives the largest possible variance.
    
- Mathematically, this means we need to maximize the equation:
    
    $$v^T \Sigma v$$
- Solving this gives us **Eigenvectors** (the directions of the new axes) and **Eigenvalues** ($\lambda$, the magnitude of variance along those new axes).
    

## Part 2: PCA Implementation (Step-by-Step)

The lecture demonstrates PCA on an image dataset containing 30 samples of handwritten digits (0-9).

- **Original Image Size:** $7 \times 4$ pixels.
    
- **Flattened Dimension:** $7 \times 4 = 28$ dimensions.
    
- **Dataset Shape:** $X$ is a matrix of shape $(30 \times 28)$.
    

### Step 1: Centering the Data

Before applying PCA, the data must be centered around the origin. This is done by calculating the mean of each feature (pixel) across all images and subtracting it from the original dataset.

```python
mean = np.mean(X, axis=0)
X_meaned = X - mean
```

### Step 2: Calculating the Covariance Matrix

Using the centered data, we compute the covariance matrix to understand how the 28 pixels relate to one another.

```
# rowvar=False implies that columns represent variables (28 dimensions)
cov_mat = np.cov(X_meaned, rowvar=False) 
```

![[Pasted image 20260331212057.png]]
- **Output Shape:** The resulting covariance matrix is $(28 \times 28)$.
    

### Step 3: Eigen Decomposition

We extract the eigenvalues and eigenvectors from the covariance matrix.

```
eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)
```

- **Output:** 28 eigenvalues and 28 corresponding eigenvectors. The eigenvectors have a shape of $(28 \times 28)$.
    ![[Pasted image 20260331212130.png]]
### Step 4: Sorting and Selecting Principal Components

Not all eigenvectors are equally important. We sort them based on their eigenvalues in descending order. The eigenvectors with the highest eigenvalues contain the most information (variance).

```python
# Sort eigenvalues in descending order
sorted_index = np.argsort(eigen_values)[::-1]
sorted_eigenvalue = eigen_values[sorted_index]
sorted_eigenvectors = eigen_vectors[:, sorted_index]

# Select the top 'n' components (e.g., n=2 for 2D visualization)
n_components = 2
eigenvector_subset = sorted_eigenvectors[:, 0:n_components] 
```

- **Subset Shape:** For 2 components, the subset shape is $(28 \times 2)$.
    

### Step 5: Dimensionality Reduction (Projection)

We project the original high-dimensional (28D) data onto the new, lower-dimensional (2D or 3D) feature space using the dot product.

```python
# Projecting data onto 2 components
X_reduced = np.dot(X_meaned, eigenvector_subset) 
```

- **Output Shape:** The newly projected data has a shape of $(30 \times 2)$ or $(30 \times 3)$ depending on `n_components`.
    
- **Visualization:** This allows us to plot the 30 images as individual points on a 2D or 3D scatter plot, where images of the same digits naturally cluster together.
    

### Step 6: Data Reconstruction (Inverse Transform)

To see how much information was retained, we can reconstruct the images from the reduced dimensions back to the original 28 dimensions.

```python
# Project back to 28 dimensions
reconstructed = np.dot(X_reduced_3d, eigenvector_subset.transpose())

# Add the mean back to un-center the data
final_reconstructed_data = reconstructed + mean
```

By reshaping this `final_reconstructed_data` back into $7 \times 4$ matrices, we can visualize the images to see what they look like after being compressed and decompressed using PCA.