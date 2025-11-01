# PCA-From-Scratch-On-Mall-Customer-dataset
"ML models implemented from scratch using NumPy and Pandas only"

# 🧠 Principal Component Analysis (PCA) — From Scratch

## 📘 Overview
This project demonstrates **Principal Component Analysis (PCA)** implemented **from scratch** using only NumPy and visualized with Matplotlib and Seaborn.  
PCA is an **unsupervised dimensionality reduction** technique used to identify directions (principal components) that capture the **maximum variance** in data.

---

## ⚙️ Workflow

1. **📚 Import Libraries:** `numpy`, `pandas`, `matplotlib`, `seaborn`, `sklearn`
2. **🧼 Standardize Data:** Scale features to zero mean and unit variance.
3. **📊 Compute Covariance Matrix:**  
   $$\[\text{Cov}(Z) = \frac{1}{n-1} Z^T Z\]$$
4. **🧮 Find Eigenvalues & Eigenvectors:**  
   $$\[\text{Cov}(Z)v = \lambda v\]$$
5. **📈 Sort Eigenvalues (Descending):** Select top `k` eigenvectors → principal components.
6. **💡 Project Data:**  
   $$\[Z_{\text{reduced}} = Z \cdot W\]$$
   where `W` is the matrix of top eigenvectors.
7. **🎨 Visualize Results:**  
   - Data projected on new axes  
   - Cumulative explained variance plot

---

## 🧮 Mathematical Concepts

| Concept | Formula / Description |
|----------|------------------------|
| **Standardization** | $$\( Z = \frac{X - \mu}{\sigma} \)$$ |
| **Covariance Matrix** | $$\( \text{Cov}(Z) = \frac{1}{n-1} Z^T Z \)$$ |
| **Eigen Decomposition** | $$\( \text{Cov}(Z)v = \lambda v \)$$ |
| **Explained Variance Ratio** | $$\( r_i = \frac{\lambda_i}{\sum_j \lambda_j} \)$$ |
| **Projection** | $$\( Z_{\text{reduced}} = Z \cdot W \)$$ |

---

## 📊 Visualization Outputs

1. **Covariance Matrix**  
   Displays inter-feature relationships.

2. **Projected Data (2D Scatter Plot)**

Shows data in principal component space.

3. **Cumulative Explained Variance Plot**  
Illustrates how much information is retained with each component.

---

## 🧩 Dataset
**Mall_Customers.csv**  
Features used:
- `Annual Income (k$)`  
- `Spending Score (1–100)`

---

## 💻 Code Summary
```python
# 1️⃣ Standardize
X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)

# 2️⃣ Covariance
cov_matrix = np.cov(X_scaled.T)

# 3️⃣ Eigen Decomposition
eig_vals, eig_vecs = np.linalg.eig(cov_matrix)

# 4️⃣ Sort & Project
sorted_idx = np.argsort(eig_vals)[::-1]
W = eig_vecs[:, sorted_idx[:2]]
X_pca = np.dot(X_scaled, W)


📈 Results

PCA successfully reduced data to 2 principal components.

The first component captured most variance (customer purchasing power).

The second captured smaller localized spending variations.

Compared results matched perfectly with scikit-learn PCA output.



---

🚀 Future Improvements:-

Apply PCA on higher-dimensional datasets (e.g., image or text embeddings).

Combine PCA with clustering algorithms (KMeans, DBSCAN) for hybrid insights.

Extend to Kernel PCA for non-linear transformations.



---

🏷️ Author

Anshu Pandey
📊 From Scratch Implementation with Mathematical Insights & Visualizations


---

⭐ References

Scikit-learn PCA Documentation

Mall Customer Dataset (Kaggle)
