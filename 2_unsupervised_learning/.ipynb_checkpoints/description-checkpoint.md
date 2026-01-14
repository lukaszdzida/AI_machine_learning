# Unsupervised Learning — Model Overview

This document summarizes several common unsupervised learning techniques with brief explanations, practical tips, example applications, and short scikit-learn code snippets.

Source: [Original notebook — about_models.ipynb](https://github.com/lukaszdzida/AI_ML_models/blob/c1099da398b688a17d6ad0fa9a5827de6218437f/2_unsupervised_learning/about_models.ipynb)

---

## Contents
- k-means clustering
- DBSCAN
- Principal Component Analysis (PCA)
- t-SNE
- Choosing the right technique
- Next steps & references

---

## k-means clustering

### What it is
k-means is a centroid-based clustering algorithm that partitions data into k clusters by minimizing within-cluster variance. It is simple and fast for medium-sized datasets.

### How it works (brief)
1. Initialize k centroids (random or using k-means++).
2. Assign each point to the nearest centroid.
3. Recompute centroids as the mean of assigned points.
4. Repeat assignment and update steps until convergence.

### Use case example
Customer segmentation in retail: group customers by spending patterns, frequency, and demographics. Segments can be used for tailored marketing (e.g., high-value frequent buyers vs. budget-conscious infrequent buyers).

### Pros / Cons
- Pros: Fast, easy to implement, interpretable cluster centroids.
- Cons: Requires choosing k, sensitive to initialization and scaling, assumes spherical clusters of similar size.

### Practical tips
- Standardize/normalize features before clustering.
- Use the elbow method, silhouette score, or domain knowledge to select k.
- Try k-means++ for better initial centroids.

### Scikit-learn example
```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(X)
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X_scaled)
centroids = kmeans.cluster_centers_
```

---

## DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

### What it is
DBSCAN clusters points based on density: regions with many nearby points become clusters, while points in low-density regions are treated as outliers.

### How it works (brief)
- Two parameters: `eps` (neighborhood radius) and `min_samples` (minimum points to form a dense region).
- Dense regions are expanded into clusters; sparse points are marked as noise.

### Use case example
Fraud detection in financial services: cluster normal transaction patterns and flag outliers (suspicious transactions) for further investigation.

### Pros / Cons
- Pros: Can find arbitrarily shaped clusters, robust to noise, no need to specify number of clusters.
- Cons: Sensitive to parameter choice, struggles with varying density clusters, `eps` doesn't scale well with high dimensions.

### Practical tips
- Use k-distance plot to choose `eps`.
- Consider dimensionality reduction (e.g., PCA) before DBSCAN on high-dimensional data.

### Scikit-learn example
```python
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(X)
db = DBSCAN(eps=0.5, min_samples=5).fit(X_scaled)
labels = db.labels_  # -1 means noise/outlier
```

---

## Principal Component Analysis (PCA)

### What it is
PCA is a linear dimensionality reduction technique that transforms data into orthogonal components ordered by explained variance. It is useful for compression, visualization, and noise reduction.

### How it works (brief)
PCA finds linear combinations of features (principal components) that capture the greatest variance in the data.

### Use case example
Genomics: datasets often have thousands of gene expression features. PCA reduces dimensionality to visualize gene expression patterns, identify correlated gene groups, and detect biological signals.

### Pros / Cons
- Pros: Fast, deterministic, reduces noise, helps visualization, preserves global structure.
- Cons: Linear method — cannot capture nonlinear structure; components can be hard to interpret biologically.

### Practical tips
- Scale features when they have different units.
- Inspect explained variance ratio to decide how many components to keep.
- Combine with clustering or downstream models.

### Scikit-learn example
```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
explained = pca.explained_variance_ratio_
```

---

## t-SNE (t-Distributed Stochastic Neighbor Embedding)

### What it is
t-SNE is a nonlinear dimensionality reduction technique designed for visualization. It preserves local neighborhood relationships in a low-dimensional embedding (typically 2D or 3D).

### How it works (brief)
t-SNE converts similarities between points to joint probabilities and tries to minimize the divergence between high-dimensional and low-dimensional similarities. It emphasizes preserving local structure.

### Use case example
Visualizing clusters in high-dimensional feature spaces (e.g., embeddings, gene expression, image features) to detect structure, clusters, or anomalies.

### Pros / Cons
- Pros: Excellent for visualizing complex local structure and revealing clusters.
- Cons: Computationally expensive for large datasets, results depend on hyperparameters (perplexity, learning rate), not deterministic, does not preserve global distances.

### Practical tips
- Run PCA to reduce to ~50 dimensions before t-SNE for speed and noise reduction.
- Try several perplexity values (5–50) and random seeds.
- Use Barnes-Hut or FFT implementations for larger datasets (e.g., scikit-learn’s `method='barnes_hut'` or openTSNE).

### Scikit-learn example
```python
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

X_pca = PCA(n_components=50, random_state=42).fit_transform(X_scaled)
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_pca)
```

---

## Choosing the right technique

Consider:
- Goal: visualization vs. clustering vs. outlier detection vs. dimensionality reduction.
- Data size and dimensionality: t-SNE is visualization-focused and expensive; PCA scales well.
- Cluster shape and density: k-means assumes spherical clusters; DBSCAN handles irregular shapes and noise.
- Need for interpretability: PCA components are linear combinations that can sometimes be interpreted; k-means centroids are easy to inspect.

A common workflow:
1. Preprocess and scale data.
2. Use PCA for initial dimensionality reduction and exploratory analysis.
3. Try clustering methods (k-means, DBSCAN) on reduced features.
4. Use t-SNE for final visualization of discovered clusters (with PCA pre-step for speed).

---

## Next steps & suggestions
- Add small visual examples (scatter plots of PCA/t-SNE embeddings with cluster labels).
- Include parameter-selection plots (elbow method, silhouette scores, DBSCAN k-distance).
- Convert this summary into a runnable notebook with synthetic or small real datasets for demonstrations.

---

## References
- scikit-learn documentation: clustering, PCA, t-SNE.
- For further reading, check original notebook: [about_models.ipynb](https://github.com/lukaszdzida/AI_ML_models/blob/c1099da398b688a17d6ad0fa9a5827de6218437f/2_unsupervised_learning/about_models.ipynb)
