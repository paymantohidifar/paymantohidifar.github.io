---
title: "A Short Note on Principal Component Analysis (PCA) in Bulk RNA-seq"
date: June 2025
description: A concise walkthrough of the linear algebra behind PCA for bulk RNA-seq, from variance-stabilizing transforms through SVD to interpreting loadings and variance explained, with parallel Python and R implementations.
tags:
  - Bioinformatics
  - Unsupervised Machine Learning
  - RNA-seq
---
## **Introduction**

When engineering high-throughput bioinformatics pipelines, we routinely encounter the curse of dimensionality ($p \gg n$), where $p$ represents experimental features and $n$ represents biological replicates or cell lines. A typical bulk RNA-seq experiment profiles the expression of $p \approx 20,000$ genes across a small cohort of $n \approx 20$ samples. Visualizing, parsing, or training machine learning models directly on this raw 20,000-dimensional space introduces a massive risk of overfitting, multicollinearity, and data leakage.

Principal Component Analysis (PCA) is our primary tool for unsupervised dimensionality reduction during Exploratory Data Analysis (EDA). It collapses thousands of correlated genetic features into a low-dimensional space of orthogonal axes while preserving the maximum possible global variance.

---

## **The Role of PCA in Bulk RNA-seq**

PCA acts as an essential quality control gatekeeper. By compressing the full transcriptome down to a few orthogonal axes—called **Principal Components (PCs)**—it maps sample relationships onto an interpretable 2D or 3D scatter plot. When evaluating a PCA projection, we look for two distinct signals:

* **Biological Signal:** If $PC1$ (the axis capturing the highest variance) cleanly separates samples by "Treatment" vs. "Control", the experimental perturbation successfully remodeled the cellular transcriptome over baseline biological noise.
* **Technical Noise & Batch Effects:** If samples cluster instead by "Sequencing Date", "Flow Cell Lane", or "Extraction Batch", a technical confounder is dominating the biological signal. In production pipelines, this requires halting downstream differential expression analysis until a batch-aware correction algorithm (e.g., `ComBat-seq` or explicit multi-factor design matrix modeling in `DESeq2`) is integrated.

---

## **The Prerequisite: Why VST Matters**

PCA is a *variance-maximizing* algorithm. In raw RNA-seq read counts, variance scales quadratically with mean expression due to the inherently overdispersed nature of transcriptomic count data:

$$\sigma^2 = \mu + \alpha \mu^2$$

*(Where $\alpha$ represents the empirical dispersion parameter).*

Without applying a **Variance Stabilizing Transformation (VST)**, a handful of hyper-expressed housekeeping genes (like Actin or GAPDH) will completely dominate the principal components simply because their absolute numerical fluctuations are massive—even if their relative fold-changes across experimental groups are entirely static.

VST normalizes the data by applying a monotonic transformation $g(x)$ derived from the integral of the reciprocal of the overdispersed standard deviation:

$$g(x) = \int_0^x \frac{1}{\sqrt{t + \alpha t^2}}dt$$

This mathematical transformation stabilizes the variance across the entire dynamic range of the transcriptome, ensuring that the final spatial distance between samples on a PCA plot accurately reflects global biological variation rather than absolute expression scale. For more details, refer to my short note on [heteroscedasticity in bulk RNAseq](/blogs/vst.html) or for comprehensive deep dives, review the mathematical frameworks outlined in [DESeq2](https://link.springer.com/article/10.1186/s13059-014-0550-8) and [PyDESeq2](https://academic.oup.com/bioinformatics/article/39/9/btad547/7260507).

---

## **The Linear Algebra of PCA**

To compute PCA, we rotate our high-dimensional coordinate system to align the new axes with the maximum directions of data spread.

### **Step A: Centering and Covariance**

Given a VST-transformed expression matrix $X \in \mathbb{R}^{n \times p}$ ($n$ samples, $p$ genes), we calculate the mean expression of each gene $\mu_j$ and center the data matrix:

$$\tilde{X} = X - \mathbf{1}\mu^T$$

We then derive the $p \times p$ **Covariance Matrix** ($\Sigma$), tracking how every pair of genes varies together across samples:

$$\Sigma = \frac{1}{n-1}\tilde{X}^T \tilde{X}$$

### **Step B: Eigen-decomposition**

We solve for the eigenvalues ($\lambda$) and eigenvectors ($\nu$) of our sample covariance matrix:

$$\Sigma \nu_i = \lambda_i \nu_i$$

* **Eigenvectors ($\nu$):** These define the directions of the new orthogonal axes (the Principal Components).
* **Eigenvalues ($\lambda$):** These quantify the absolute variance captured along each component. We sort them in descending order ($\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_p$) to isolate the dominant signals.

### **Step C (Production Scale): Singular Value Decomposition (SVD)**

In real-world bioinformatics workflows, computing a $20,000 \times 20,000$ covariance matrix $\Sigma$ is highly inefficient. Instead, we compute **Singular Value Decomposition (SVD)** directly on our centered expression matrix $\tilde{X}$:

$$\tilde{X} = USV^T$$

* $V^T \in \mathbb{R}^{n \times p}$ (or $V \in \mathbb{R}^{p \times n}$ for economy SVD): Contains the **Loadings** (the directional weights $w_j$ of each gene for each PC). The larger the absolute loading value $|w_j|$, the more that gene drives the separation of samples along that specific axis.
* $S$: A diagonal matrix containing singular values ($s_i$), directly related to our covariance eigenvalues by $\lambda_i = \frac{s_i^2}{n-1}$.
* $U \in \mathbb{R}^{n \times n}$: Left-singular vectors, which map directly to our final sample coordinates.

> **Note**: For foundational linear algebra references on SVD mechanics, watch [Gilbert Strang's lecture on MIT OpenCourseWare](https://www.youtube.com/watch?v=rYz83XPxiZo).

---

## **Implementation: Python vs. R**

In production-grade engineering, it is standard practice to filter the input matrix to the top 500–1000 most variable genes. This mitigates transcriptional background noise and accelerates convergence. Below are fully unified, equivalent workflows in both languages. For more hands-on experience, check out my [notebook](https://github.com/paymantohidifar/bulk-rnaseq-analysis-workflows/blob/dev-python-wf/python-wf/bonus/notebooks/clustring.ipynb).

#### **Python Implementation**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Assuming 'vsd_df' is a pandas DataFrame [Samples x Genes] of VST counts
# Filter for top 500 most variable genes
gene_vars = vsd_df.var(axis=0)
top_genes = gene_vars.nlargest(500).index
vsd_subset = vsd_df[top_genes]

# --- Standard PCA via Scikit-Learn (Handles Centering Automatically) ---
pca = PCA(n_components=2)
X_pca = pca.fit_transform(vsd_subset)

# Extract and sort gene loadings for PC1
pc1_loadings = pd.Series(np.abs(pca.components_[0]), index=top_genes)
print("Top PC1 Loadings (scikit-learn):")
print(pc1_loadings.sort_values(ascending=False).head())

# --- Low-Level SVD Equivalence (Requires Manual Centering) ---
X_centered = StandardScaler(with_std=False).fit_transform(vsd_subset)
U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
eigenvectors = Vt.T  # Dimensions: [Genes x Components]

# --- Visualization: Biplot Construction ---
plt.figure(figsize=(10, 7))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c='skyblue', edgecolor='k', alpha=0.7, label='Samples')

# Overlay top 5 driver gene vectors (loadings)
top_drivers = pc1_loadings.nlargest(5).index
for gene in top_drivers:
    gene_idx = vsd_subset.columns.get_loc(gene)
    # Scale up vector lengths for visual pop on the score plot
    plt.arrow(0, 0, pca.components_[0, gene_idx]*3, pca.components_[1, gene_idx]*3, 
              color='red', head_width=0.05, alpha=0.8)
    plt.text(pca.components_[0, gene_idx]*3.5, pca.components_[1, gene_idx]*3.5, 
             gene, color='darkred', ha='center', va='center')

plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.grid(True, linestyle='--')
plt.show()

```

#### **R Implementation**

```r
library(DESeq2)
library(factoextra)
library(matrixStats)

# Assuming 'vsd' is a VST-transformed DESeq2 object
# Filter for top 500 most variable genes
rv <- rowVars(assay(vsd))
select <- order(rv, decreasing = TRUE)[seq_len(min(500, length(rv)))]
vsd_subset <- t(assay(vsd)[select, ])

# --- Standard PCA via prcomp() (Handles Centering Automatically) ---
pca_res <- prcomp(vsd_subset, rank. = 2)

# Extract and sort gene loadings for PC1
pc1_loadings <- sort(abs(pca_res$rotation[, 1]), decreasing = TRUE)
print("Top PC1 Loadings (prcomp):")
print(head(pc1_loadings))

# --- Low-Level SVD Equivalence (Requires Manual Centering) ---
X_centered <- scale(vsd_subset, center = TRUE, scale = FALSE)
svd_res <- svd(X_centered)
eigenvectors <- svd_res$v  # Dimensions: [Genes x Components]

# --- Visualization: Interactive Biplot ---
fviz_pca_biplot(pca_res, 
                geom.ind = "point", 
                select.var = list(contrib = 10), # Overlay top 10 driver genes
                col.ind = "cos2")

```

---

## **Projection and Dimensionality Reduction**

The physical projection step is modeled as a linear transformation via matrix multiplication. If $\tilde{X} \in \mathbb{R}^{n \times p}$ is our centered data matrix and $V_k \in \mathbb{R}^{p \times k}$ represents the matrix containing the first $k$ eigenvectors (loadings), our low-dimensional score coordinate matrix $Z \in \mathbb{R}^{n \times k}$ is computed as:

$$Z = \tilde{X}V_k$$

To generate a standard 2D PCA plot, we isolate the first two components ($k = 2$). Each row $i$ in $Z$ provides the precise $(x, y)$ coordinates mapping Sample $i$ onto our exploratory low-dimensional space.

---

## **Summary of Core Performance Metrics**

* **Proportion of Variance Explained (PVE):** Defined as $\frac{\lambda_k}{\sum_{i=1}^{p} \lambda_i}$. If $PC1$ captures a dominant proportion of the global variance (e.g., $>50\%$) and maps cleanly to our phenotypic classes, our experimental design yields a highly robust signal-to-noise ratio.
* **Loadings Matrix ($V$):** Evaluating the magnitude and sign of values within $V$ isolates the explicit gene drivers separating our sample groups. Genes with high absolute weights along a separating PC coordinate represent direct candidates for downstream mechanistic validation.