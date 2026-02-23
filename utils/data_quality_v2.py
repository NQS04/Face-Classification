import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# --- Landmark normalization utils ---
def normalize_landmarks(X):
    X = X.values if isinstance(X, pd.DataFrame) else X
    X = X.reshape((-1, 468, 3))
    X_norm = []
    for face in X:
        center = face[1, :2]
        face[:, :2] -= center
        
        left_eye = face[33, :2]
        right_eye = face[263, :2]
        eye_dist = np.linalg.norm(left_eye - right_eye)
        if eye_dist > 1e-6:
            face[:, :2] /= eye_dist
            
        dx, dy = right_eye - left_eye
        angle = np.arctan2(dy, dx)
        rot = np.array([[np.cos(-angle), -np.sin(-angle)], [np.sin(-angle), np.cos(-angle)]])
        face[:, :2] = face[:, :2] @ rot.T
        X_norm.append(face.flatten())
    return np.array(X_norm)

# --- Load Dataset ---
print("Loading data...")
file_path = 'processed_data/raw_landmarks_training_set.csv'
if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
    sys.exit(1)

train_df = pd.read_csv(file_path)
# Only use training set for quality check
X = train_df.drop(columns=['label'])
y = train_df['label']

print(f"Total samples: {len(X)}")

# --- Preprocessing ---
print("Normalizing Landmarks...")
X_norm = normalize_landmarks(X)

print("Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_norm)

# --- PCA Analysis (Variance) ---
print("Running PCA to check variance...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(f"Explained Variance Ratio (PC1, PC2): {pca.explained_variance_ratio_}")

# --- t-SNE Visualization ---
print("Running t-SNE (this might take a moment)...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# --- Plotting ---
plt.figure(figsize=(16, 7))

# Plot PCA
plt.subplot(1, 2, 1)
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y, alpha=0.6, palette='viridis')
plt.title('PCA Projection (Linear Separation Check)')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Plot t-SNE
plt.subplot(1, 2, 2)
sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=y, alpha=0.6, palette='viridis')
plt.title('t-SNE Projection (Non-linear Cluster Check)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
output_img = 'utils/data_quality_visualization.png'
plt.savefig(output_img, dpi=300, bbox_inches='tight')
print(f"Visualization saved to {output_img}")

# --- Silhouette Score ---
print("\nCalculating Silhouette Score...")
# Using PCA 50 components for robust scoring
pca_50 = PCA(n_components=50)
X_pca_50 = pca_50.fit_transform(X_scaled)
sil_score = silhouette_score(X_pca_50, y)

print(f"Silhouette Score: {sil_score:.4f}")
print("(Score range: -1 to 1. Near 0 or negative means overlapping clusters.)")

if sil_score < 0.1:
    print("\n>>> CONCLUSION: Very low Silhouette Score indicates heavy overlapping between classes.")
    print(">>> The labels are likely inconsistent or the landmarks alone cannot distinguish shape.")
else:
    print("\n>>> CONCLUSION: Silhouette Score is decent. Model tuning might still help.")
