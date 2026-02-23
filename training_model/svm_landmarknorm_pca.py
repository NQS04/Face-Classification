import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# --- Landmark normalization utils ---
def normalize_landmarks(X):
    # X: (n_samples, 1404) with columns x_0, y_0, z_0, ..., x_467, y_467, z_467
    X = X.values if isinstance(X, pd.DataFrame) else X
    X = X.reshape((-1, 468, 3))
    X_norm = []
    for face in X:
        # 1. Center: subtract nose tip (landmark 1)
        center = face[1, :2]  # (x, y) only
        face[:, :2] -= center
        # 2. Scale: distance between eyes (landmarks 33, 263)
        left_eye = face[33, :2]
        right_eye = face[263, :2]
        eye_dist = np.linalg.norm(left_eye - right_eye)
        if eye_dist > 1e-6:
            face[:, :2] /= eye_dist
        # 3. Rotate: align eyes horizontally
        dx, dy = right_eye - left_eye
        angle = np.arctan2(dy, dx)
        rot = np.array([[np.cos(-angle), -np.sin(-angle)], [np.sin(-angle), np.cos(-angle)]])
        face[:, :2] = face[:, :2] @ rot.T
        X_norm.append(face.flatten())
    return np.array(X_norm)

# --- Load data ---
train_df = pd.read_csv('processed_data/raw_landmarks_training_set.csv')
test_df = pd.read_csv('processed_data/raw_landmarks_testing_set.csv')

X_train = train_df.drop(columns=['label'])
y_train = train_df['label']
X_test = test_df.drop(columns=['label'])
y_test = test_df['label']

# --- Normalize landmarks ---
print('Normalizing landmarks...')
X_train_norm = normalize_landmarks(X_train)
X_test_norm = normalize_landmarks(X_test)

# --- Standardize (fit only on train) ---
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train_norm)
X_test_std = scaler.transform(X_test_norm)

# --- Reduce dimension (PCA) ---
pca = PCA(n_components=0.98, svd_solver='full', random_state=42)  # keep 98% variance
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
print(f'PCA reduced to {X_train_pca.shape[1]} components')

# --- Encode labels ---
label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_train)
y_test_enc = label_encoder.transform(y_test)

# --- Train SVM ---
svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
print('Training SVM...')
svm.fit(X_train_pca, y_train_enc)

# --- Evaluate ---
y_pred = svm.predict(X_test_pca)
acc = accuracy_score(y_test_enc, y_pred)
print(f'Accuracy on test set: {acc:.4f}')

report = classification_report(y_test_enc, y_pred, target_names=label_encoder.classes_)
print(report)

with open('model_report/svm_landmarknorm_pca_report.txt', 'w') as f:
    f.write(f'Accuracy: {acc:.4f}\n\n')
    f.write(report)

# --- Confusion matrix ---
cm = confusion_matrix(y_test_enc, y_pred)
plt.figure(figsize=(7,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix — SVM (landmark norm + PCA)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('model_report/svm_landmarknorm_pca_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()
