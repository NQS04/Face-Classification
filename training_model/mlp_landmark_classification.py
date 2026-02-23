import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

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

# --- Load data ---
print("Loading data...")
train_df = pd.read_csv('processed_data/raw_landmarks_training_set.csv')
test_df = pd.read_csv('processed_data/raw_landmarks_testing_set.csv')

X_train_raw = train_df.drop(columns=['label'])
y_train = train_df['label']
X_test_raw = test_df.drop(columns=['label'])
y_test = test_df['label']

# --- Normalize & Scale ---
print("Normalizing Landmarks...")
X_train_norm = normalize_landmarks(X_train_raw)
X_test_norm = normalize_landmarks(X_test_raw)

print("Scaling features (StandardScaler)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_norm)
X_test_scaled = scaler.transform(X_test_norm)

# --- Encode Labels ---
label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_train)
y_test_enc = label_encoder.transform(y_test)

# --- Define and Train MLP ---
print("\nTraining MLP Classifier (Neural Network)...")
# Architecture: 3 hidden layers (512, 256, 128)
# Early stopping enabled to prevent overfitting
mlp = MLPClassifier(
    hidden_layer_sizes=(512, 256, 128),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    batch_size=64,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=42,
    verbose=True
)

mlp.fit(X_train_scaled, y_train_enc)

# --- Evaluate ---
y_pred = mlp.predict(X_test_scaled)
acc = accuracy_score(y_test_enc, y_pred)
print(f"\nAccuracy on test set: {acc:.4f}")

report = classification_report(y_test_enc, y_pred, target_names=label_encoder.classes_)
print(report)

# Save report
with open('model_report/mlp_classifier_report.txt', 'w') as f:
    f.write(f"MLP Classifier Accuracy: {acc:.4f}\n\n")
    f.write(report)

# Confusion Matrix
cm = confusion_matrix(y_test_enc, y_pred)
plt.figure(figsize=(7,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix — MLP Classifier')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('model_report/mlp_classifier_heatmap.png', dpi=300, bbox_inches='tight')
print("Confusion matrix saved to model_report/mlp_classifier_heatmap.png")
