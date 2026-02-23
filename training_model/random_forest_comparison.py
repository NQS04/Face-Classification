import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# --- Landmark normalization utils (same as before) ---
def normalize_landmarks(X):
    # X: (n_samples, 1404)
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

# --- Feature Engineering (same function as before for comparison) ---
def extract_geometric_features(X):
    # This expects un-flattened or handles flattening inside. 
    # normalize_landmarks returns flattened array.
    # Let's reshape inside.
    X_vals = X.values if isinstance(X, pd.DataFrame) else X
    n_samples = X_vals.shape[0]
    X_reshaped = X_vals.reshape(n_samples, 468, 3)
    
    features = []
    for i in range(n_samples):
        face = X_reshaped[i]
        
        top = face[10]
        bottom = face[152]
        left_cheek = face[234]; right_cheek = face[454]
        left_jaw = face[132]; right_jaw = face[361]
        left_forehead = face[103]; right_forehead = face[332]
        
        face_height = np.linalg.norm(top - bottom)
        cheek_width = np.linalg.norm(right_cheek - left_cheek)
        jaw_width = np.linalg.norm(right_jaw - left_jaw)
        forehead_width = np.linalg.norm(right_forehead - left_forehead)
        
        eps = 1e-6
        sample_feats = [
            cheek_width / (face_height + eps),
            jaw_width / (cheek_width + eps),
            forehead_width / (cheek_width + eps),
            jaw_width / (forehead_width + eps),
            # ratio_jaw_chin_height
            ((np.linalg.norm(left_jaw - bottom) + np.linalg.norm(right_jaw - bottom)) / 2) / (face_height + eps)
        ]
        features.append(sample_feats)
    return np.array(features)

# --- Load data ---
print("Loading data...")
train_df = pd.read_csv('processed_data/raw_landmarks_training_set.csv')
test_df = pd.read_csv('processed_data/raw_landmarks_testing_set.csv')

X_train_raw = train_df.drop(columns=['label'])
y_train = train_df['label']
X_test_raw = test_df.drop(columns=['label'])
y_test = test_df['label']

label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_train)
y_test_enc = label_encoder.transform(y_test)

def train_and_eval(X_train, X_test, name):
    print(f"\nTraining Random Forest on {name}...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train_enc)
    
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test_enc, y_pred)
    print(f"Accuracy ({name}): {acc:.4f}")
    print(classification_report(y_test_enc, y_pred, target_names=label_encoder.classes_))
    return rf, acc

# Experiment 1: Engineered Features
X_train_eng = extract_geometric_features(X_train_raw)
X_test_eng = extract_geometric_features(X_test_raw)
train_and_eval(X_train_eng, X_test_eng, "Engineered Features")

# Experiment 2: Full Normalized Landmarks
print("\nNormalizing landmarks for full feature set...")
X_train_norm = normalize_landmarks(X_train_raw)
X_test_norm = normalize_landmarks(X_test_raw)
# Random Forest handles high dimensions reasonably well, no PCA needed for a first try
train_and_eval(X_train_norm, X_test_norm, "Full Normalized Landmarks")

# Experiment 3: Raw Landmarks (The baseline "Do Nothing")
# Just to see if normalization is actually hurting or helping for RF
train_and_eval(X_train_raw, X_test_raw, "Raw Landmarks")
