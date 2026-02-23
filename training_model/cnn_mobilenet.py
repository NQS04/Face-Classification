import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# --- Config ---
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 15
DATA_DIR = "FaceShape Dataset"

def load_data(data_dir, split_type="training_set"):
    images = []
    labels = []
    path = os.path.join(data_dir, split_type)
    
    if not os.path.exists(path):
        print(f"Error: Path not found {path}")
        return np.array([]), np.array([])
        
    print(f"Loading {split_type}...")
    
    for label in os.listdir(path):
        label_path = os.path.join(path, label)
        if not os.path.isdir(label_path):
            continue
            
        # Limit per class for speed testing if needed, currently loading all
        cnt = 0
        for img_name in os.listdir(label_path):
            try:
                img_path = os.path.join(label_path, img_name)
                # Read image
                img = cv2.imread(img_path)
                if img is None: continue
                
                # Resize and Convert to RGB
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                images.append(img)
                labels.append(label)
                cnt += 1
            except Exception as e:
                pass
        print(f" - Loaded {cnt} images for class {label}")
                
    return np.array(images), np.array(labels)

# --- Main Pipeline ---
def main():
    # 1. Load Data
    X_train, y_train_labels = load_data(DATA_DIR, "training_set")
    X_test, y_test_labels = load_data(DATA_DIR, "testing_set")

    if len(X_train) == 0:
        print("No training data found!")
        return

    # 2. Preprocess
    # Normalize pixel values to [0, 1]
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # Encode Labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train_labels)
    y_test_enc = le.transform(y_test_labels)
    
    # One-hot encoding
    num_classes = len(np.unique(y_train_enc))
    y_train_cat = to_categorical(y_train_enc, num_classes)
    y_test_cat = to_categorical(y_test_enc, num_classes)
    
    print(f"Training shape: {X_train.shape}, Labels: {y_train_cat.shape}")
    print(f"Testing shape: {X_test.shape}, Labels: {y_test_cat.shape}")

    # 3. Build Model (Transfer Learning with MobileNetV2)
    # MobileNetV2 is lightweight and good for simple image classification tasks
    print("\nBuilding MobileNetV2 model...")
    base_model = MobileNetV2(
        weights='imagenet', 
        include_top=False, 
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    # Freeze base model first
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # 4. Train
    print("\nStarting Training...")
    history = model.fit(
        X_train, y_train_cat,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test_cat),
        verbose=1
    )

    # 5. Evaluate
    print("\nEvaluating on Test Set...")
    loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"Test Accuracy: {acc:.4f}")
    
    # Save Report
    with open('model_report/cnn_mobilenet_report.txt', 'w') as f:
        f.write(f"CNN (MobileNetV2) Accuracy: {acc:.4f}\n")
    
    # Plot history
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('CNN Training History')
    plt.legend()
    plt.savefig('model_report/cnn_training_history.png')
    print("Training history saved.")

if __name__ == "__main__":
    main()
