import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import ImageFile

# Ensure truncated images don't crash the generator
ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- CONFIG ---
IMG_SIZE = 224
BATCH_SIZE = 32
DATA_DIR = "FaceShape Dataset"

def extract_features(model, generator, sample_count):
    features = np.zeros(shape=(sample_count, 7, 7, 512)) # VGG16 output shape for 224x224
    labels = np.zeros(shape=(sample_count, 5)) # 5 classes one-hot (generator outputs one-hot)
    
    i = 0
    # Generator yields (inputs, targets)
    # We use shuffle=False, so order is preserved
    print("Extracting features (this may take a while)...")
    for inputs_batch, labels_batch in generator:
        features_batch = model.predict(inputs_batch, verbose=0)
        
        batch_size_actual = inputs_batch.shape[0]
        
        # Check if we exceed expected count
        if i + batch_size_actual > sample_count:
            # Take only remaining
            remaining = sample_count - i
            features[i : i + remaining] = features_batch[:remaining]
            labels[i : i + remaining] = labels_batch[:remaining]
            break
        else:
            features[i : i + batch_size_actual] = features_batch
            labels[i : i + batch_size_actual] = labels_batch
            
        i += batch_size_actual
        if i % (BATCH_SIZE * 5) == 0:
            print(f"Processed {i}/{sample_count} images")
            
        if i >= sample_count:
            break
            
    # Flatten features: (N, 7, 7, 512) -> (N, 25088)
    return features.reshape(sample_count, -1), np.argmax(labels, axis=1)

def main():
    # 1. Load Pre-trained VGG16
    print("Loading VGG16 model (headless)...")
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # 2. Data Generators (Shuffle MUST be False to align features/labels correctly if we need to map back, 
    # but here we extract (X, y) pairs directly from loop, so shuffle=False ensures stability)
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_dir = os.path.join(DATA_DIR, 'training_set')
    test_dir = os.path.join(DATA_DIR, 'testing_set')

    # Get count first
    train_gen_dummy = datagen.flow_from_directory(train_dir, class_mode='categorical')
    test_gen_dummy = datagen.flow_from_directory(test_dir, class_mode='categorical')
    
    n_train = train_gen_dummy.samples
    n_test = test_gen_dummy.samples
    
    # Real generators
    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False 
    )

    test_generator = datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    class_names = list(train_generator.class_indices.keys())
    print(f"Classes: {class_names}")

    # 3. Extract Features
    print("\n--- Processing Training Set ---")
    X_train, y_train = extract_features(base_model, train_generator, n_train)
    
    print("\n--- Processing Testing Set ---")
    X_test, y_test = extract_features(base_model, test_generator, n_test)
    
    print(f"\nFeature extraction complete.")
    print(f"Train features shape: {X_train.shape}")
    print(f"Test features shape: {X_test.shape}")

    # 4. Train SVM
    print("\nTraining SVM on VGG16 features...")
    # Use RBF kernel, maybe adjust C if needed. Since dim is high (25088), LinearSVC might be faster,
    # but let's stick to SVC RBF or Poly. High dim usually linear works strictly better?
    # Actually, with n_features > n_samples, Linear Kernel is recommended.
    
    svm = SVC(kernel='linear', C=0.01, random_state=42) # Linear often better for High-Dim CNN features
    svm.fit(X_train, y_train)

    # 5. Evaluate
    print("Evaluating...")
    y_pred = svm.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\nHybrid VGG16-SVM Accuracy: {acc:.4f}")
    
    report = classification_report(y_test, y_pred, target_names=class_names)
    print(report)
    
    # Save Report
    with open('model_report/hybrid_vgg16_svm_report.txt', 'w') as f:
        f.write(f"Hybrid VGG16 (ImageNet) + SVM Accuracy: {acc:.4f}\n\n")
        f.write(report)
        
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='OrRd', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix: Hybrid VGG16 + SVM')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('model_report/hybrid_vgg16_svm_heatmap.png')
    print("Saved heatmap.")

if __name__ == "__main__":
    main()
