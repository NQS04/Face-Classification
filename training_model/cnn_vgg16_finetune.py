import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt

# --- CONFIGURATIONS ---
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 40  # Tang epoch len
LEARNING_RATE = 1e-4 # Learning rate nho cho fine-tuning
DATA_DIR = "FaceShape Dataset"

def build_model(num_classes):
    # Load VGG16 pre-trained on ImageNet
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

    # Unfreeze the last block (Block 5) for fine-tuning
    # Blocks 1-4 remain frozen to keep generic features (lines, edges)
    for layer in base_model.layers:
        if 'block5' in layer.name:
            layer.trainable = True
        else:
            layer.trainable = False
            
    # Add custom classification head
    x = base_model.output
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x) # Strong dropout to prevent overfitting
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def main():
    # 1. Data Generators (with Augmentation for Training)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,      # Xoay nhe
        width_shift_range=0.1,  # Dich chuyen nhe
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,   # Lat ngang la hop ly voi mat nguoi
        fill_mode='nearest'
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    print("Loading Data Generators...")
    train_generator = train_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'training_set'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    validation_generator = test_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'testing_set'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    num_classes = len(train_generator.class_indices)
    print(f"Detected {num_classes} classes: {train_generator.class_indices}")

    # 2. Build Model
    print("Building VGG16 Fine-tuning Model...")
    model = build_model(num_classes)
    
    # 3. Compile
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()

    # 4. Callbacks
    checkpoint = ModelCheckpoint(
        'training_model/best_vgg16_model.keras', 
        monitor='val_accuracy', 
        save_best_only=True, 
        mode='max',
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.2, 
        patience=5, 
        min_lr=1e-6,
        verbose=1
    )
    
    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    # 5. Train
    print("\nStarting Training (Fine-tuning)...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=[checkpoint, reduce_lr, early_stop],
        verbose=1
    )

    # 6. Final Evaluation
    print("\nEvaluating best model...")
    loss, acc = model.evaluate(validation_generator)
    print(f"Final Test Accuracy: {acc:.4f}")
    
    with open('model_report/cnn_vgg16_report.txt', 'w') as f:
        f.write(f"VGG16 Fine-tuned Accuracy: {acc:.4f}\n")

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('VGG16 Fine-tuning History')
    plt.legend()
    plt.savefig('model_report/vgg16_training_history.png')
    print("Saved history plot.")

if __name__ == "__main__":
    main()
