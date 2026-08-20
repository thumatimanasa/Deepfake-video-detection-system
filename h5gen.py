import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Build a simple CNN model
def build_model():
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Binary classification (Real or Fake)
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create and save model
model = build_model()
model.save("deepfake_model_new1.h5")

print("Model saved successfully as deepfake_model_new.h5!")
