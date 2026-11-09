import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

print("Loading your custom collected sign data...")

# Load your custom saved landmarks data
X = np.load("sign_data.npy")
y = np.load("sign_labels.npy")

print(f"Data shape: {X.shape}, Labels shape: {y.shape}")

# Build a clean Neural Network for MediaPipe landmarks (63 inputs: 21 landmarks * 3 coordinates x,y,z)
model = models.Sequential([
    layers.Input(shape=(X.shape[1],)),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(64, activation="relu"),
    layers.Dense(26, activation="softmax") # 26 classes for A-Z
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Train the model on your custom data
model.fit(X, y, epochs=25, validation_split=0.2, batch_size=32)

# Save the newly trained custom model
model.save("sign_model.keras")
print("Successfully trained and saved as sign_model.keras!")
