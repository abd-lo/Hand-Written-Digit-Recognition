import tensorflow as tf
from tensorflow.keras import layers, models


# Load MNIST handwritten digits dataset
print("Loading MNIST dataset...")

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


# Normalize pixel values to range 0-1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


# Build neural network model
model = models.Sequential([

    # Input image size: 28x28 pixels
    layers.Input(shape=(28, 28)),

    # Convert image into a single vector
    layers.Flatten(),

    # Hidden layer with 128 neurons
    layers.Dense(128, activation='relu'),

    # Prevent overfitting
    layers.Dropout(0.2),

    # Output layer for digits 0-9
    layers.Dense(10, activation='softmax')
])


# Configure model training
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# Train the model
print("Training model...")

model.fit(
    x_train,
    y_train,
    epochs=5,
    validation_split=0.1
)


# Evaluate model accuracy
test_loss, test_acc = model.evaluate(
    x_test,
    y_test,
    verbose=0
)

print(f"Test Accuracy: {test_acc * 100:.2f}%")


# Save trained model for prediction
model.save("digit_model.keras")

print("Model saved as digit_model.keras")