import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import matplotlib.pyplot as plt

# Load Data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build Model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train
model.fit(x_train, y_train, epochs=5, validation_split=0.1)

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'\nTest accuracy: {test_acc:.4f}')

# Predict on test images
import random
from PIL import Image

# Random test prediction
n = random.randint(0, len(x_test) - 1)
prediction = model.predict(x_test[n:n+1])
predicted_label = prediction.argmax()

plt.figure(figsize=(8, 8), dpi=100)
plt.imshow(x_test[n], cmap='gray', interpolation='none')
plt.title(f"Prediction: {predicted_label} | Actual: {y_test[n]}", fontsize=18)
plt.axis('off')
plt.grid(False)
plt.tight_layout(pad=3.0)
plt.show()
