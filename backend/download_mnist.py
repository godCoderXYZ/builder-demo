import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.datasets import mnist

print("📥 Downloading and preprocessing MNIST data...")

# Download MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Cut down dataset size for memory constraints
x_train = x_train[:16000]
y_train = y_train[:16000]
x_test = x_test[:4000]
y_test = y_test[:4000]

# Preprocess
x_train = x_train / 255.0
x_test = x_test / 255.0
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

# Save as numpy files
# MAKE SURE THAT THIS FOLDER IS SAVED IN THE BACKEND DIRECTORY
os.makedirs('mnist_data', exist_ok=True)
np.save('mnist_data/x_train.npy', x_train)
np.save('mnist_data/y_train.npy', y_train) 
np.save('mnist_data/x_test.npy', x_test)
np.save('mnist_data/y_test.npy', y_test)

print("✅ MNIST data saved to mnist_data/ directory")
print(f"📊 Training data shape: {x_train.shape}")
print(f"📊 Test data shape: {x_test.shape}")
