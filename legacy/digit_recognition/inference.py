import numpy as np
import tensorflow as tf
from tensorflow import keras
import random
import matplotlib.pyplot as plt

# Load model đã huấn luyện
model_path = "saved_models\digit_recognition_optimized_colab.h5"
model = tf.keras.models.load_model(model_path)
# Lấy tập test MNIST
mnist = keras.datasets.mnist
(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test.astype('float32') / 255.0
x_test = np.expand_dims(x_test, axis=-1)  # (10000, 28, 28, 1)

# Lấy ngẫu nhiên 10 chỉ số
indices = random.sample(range(len(x_test)), 10)

plt.figure(figsize=(15, 4))
for i, idx in enumerate(indices):
    img = x_test[idx]
    label = y_test[idx]
    img_input = np.expand_dims(img, axis=0)  # (1,28,28,1)
    pred = model.predict(img_input, verbose=0)
    pred_digit = int(np.argmax(pred))
    plt.subplot(2, 5, i+1)
    plt.imshow(img.squeeze(), cmap='gray')
    plt.title(f"Label: {label}\nPred: {pred_digit}")
    plt.axis('off')
plt.tight_layout()
plt.show()
