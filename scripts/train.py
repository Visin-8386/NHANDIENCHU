import os
import numpy as np
from tensorflow import keras
from src.models.model import build_model, save_model

# Tải dataset MNIST
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Chuẩn hóa dữ liệu (0-1) và reshape cho CNN
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = np.expand_dims(x_train, axis=-1)  # (60000, 28, 28, 1)
x_test = np.expand_dims(x_test, axis=-1)

# Tăng cường dữ liệu (Data Augmentation)
datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)
datagen.fit(x_train)

# Xây dựng mô hình CNN tối ưu
model = build_model(input_shape=(28, 28, 1), num_classes=10)

# Dừng sớm nếu không cải thiện sau 5 epoch
early_stopping = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)

# Huấn luyện mô hình (với data augmentation) và lưu lịch sử huấn luyện
history = model.fit(datagen.flow(x_train, y_train, batch_size=64),
                    epochs=30,
                    validation_data=(x_test, y_test),
                    callbacks=[early_stopping])

# Đánh giá độ chính xác trên tập kiểm tra
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"🎯 Độ chính xác trên tập kiểm tra: {test_acc:.4f}")
# Biểu ĐỒ
from src.visualization.visualize import plot_training_history
plot_training_history(history)
# Lưu mô hình
model_path = "saved_models/digit_recognition_optimized.h5"
os.makedirs(os.path.dirname(model_path), exist_ok=True)
save_model(model, model_path)
print(f"Model đã lưu tại {model_path}")

