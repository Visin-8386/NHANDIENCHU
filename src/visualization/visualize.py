import matplotlib.pyplot as plt

def show_image(image, title=None):
    plt.imshow(image.squeeze(), cmap='gray')
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Độ chính xác (training)')
    plt.plot(history.history['val_accuracy'], label='Độ chính xác (validation)')
    plt.title('Biểu đồ độ chính xác')
    plt.xlabel('Epoch')
    plt.ylabel('Độ chính xác')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Mất mát (training)')
    plt.plot(history.history['val_loss'], label='Mất mát (validation)')
    plt.title('Biểu đồ mất mát')
    plt.xlabel('Epoch')
    plt.ylabel('Giá trị mất mát')
    plt.legend()
    plt.show()
