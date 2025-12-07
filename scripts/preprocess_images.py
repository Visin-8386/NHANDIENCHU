import os
import cv2
from src.data import preprocessing

def process_and_save_images(raw_dir, processed_dir):
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    img_files = [f for f in os.listdir(raw_dir) if f.lower().endswith(('.jpg', '.png'))]
    for fname in img_files:
        raw_path = os.path.join(raw_dir, fname)
        img = cv2.imread(raw_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f'Không đọc được file: {fname}')
            continue
        # Tiền xử lý đầy đủ: làm mờ, enhance nền, tăng nét, resize, chuẩn hóa
        blurred = preprocessing.apply_gaussian_blur(img)
        enhanced = preprocessing.enhance_black_background(blurred)
        thickened = preprocessing.increase_stroke_thickness(enhanced)
        processed = preprocessing.resize_and_pad(thickened, size=28, padding=4)
        # Lưu ảnh đã xử lý sang processed_dir
        processed_path = os.path.join(processed_dir, fname)
        cv2.imwrite(processed_path, processed)
        print(f'Đã xử lý và lưu: {processed_path}')

if __name__ == '__main__':
    raw_dir = 'data/raw'
    processed_dir = 'data/processed'
    process_and_save_images(raw_dir, processed_dir)
