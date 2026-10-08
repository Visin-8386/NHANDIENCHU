import numpy as np
import cv2
import base64
from sklearn.cluster import DBSCAN
# Hàm chuyển ảnh numpy array sang chuỗi base64 để hiển thị trên web
def image_to_base64(img: np.ndarray) -> str:
    ret, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')

# Hàm áp dụng Gaussian Blur
def apply_gaussian_blur(image, kernel_size=(5, 5), sigma=0):
    return cv2.GaussianBlur(image, kernel_size, sigma)

# Hàm cải thiện nền đen
def enhance_black_background(img_gray):
    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    threshold, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    pixels_below_threshold = np.sum(hist[:int(threshold)])
    pixels_above_threshold = np.sum(hist[int(threshold):])
    if pixels_above_threshold > pixels_below_threshold:
        binary = cv2.bitwise_not(binary)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    return binary

# Hàm thay đổi kích thước và thêm padding
def resize_and_pad(roi, size=28, padding=5):
    h, w = roi.shape
    if h > w:
        new_h = size - 2 * padding
        new_w = int(w * (new_h / h))
    else:
        new_w = size - 2 * padding
        new_h = int(h * (new_w / w))
    resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size), dtype=np.uint8)
    x_offset = (size - new_w) // 2
    y_offset = (size - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas


def group_boxes(digit_boxes, eps=30, min_samples=1):
    if not digit_boxes:
        return []
    centers_y = np.array([y + h / 2 for x, y, w, h in digit_boxes]).reshape(-1, 1)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(centers_y)
    labels = clustering.labels_
    rows = {}
    for i, label in enumerate(labels):
        if label not in rows:
            rows[label] = []
        rows[label].append(digit_boxes[i])
    if -1 in rows:
        del rows[-1]
    sorted_rows = sorted(rows.items(), key=lambda item: np.mean([b[1] + b[3] / 2 for b in item[1]]))
    for _, row in sorted_rows:
        row.sort(key=lambda b: b[0])
    sorted_boxes = [box for _, row in sorted_rows for box in row]
    return sorted_boxes

# Hàm làm dày nét chữ
def increase_stroke_thickness(roi, dilation_kernel_size=(5, 5), dilation_iterations=3, padding=10):
    dilation_kernel = np.ones(dilation_kernel_size, np.uint8)
    roi_dilated = cv2.dilate(roi, dilation_kernel, iterations=dilation_iterations)
    roi_padded = cv2.copyMakeBorder(roi_dilated, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)
    return roi_padded

# Hàm tiền xử lý ảnh hoàn chỉnh
def preprocess_image(image):
    steps = {}
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    steps["gray"] = image_to_base64(img_gray)
    img_blurred = apply_gaussian_blur(img_gray, kernel_size=(5, 5))
    steps["blurred"] = image_to_base64(img_blurred)
    img_optimized = enhance_black_background(img_blurred)
    steps["optimized_background"] = image_to_base64(img_optimized)
    contours, _ = cv2.findContours(img_optimized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_with_contours = cv2.cvtColor(img_optimized.copy(), cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img_with_contours, contours, -1, (0, 255, 0), 2)
    steps["contours"] = image_to_base64(img_with_contours)
    digit_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 5 and h > 5 and cv2.contourArea(cnt) > 25:
            pad = 5
            x_pad = max(0, x - pad)
            y_pad = max(0, y - pad)
            w_pad = w + 2 * pad
            h_pad = h + 2 * pad
            if x_pad + w_pad > img_optimized.shape[1]:
                w_pad = img_optimized.shape[1] - x_pad
            if y_pad + h_pad > img_optimized.shape[0]:
                h_pad = img_optimized.shape[0] - y_pad
            digit_boxes.append((x_pad, y_pad, w_pad, h_pad))
    img_with_boxes = cv2.cvtColor(img_optimized.copy(), cv2.COLOR_GRAY2BGR)
    for (x, y, w, h) in digit_boxes:
        cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 0, 255), 2)
    steps["boxes"] = image_to_base64(img_with_boxes)
    
    digit_boxes = group_boxes(digit_boxes, eps=30, min_samples=1)
    
    img_with_sorted_boxes = cv2.cvtColor(img_optimized.copy(), cv2.COLOR_GRAY2BGR)
    for i, (x, y, w, h) in enumerate(digit_boxes):
        cv2.rectangle(img_with_sorted_boxes, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(img_with_sorted_boxes, str(i), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    steps["sorted_boxes"] = image_to_base64(img_with_sorted_boxes)
    processed_digits = []
    steps["digits_raw"] = []
    steps["thickened_digits"] = []
    steps["digits_resized"] = []
    steps["is_thickened_digits"] = []
    roi_list = []
    for (x, y, w, h) in digit_boxes:
        roi = img_optimized[y:y+h, x:x+w]
        roi_list.append(roi)
    HEIGHT_THRESHOLD = 350
    for idx, roi in enumerate(roi_list):
        steps["digits_raw"].append(image_to_base64(roi))
        h = roi.shape[0]
        if h > HEIGHT_THRESHOLD:
            roi_thickened = increase_stroke_thickness(
                roi, dilation_kernel_size=(5, 5), dilation_iterations=2, padding=6)
            steps["thickened_digits"].append(image_to_base64(roi_thickened))
            steps["is_thickened_digits"].append(True)
        else:
            roi_thickened = roi
            steps["thickened_digits"].append(image_to_base64(roi_thickened))
            steps["is_thickened_digits"].append(False)
        roi_resized = resize_and_pad(roi_thickened, size=28, padding=5)
        steps["digits_resized"].append(image_to_base64(roi_resized))
        roi_norm = roi_resized.astype("float32") / 255.0
        roi_norm = np.expand_dims(roi_norm, axis=-1)
        roi_norm = np.expand_dims(roi_norm, axis=0)
        processed_digits.append(roi_norm)
    return processed_digits, steps