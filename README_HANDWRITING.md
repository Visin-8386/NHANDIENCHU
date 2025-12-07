# 🖊️ Hệ thống nhận diện chữ viết tay bằng AI

Dự án nhận diện chữ viết tay sử dụng mô hình Encoder-Decoder (PyTorch) được huấn luyện trên dataset IAM Handwriting.

## 🎯 Tính năng

- ✅ Nhận diện chữ viết tay (chữ thường a-z, số 0-9, khoảng trắng)
- ✅ Hỗ trợ nhận diện một dòng
- ✅ Giao diện web tương tác với canvas vẽ
- ✅ Upload ảnh từ máy tính
- ✅ Hiển thị các bước xử lý ảnh chi tiết
- ✅ Model Encoder-Decoder với Transformer
- ✅ Spell-check post-processing (optional) using pyspellchecker

## 📋 Yêu cầu

- Python 3.8+
- PyTorch 2.0+
- Flask
- OpenCV
- NumPy

## 🚀 Cài đặt

### 1. Clone repository

```bash
git clone <repo-url>
cd WEB_AI
```

### 2. Tạo virtual environment

```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```
If using beam search, you can specify the beam width (number of candidates) using the `beam_width` parameter (default top-3). Example request body (with beam search and top-3):

```json
{
    "image": "data:image/png;base64,...",
    "mode": "single",
    "decode_mode": "beam",
    "beam_width": 3,
    "spellcheck": true
}
```

If you use VS Code and the Pylance language server reports "Import 'spellchecker' could not be resolved", make sure to select the Python interpreter for your project's virtual environment and run the command above. If issues persist, run:

```bash
pip install pyspellchecker
```
Or, if you've pinned a version:
```bash
pip install pyspellchecker==0.7.1
```

### 4. Đảm bảo model file tồn tại

Model file phải được đặt tại:
```
iam_p1/best_encoder_decoder.pth
```

## 💻 Sử dụng

### 1. Khởi động server

```bash
python app.py
```

Server sẽ chạy tại: `http://127.0.0.1:5000`

### 2. Sử dụng web interface

1. Mở trình duyệt và truy cập `http://127.0.0.1:5000`
2. Chế độ: Hệ thống hiện chỉ hỗ trợ **Một dòng** (Nhận diện từ hoặc cụm từ ngắn)
3. Viết chữ trên canvas hoặc upload ảnh
4. Nhấn "Dự đoán" để nhận kết quả
5. Xem kết quả và các bước xử lý ảnh

### 3. API Endpoint

#### POST `/predict_handwriting`

**Request:**
```json
{
  "image": "data:image/png;base64,...",
  "mode": "single",
  "spellcheck": true // optional: boolean to enable pyspellchecker spell correction on the returned text
}
```

**Response (single mode):**
```json
{
  "mode": "single",
  "text": "hello world",
  "confidence": 0.95,
  "steps": {
    "1_original": "data:image/png;base64,...",
    "2_binary": "data:image/png;base64,...",
    ...
  }
}
```

<!-- Multi-line mode removed: system supports single-line only -->

## 🏗️ Kiến trúc Model

### Encoder-Decoder Architecture

```
Input Image (64x256)
    ↓
CNN Backbone (SimplifiedCNN)
    ├─ Conv2D + BatchNorm + GELU
    ├─ MaxPool2D
    └─ Output: [B, 256, 8, 64]
    ↓
2D Positional Encoding
    ↓
Transformer Encoder (4 layers)
    ├─ Multi-Head Attention
    ├─ Feed-Forward Network
    └─ Layer Normalization
    ↓
Memory Features
    ↓
Transformer Decoder (3 layers)
    ├─ Self-Attention
    ├─ Cross-Attention (to encoder memory)
    ├─ Feed-Forward Network
    └─ Layer Normalization
    ↓
Output Projection
    ↓
Character Sequence
```

### Model Parameters

- **d_model**: 256
- **Encoder layers**: 4
- **Decoder layers**: 3
- **Attention heads**: 8
- **FFN dimension**: 1024
- **Dropout**: 0.1
- **Vocabulary size**: 40 tokens (PAD, SOS, EOS + a-z + 0-9 + space)

## 📁 Cấu trúc thư mục

```
WEB_AI/
├── app.py                      # Flask application
├── index.html                  # Web interface
├── requirements.txt            # Dependencies
├── iam_p1/
│   └── best_encoder_decoder.pth  # Trained model weights
├── src/
│   ├── data/
│   │   └── handwriting_preprocessing.py  # Image preprocessing
│   └── models/
│       └── handwriting_model.py         # Model architecture
├── static/
│   ├── main.js                 # Frontend JavaScript
│   └── style.css               # Styling
└── README_HANDWRITING.md       # This file
```

## 🔧 Preprocessing Pipeline

1. **Convert to Grayscale**: Chuyển ảnh màu sang grayscale
2. **Binary Threshold**: Áp dụng Otsu's thresholding
3. **Invert**: Đảo màu nếu cần (background trắng, text đen)
4. **Denoise**: Khử nhiễu bằng morphological operations
5. **Crop**: Cắt vùng chứa chữ viết
6. **Resize**: Thay đổi kích thước giữ tỷ lệ
7. **Pad**: Thêm padding về kích thước chuẩn (64x256)

## 📊 Model Performance

- **Dataset**: IAM Handwriting Words Database
- **Training samples**: ~90,000 words
- **Validation samples**: ~10,000 words
- **Architecture**: Encoder-Decoder with Transformer
- **Character Error Rate (CER)**: < 8%

## 🎨 Giao diện

- Dark theme
- Real-time canvas drawing
- Adjustable brush thickness & color
- Background color customization
- Image upload support
- Detailed processing steps visualization
- Single/Multi-line mode selector

## 🚧 Lưu ý

- Model hỗ trợ: chữ thường (a-z), số (0-9), và khoảng trắng
- Chữ HOA sẽ được chuyển thành chữ thường
- Viết chữ rõ ràng để đạt độ chính xác cao
- Nền đen, chữ trắng hoạt động tốt nhất

## 🔄 So sánh với phiên bản cũ

| Tính năng | Phiên bản cũ | Phiên bản mới |
|-----------|--------------|---------------|
| Model | TensorFlow CNN | PyTorch Encoder-Decoder |
| Task | Nhận diện chữ số (0-9) | Nhận diện chữ viết tay (a-z, 0-9) |
| Input | 28x28 px | 64x256 px |
| Output | Single digit | Text sequence |
| Architecture | CNN | Transformer Encoder-Decoder |
| Parameters | ~6M | ~6.5M |

## 📝 License

MIT License

## 👨‍💻 Author

Your Name

## 🙏 Acknowledgments

- IAM Handwriting Database
- PyTorch Team
- TrOCR Architecture inspiration
