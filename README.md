# 📝 Handwriting Recognition Web App

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

Ứng dụng web nhận diện chữ viết tay sử dụng Deep Learning (CNN Encoder + LSTM Decoder). Hỗ trợ nhận diện từ đơn và nhiều từ trong một dòng.

## ✨ Tính năng

- 🖊️ **Vẽ và nhận diện trực tiếp** trên canvas HTML5
- 📸 **Upload ảnh** để nhận diện
- 🔤 **Hai chế độ**:
  - Nhận diện từ đơn (single-word)
  - Nhận diện nhiều từ trong dòng (multi-word)
- 🔍 **Tùy chỉnh**:
  - Beam search width
  - Greedy vs Beam decode
  - Spell correction (bật/tắt)
- 📊 **Hiển thị confidence score** và các bước xử lý

## 🏗️ Kiến trúc

**Model**: CNN Encoder + LSTM Decoder + CTC Decode
- **CNN Encoder**: Trích xuất đặc trưng từ ảnh
- **LSTM Decoder**: Dự đoán chuỗi ký tự
- **CTC Decode**: Giải mã kết quả

**Preprocessing Pipeline**:
1. Grayscale conversion
2. CLAHE enhancement
3. Adaptive thresholding
4. Morphological operations
5. Resize to model input
6. Normalization
7. Tensor conversion

**Text Segmentation**: Vertical Projection method cho multi-word mode

## 🚀 Cài đặt

### Yêu cầu hệ thống

- Python 3.8+
- pip

### Các bước cài đặt

1. **Clone repository**
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

2. **Tạo virtual environment** (khuyên dùng)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Cài đặt dependencies**
```bash
pip install -r requirements.txt
```

4. **Download model file**

⚠️ **QUAN TRỌNG**: File model (`best_encoder_decoder.pth`) không được đưa lên GitHub do kích thước lớn. Bạn cần:

**Option 1**: Download từ link
- Tải model từ: [LINK_TO_YOUR_MODEL] (Google Drive/Dropbox)
- Đặt vào thư mục: `iam_p4/best_encoder_decoder.pth`

**Option 2**: Train model của bạn
- Xem hướng dẫn training trong `README_HANDWRITING.md`

Cấu trúc thư mục sau khi có model:
```
WEB_AI/
├── app.py
├── iam_p4/
│   └── best_encoder_decoder.pth  ← File này cần có
├── src/
├── static/
└── ...
```

5. **Chạy ứng dụng**
```bash
python app.py
```

Mở trình duyệt: `http://localhost:5000`

## 📦 Deploy lên Cloud

### Option 1: Deploy lên Render (Free, Recommended)

1. **Tạo tài khoản Render**: https://render.com

2. **Push code lên GitHub** (không bao gồm model file)

3. **Trên Render Dashboard**:
   - New → Web Service
   - Connect GitHub repo
   - Cấu hình:
     - **Environment**: Python 3
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `gunicorn app:app`
     - **Environment Variables**:
       - `PYTHON_VERSION`: `3.10.0`

4. **Upload model file**:
   - Sau khi deploy, dùng Render's persistent disk hoặc
   - Dùng external storage (Google Drive API) để load model

⚠️ **Lưu ý**: Free tier có giới hạn RAM (512MB), model lớn có thể bị crash. Xem xét:
- Dùng quantized model
- Upgrade lên paid tier ($7/month)

### Option 2: Deploy lên Railway

1. Tạo tài khoản: https://railway.app
2. New Project → Deploy from GitHub
3. Cấu hình tương tự Render
4. Add environment variables nếu cần

### Option 3: Deploy lên Heroku

```bash
# Cài Heroku CLI
# Login
heroku login

# Tạo app
heroku create your-app-name

# Push code
git push heroku main

# Set environment
heroku config:set PYTHON_VERSION=3.10
```

## 🔧 Cấu hình Model Path

Nếu muốn đổi model khác, sửa trong `app.py`:

```python
# Line 39
model_path = "iam_p4/best_encoder_decoder.pth"  # Đổi thành path khác
```

## 📚 Tài liệu kỹ thuật

Chi tiết về kiến trúc, training process, và các diagram:
- Xem file `README_HANDWRITING.md`

## 🐛 Troubleshooting

**Lỗi "Model file not found"**:
- Kiểm tra đường dẫn `iam_p4/best_encoder_decoder.pth` có tồn tại
- Download model như hướng dẫn ở bước 4

**Lỗi "CUDA out of memory"**:
- Model tự động chuyển sang CPU nếu không có GPU
- Nếu trên server, đảm bảo đủ RAM

**Segmentation không chính xác**:
- Điều chỉnh parameters trong `TextSegmenter` (file `src/data/segmentation.py`)
- Thử chế độ single-word cho ảnh phức tạp

## 📄 License

MIT License - xem file LICENSE

## 👨‍💻 Author

[YOUR_NAME] - [YOUR_EMAIL]

## 🙏 Acknowledgments

- IAM Handwriting Database
- PyTorch team
- OpenCV community
