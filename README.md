<div align="center">

# 📝 Handwriting Recognition Web App

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Architecture](https://img.shields.io/badge/Architecture-ResNet%20%2B%20Transformer-8A2BE2)

Ứng dụng web nhận diện chữ viết tay hiện đại, tối ưu cho trải nghiệm thực tế: từ canvas đến ảnh upload, từ single-word đến multi-word.

</div>

---

## 🌟 Điểm nổi bật

- 🖊️ Vẽ trực tiếp trên canvas HTML5 và nhận diện ngay.
- 📸 Upload ảnh chữ viết tay để suy luận nhanh.
- 🔤 Hỗ trợ 2 chế độ: single-word và multi-word.
- 🔍 Tùy chỉnh decoding: greedy hoặc beam search.
- 🧠 Có spell correction để cải thiện độ đọc kết quả.
- 📊 Hiển thị confidence score và luồng xử lý rõ ràng.

---

## 🧭 Mục lục

- [Kiến trúc](#-kiến-trúc)
- [Pipeline xử lý ảnh](#-pipeline-xử-lý-ảnh)
- [Cài đặt nhanh](#-cài-đặt-nhanh)
- [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
- [Deploy Cloud](#-deploy-cloud)
- [Cấu hình Model Path](#-cấu-hình-model-path)
- [Troubleshooting](#-troubleshooting)
- [Author](#-author)

---

## 🏗️ Kiến trúc

> ✅ **Kiến trúc thực tế trong code hiện tại**: ResNet-style CNN Backbone + Transformer Encoder-Decoder

| Thành phần | Vai trò |
|---|---|
| CNN Backbone (ResNet-style) | Trích xuất đặc trưng từ ảnh chữ viết tay |
| Transformer Encoder | Mã hóa đặc trưng ảnh thành biểu diễn chuỗi |
| Transformer Decoder | Sinh chuỗi ký tự autoregressive với cross-attention |
| Decoding | Greedy hoặc Beam Search |
| Training Loss | CrossEntropyLoss + label smoothing |

⚠️ **Lưu ý quan trọng**: Dự án hiện tại **không** dùng LSTM + CTC decode.

---

## 🧪 Pipeline xử lý ảnh

1. Grayscale conversion
2. CLAHE enhancement
3. Adaptive thresholding
4. Morphological operations
5. Resize to model input
6. Normalization
7. Tensor conversion

**Text Segmentation**: Vertical Projection method cho multi-word mode.

---

## 🚀 Cài đặt nhanh

### Yêu cầu hệ thống

- Python 3.8+
- pip

### Các bước cài đặt

1. **Clone repository**

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

2. **Tạo virtual environment**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Cài dependencies**

```bash
pip install -r requirements.txt
```

4. **Bổ sung model file**

⚠️ File `best_encoder_decoder.pth` không đưa lên GitHub do kích thước lớn.

**Option 1**: Download từ link
- Tải model từ: [LINK_TO_YOUR_MODEL]
- Đặt vào: `iam_p4/best_encoder_decoder.pth`

**Option 2**: Tự train model
- Xem thêm trong `README_HANDWRITING.md`

5. **Chạy ứng dụng**

```bash
python app.py
```

Mở trình duyệt tại: `http://localhost:5000`

---

## 📁 Cấu trúc thư mục

```text
WEB_AI/
├── app.py
├── iam_p4/
│   ├── best_encoder_decoder.pth
│   └── handwritten-text-recognition-iam-crnn.ipynb
├── src/
├── static/
├── templates/
└── README.md
```

---

## 📦 Deploy Cloud

### Option 1: Render (Recommended)

1. Tạo tài khoản tại https://render.com
2. Push code lên GitHub (không kèm model file)
3. Cấu hình service:
- Environment: Python 3
- Build Command: `pip install -r requirements.txt`
- Start Command: `gunicorn app:app`
- Environment Variables: `PYTHON_VERSION=3.10.0`
4. Cấp model file qua persistent disk hoặc external storage

⚠️ Free tier giới hạn RAM, model lớn có thể OOM.

### Option 2: Railway

1. Tạo tài khoản tại https://railway.app
2. New Project → Deploy from GitHub
3. Cấu hình build/start tương tự Render

### Option 3: Heroku

```bash
heroku login
heroku create your-app-name
git push heroku main
heroku config:set PYTHON_VERSION=3.10
```

---

## 🔧 Cấu hình Model Path

Trong `app.py`, chỉnh đường dẫn model nếu cần:

```python
model_path = "iam_p4/best_encoder_decoder.pth"
```

---

## 🐛 Troubleshooting

### Lỗi "Model file not found"

- Kiểm tra file `iam_p4/best_encoder_decoder.pth` có tồn tại.
- Kiểm tra biến `model_path` trong `app.py`.

### Lỗi "CUDA out of memory"

- Chuyển sang CPU nếu không cần tốc độ cao.
- Giảm batch size khi chạy training/inference.
- Dùng model nhẹ hơn hoặc quantized model.

### Segmentation không chính xác

- Điều chỉnh tham số trong `src/data/segmentation.py`.
- Thử ảnh sạch hơn hoặc chuyển sang single-word mode.

---

## 📄 License

MIT License - xem file LICENSE.

---

## 👨‍💻 Author

**Lê Hoàng**

AI Software Engineer crafting practical AI products from research to real-world deployment.

📧 **Email**: le294594@gmail.com  
🌟 **Focus**: Handwriting Recognition, Computer Vision, and AI-driven Web Applications

> "I build AI that does not stop at demo quality; it ships, scales, and solves real problems."

---

## 🙏 Acknowledgments

- IAM Handwriting Database
- PyTorch team
- OpenCV community
