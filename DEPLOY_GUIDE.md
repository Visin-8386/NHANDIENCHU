# 🚀 Hướng dẫn Deploy lên GitHub và Cloud

## 📋 Mục lục
1. [Push lên GitHub](#1-push-lên-github)
2. [Deploy lên Render (Free)](#2-deploy-lên-render-miễn-phí)
3. [Deploy lên Railway](#3-deploy-lên-railway)
4. [Xử lý Model File lớn](#4-xử-lý-model-file-lớn)
5. [Troubleshooting](#5-troubleshooting)

---

## 1. Push lên GitHub

### Bước 1: Tạo repository trên GitHub
1. Đăng nhập GitHub
2. Click **New repository**
3. Đặt tên: `handwriting-recognition-app`
4. Chọn **Public** hoặc **Private**
5. **KHÔNG** chọn "Add a README" (vì đã có sẵn)
6. Click **Create repository**

### Bước 2: Khởi tạo Git và push code

Mở terminal trong thư mục `d:\WEB_AI` và chạy:

```bash
# Khởi tạo Git repository
git init

# Add tất cả file (trừ những file trong .gitignore)
git add .

# Commit lần đầu
git commit -m "Initial commit: Handwriting recognition web app"

# Link với GitHub repo (thay YOUR_USERNAME và YOUR_REPO)
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git

# Push lên GitHub
git branch -M main
git push -u origin main
```

**Lưu ý**: File model (`.pth`) sẽ KHÔNG được push lên GitHub do có trong `.gitignore` (file quá lớn).

---

## 2. Deploy lên Render (Miễn phí)

### ⚠️ Giới hạn Free Tier:
- **RAM**: 512MB
- **Build time**: 15 phút
- **Sleep sau 15 phút không hoạt động**
- Model PyTorch lớn (~300MB) có thể gây vấn đề về RAM

### Bước 1: Upload Model lên Cloud Storage

**Option A: Google Drive (Khuyên dùng)**

1. Upload file `iam_p4/best_encoder_decoder.pth` lên Google Drive
2. Share file → Get link
3. Lấy **File ID** từ link:
   ```
   https://drive.google.com/file/d/FILE_ID_HERE/view?usp=sharing
   ```
4. Tạo direct download link:
   ```
   https://drive.google.com/uc?id=FILE_ID_HERE&export=download
   ```

**Option B: Dropbox**
- Upload và lấy direct link

### Bước 2: Sửa code để download model tự động

Thêm vào `app.py` (đầu file, sau imports):

```python
import os
import urllib.request

def download_model_if_missing():
    model_path = "iam_p4/best_encoder_decoder.pth"
    if not os.path.exists(model_path):
        print("⬇️ Downloading model from cloud storage...")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Thay YOUR_FILE_ID bằng ID thật từ Google Drive
        url = "https://drive.google.com/uc?id=YOUR_FILE_ID&export=download"
        
        try:
            urllib.request.urlretrieve(url, model_path)
            print("✅ Model downloaded successfully!")
        except Exception as e:
            print(f"❌ Failed to download model: {e}")
            raise
    else:
        print("✅ Model file found locally")

# Gọi trước khi load model
download_model_if_missing()
```

**Commit thay đổi**:
```bash
git add app.py
git commit -m "Add auto model download from cloud"
git push
```

### Bước 3: Deploy trên Render

1. **Tạo tài khoản Render**: https://render.com (dùng GitHub login)

2. **Tạo Web Service**:
   - Dashboard → **New** → **Web Service**
   - Connect GitHub repository của bạn
   - Cấu hình:
     - **Name**: `handwriting-app` (hoặc tên bạn muốn)
     - **Environment**: `Python 3`
     - **Branch**: `main`
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `gunicorn app:app`
     - **Instance Type**: `Free`

3. **Environment Variables** (nếu cần):
   - Click **Environment** → **Add Environment Variable**
   - Thêm biến nếu cần (ví dụ: `MODEL_URL`)

4. **Deploy**:
   - Click **Create Web Service**
   - Đợi 10-15 phút build
   - Xem logs để kiểm tra

5. **Kiểm tra**:
   - Sau khi deploy xong, click link: `https://your-app-name.onrender.com`

---

## 3. Deploy lên Railway

### Bước 1: Tạo tài khoản Railway
1. Truy cập: https://railway.app
2. Sign up với GitHub

### Bước 2: Deploy
1. **New Project** → **Deploy from GitHub repo**
2. Chọn repository của bạn
3. Railway tự động detect Python và chạy
4. Thêm environment variables nếu cần

### Bước 3: Custom Start Command
1. Settings → **Start Command**:
   ```
   gunicorn app:app --bind 0.0.0.0:$PORT
   ```

Railway có **512MB RAM** (free) nhưng không sleep app.

---

## 4. Xử lý Model File lớn

### Giải pháp 1: Model Quantization (Giảm kích thước)

Tạo script `optimize_model.py`:

```python
import torch

# Load model gốc
model = torch.load('iam_p4/best_encoder_decoder.pth', map_location='cpu')

# Quantize (giảm từ float32 → int8)
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear, torch.nn.LSTM}, dtype=torch.qint8
)

# Save model nhẹ hơn (khoảng 75% kích thước gốc)
torch.save(quantized_model, 'iam_p4/best_encoder_decoder_quantized.pth')
print("✅ Model quantized successfully!")
```

Chạy local:
```bash
python optimize_model.py
```

Rồi upload model quantized lên cloud storage.

### Giải pháp 2: Hugging Face Hub (Recommended)

1. **Tạo tài khoản Hugging Face**: https://huggingface.co
2. **Upload model**:
   ```bash
   pip install huggingface_hub
   ```
   
   ```python
   from huggingface_hub import HfApi
   
   api = HfApi()
   api.upload_file(
       path_or_fileobj="iam_p4/best_encoder_decoder.pth",
       path_in_repo="best_encoder_decoder.pth",
       repo_id="YOUR_USERNAME/handwriting-model",
       repo_type="model",
   )
   ```

3. **Load từ HF trong app.py**:
   ```python
   from huggingface_hub import hf_hub_download
   
   model_path = hf_hub_download(
       repo_id="YOUR_USERNAME/handwriting-model",
       filename="best_encoder_decoder.pth"
   )
   ```

---

## 5. Troubleshooting

### ❌ Build Failed: "Out of Memory"
**Giải pháp**:
- Dùng model quantized nhỏ hơn
- Upgrade lên paid tier ($7/month trên Render)
- Dùng Railway (có thể handle tốt hơn)

### ❌ "Model file not found"
**Giải pháp**:
- Kiểm tra URL download model có đúng không
- Xem logs: file có tải về thành công không
- Đảm bảo thư mục `iam_p4/` được tạo

### ❌ App sleep sau 15 phút (Render Free)
**Giải pháp**:
- Dùng Railway (không sleep)
- Hoặc upgrade Render
- Hoặc dùng UptimeRobot để ping app 5 phút/lần

### ❌ CORS Error
**Giải pháp**:
- Đã có `flask-cors` trong code
- Nếu vẫn lỗi, thêm domain cụ thể:
  ```python
  CORS(app, origins=["https://your-frontend-domain.com"])
  ```

### ❌ Slow Response
**Giải pháp**:
- Model lớn + CPU chậm trên free tier
- Xem xét giảm model size
- Hoặc dùng paid tier có GPU

---

## ✅ Checklist Deploy

- [ ] Code đã push lên GitHub
- [ ] File `.gitignore` đã loại trừ model và notebooks
- [ ] Model đã upload lên cloud storage (Google Drive/HF)
- [ ] Code có logic download model tự động
- [ ] `requirements.txt` có đầy đủ dependencies
- [ ] `Procfile` và `runtime.txt` đã tạo
- [ ] Deploy trên Render/Railway thành công
- [ ] Test app: vẽ chữ và kiểm tra kết quả
- [ ] Logs không có lỗi critical

---

## 📞 Support

Nếu gặp lỗi:
1. Kiểm tra **Logs** trên Render/Railway dashboard
2. Tìm dòng lỗi cụ thể (màu đỏ)
3. Google error message
4. Hoặc hỏi tôi với thông tin logs cụ thể

**Chúc bạn deploy thành công! 🎉**
