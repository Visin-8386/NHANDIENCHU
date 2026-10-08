import os
import sys
import base64
import numpy as np
from PIL import Image
from io import BytesIO
import cv2
import torch
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Import handwriting model and utilities
from src.models.handwriting_model import (
    load_handwriting_model,
    decode_sequence,
    idx_to_char,
    SOS_IDX,
    EOS_IDX
)
from src.data.handwriting_preprocessing import (
    preprocess_handwriting_image,
    preprocess_word_segment,
    preprocess_batch,
    image_to_base64
)
from src.data.segmentation import (
    segment_text_image, 
    visualize_segmentation,
    visualize_segmentation_detailed,
    TextSegmenter,
    reconstruct_text_from_predictions
)
from src.postprocessing.spellcheck import SpellCorrector

# Đường dẫn đến mô hình đã huấn luyện
# iam_p3: Larger model (d=384, 6+4 layers) - Better for single chars
model_path = "models/iam_p4/best_encoder_decoder.pth"

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔥 Loading model on {device}...")
print(f"📦 Model: {model_path}")
model = load_handwriting_model(model_path, device=device)

# Try to load a custom wordlist if present
custom_words = []
custom_wordlist_path = os.path.join(os.path.dirname(__file__), 'data', 'wordlist.txt')
if os.path.isfile(custom_wordlist_path):
    try:
        with open(custom_wordlist_path, 'r', encoding='utf-8') as f:
            custom_words = [l.strip() for l in f.readlines() if l.strip()]
    except Exception as e:
        print(f"⚠️ Could not load custom wordlist: {e}")

# Initialize a SpellCorrector (default english, can be extended)
spell_corrector = SpellCorrector(language='en', custom_word_list=custom_words)
try:
    from src.postprocessing.spellcheck import is_spellchecker_available
    if not is_spellchecker_available():
        print("⚠️ SpellChecker backend (pyspellchecker) not available. Spellcheck is disabled until the package is installed.")
except Exception:
    # If import fails for diagnostics, continue silently (we already have fallback)
    pass

app = Flask(__name__)
CORS(app)


def predict_multi_word(image_np, decode_mode, beam_width, spellcheck_enabled):
    """
    Predict multi-line or multi-word text by segmenting and processing each word separately
    Uses batch processing for improved performance
    """
    try:
        print(f"🔍 Multi-word mode: Segmenting image...")
        
        # Convert to grayscale if needed
        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np
        
        # Use enhanced segmenter with optimized parameters
        segmenter = TextSegmenter(
            min_line_height=10,    # Giảm để không bỏ sót dòng nhỏ
            min_word_width=5,      # Giảm để không bỏ sót từ ngắn
            min_char_width=2,      # Giảm để capture ký tự mảnh như 'i', 'l'
            line_merge_threshold=0.3,
            word_gap_ratio=0.4
        )
        
        # Segment into words using hybrid method
        segments = segmenter.segment_full_image(gray, line_method='contours', word_method='hybrid')
        print(f"📦 Found {len(segments)} word segments")
        
        # Debug: print line info
        lines_info = {}
        for _, _, line_idx, word_idx in segments:
            if line_idx not in lines_info:
                lines_info[line_idx] = 0
            lines_info[line_idx] += 1
        print(f"📊 Lines detected: {len(lines_info)}, Words per line: {lines_info}")
        
        if len(segments) == 0:
            return jsonify({'error': 'Không tìm thấy văn bản trong ảnh'}), 400
        
        # Create visualization with predicted text first
        vis_img = visualize_segmentation_detailed(gray, segments, [])
        vis_b64 = image_to_base64(vis_img)
        
        # Create segmentation processing steps visualization
        processing_steps = []
        
        # Step 1: Original image (grayscale)
        processing_steps.append({
            'name': '1. Ảnh gốc (Grayscale)',
            'image': image_to_base64(gray),
            'shape': list(gray.shape)
        })
        
        # Step 2: Enhanced and binary for segmentation
        _, binary_seg = segmenter.preprocess_for_segmentation(gray)
        processing_steps.append({
            'name': '2. Nhị phân hóa cho segmentation',
            'image': image_to_base64(binary_seg),
            'shape': list(binary_seg.shape)
        })
        
        # Step 3: Line segmentation visualization
        lines_vis = gray.copy()
        if len(lines_vis.shape) == 2:
            lines_vis = cv2.cvtColor(lines_vis, cv2.COLOR_GRAY2BGR)
        
        # Draw line boxes
        line_bboxes = {}
        for _, bbox, line_idx, _ in segments:
            if line_idx not in line_bboxes:
                line_bboxes[line_idx] = []
            line_bboxes[line_idx].append(bbox)
        
        # Merge bboxes per line and draw
        for line_idx, bboxes in line_bboxes.items():
            x_min = min(b[0] for b in bboxes)
            y_min = min(b[1] for b in bboxes)
            x_max = max(b[0] + b[2] for b in bboxes)
            y_max = max(b[1] + b[3] for b in bboxes)
            
            color = [(255, 0, 0), (0, 255, 0), (255, 165, 0)][line_idx % 3]
            cv2.rectangle(lines_vis, (x_min, y_min), (x_max, y_max), color, 3)
            cv2.putText(lines_vis, f"Line {line_idx}", (x_min, y_min - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        processing_steps.append({
            'name': f'3. Phát hiện {len(line_bboxes)} dòng chữ',
            'image': image_to_base64(lines_vis),
            'shape': list(lines_vis.shape)
        })
        
        # Step 4: Word segmentation
        processing_steps.append({
            'name': f'4. Tách thành {len(segments)} từ',
            'image': vis_b64,
            'shape': list(gray.shape)
        })
        
        # Prepare batch for recognition
        word_images = [seg[0] for seg in segments]
        batch_tensor = preprocess_batch(word_images, device=device)
        print(f"📊 Batch tensor shape: {batch_tensor.shape}")
        
        # Run batch inference
        model.eval()
        all_results = []
        predictions = []
        confidences_list = []
        
        with torch.no_grad():
            # Process in mini-batches if too large
            batch_size = min(16, len(segments))
            
            for i in range(0, len(segments), batch_size):
                end_idx = min(i + batch_size, len(segments))
                mini_batch = batch_tensor[i:end_idx]
                
                result = model.generate(
                    mini_batch, SOS_IDX, EOS_IDX, 
                    max_len=27, 
                    mode=decode_mode, 
                    beam_width=beam_width,
                    verbose=False,
                    return_confidence=True
                )
                
                if isinstance(result, tuple):
                    pred_tokens, confs = result
                    for j in range(len(pred_tokens)):
                        pred_text = decode_sequence(pred_tokens[j], idx_to_char)
                        confidence = confs[j].item() if j < len(confs) else 0.95
                        predictions.append(pred_text)
                        confidences_list.append(confidence)
                else:
                    for j in range(len(result)):
                        pred_text = decode_sequence(result[j], idx_to_char)
                        predictions.append(pred_text)
                        confidences_list.append(0.95)
        
        # Apply spellcheck and build results
        reconstructed_text = []
        current_line = -1
        line_words = []
        
        for idx, (word_img, bbox, line_idx, word_idx) in enumerate(segments):
            pred_text = predictions[idx]
            confidence = confidences_list[idx]
            
            # Optional spellcheck
            if spellcheck_enabled:
                try:
                    corrected = spell_corrector.correct_text(pred_text)
                    if corrected and corrected != pred_text:
                        pred_text = corrected
                except:
                    pass
            
            # Store result
            x, y, w, h = bbox
            all_results.append({
                'text': pred_text,
                'confidence': confidence,
                'bbox': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)},
                'line': int(line_idx),
                'word': int(word_idx)
            })
            
            # Reconstruct text line by line
            if line_idx != current_line:
                if line_words:
                    reconstructed_text.append(' '.join(line_words))
                line_words = [pred_text]
                current_line = line_idx
            else:
                line_words.append(pred_text)
            
            print(f"  L{line_idx}W{word_idx}: '{pred_text}' (conf: {confidence:.2%})")
        
        # Add last line
        if line_words:
            reconstructed_text.append(' '.join(line_words))
        
        full_text = '\n'.join(reconstructed_text)
        
        # Update visualization with predicted text
        vis_img_final = visualize_segmentation_detailed(gray, segments, predictions)
        vis_b64 = image_to_base64(vis_img_final)
        
        # Update step 4 with predictions
        processing_steps[3] = {
            'name': f'4. Tách thành {len(segments)} từ (với nhận diện)',
            'image': vis_b64,
            'shape': list(gray.shape)
        }
        
        print(f"✅ Multi-word prediction complete!")
        print(f"📝 Full text:\n{full_text}")
        
        return jsonify({
            'mode': 'multi',
            'text': full_text,
            'word_count': len(all_results),
            'line_count': current_line + 1 if current_line >= 0 else 0,
            'words': all_results,
            'segmentation_image': vis_b64,
            'processing_steps': processing_steps
        })
        
    except Exception as e:
        import traceback
        print(f"❌ Multi-word error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': f'Lỗi xử lý multi-word: {str(e)}'}), 500


@app.route('/')
def index():
    return send_from_directory(os.path.dirname(__file__), 'index.html')


@app.route('/predict_handwriting', methods=['POST'])
def predict_handwriting():
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'Không có ảnh được gửi!'}), 400

        # Decode image
        image_data = data['image'].split(",")[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        image_np = np.array(image)

        print(f"📊 Image shape: {image_np.shape}, dtype: {image_np.dtype}, min: {image_np.min()}, max: {image_np.max()}")

        # Validate image
        if image_np.size == 0:
            return jsonify({'error': 'Ảnh rỗng!'}), 400

        if len(image_np.shape) not in [2, 3]:
            return jsonify({'error': f'Định dạng ảnh không hợp lệ: {image_np.shape}'}), 400

        # Get mode: 'single' or 'multi' (multi-line/multi-word)
        mode = data.get('mode', 'single')
        decode_mode = data.get('decode_mode', 'greedy')  # 'greedy' or 'beam'
        spellcheck_enabled = data.get('spellcheck', False)
        
        # Beam width for beam search (default top-3)
        try:
            beam_width = int(data.get('beam_width', 3))
            if beam_width < 1:
                beam_width = 1
            # Cap beam width to something reasonable (e.g., 50)
            beam_width = min(beam_width, 50)
        except Exception:
            beam_width = 3

        # Check if multi-line/multi-word mode
        if mode == 'multi':
            return predict_multi_word(image_np, decode_mode, beam_width, spellcheck_enabled)

        # Single word mode - Preprocess image with steps
        try:
            tensor, steps = preprocess_handwriting_image(image_np, return_steps=True)
            print(f"✅ Preprocessing done. Tensor shape: {tensor.shape}")
        except Exception as e:
            print(f"❌ Preprocessing error: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Lỗi xử lý ảnh: {str(e)}'}), 500

        # Move to device
        tensor = tensor.to(device)

        # Predict
        decode_method = "Beam Search (top-10)" if decode_mode == 'beam' else "Greedy"
        print(f"🤖 Running model inference with {decode_method}...")
        model.eval()
        with torch.no_grad():
            result = model.generate(tensor, SOS_IDX, EOS_IDX, max_len=27, mode=decode_mode, beam_width=beam_width, verbose=(decode_mode == 'beam'), return_confidence=True)

            # Unpack result
            if isinstance(result, tuple):
                pred_tokens, confidences = result
                confidence = confidences[0].item()
            else:
                pred_tokens = result
                confidence = 0.95  # Fallback

        # Decode
        pred_text = decode_sequence(pred_tokens[0], idx_to_char)
        print(f"📝 Predicted text: '{pred_text}' (using {decode_method}, confidence: {confidence:.2%})")

        # Optional spellcheck postprocessing
        if spellcheck_enabled:
            try:
                corrected = spell_corrector.correct_text(pred_text)
                if corrected and corrected != pred_text:
                    print(f"🛠️ Spell correction: '{pred_text}' -> '{corrected}'")
                    pred_text = corrected
            except Exception as e:
                print(f"⚠️ Spellcheck error: {e}")

        # Convert steps to array format for frontend visualization
        processing_steps = []
        
        # Define step order and display names
        step_info = {
            '1_original': 'Ảnh gốc (Grayscale)',
            '2_inverted': 'Đảo màu (nếu cần)',
            '3_enhanced': 'Tăng cường độ tương phản (CLAHE)',
            '4_binary_temp': 'Nhị phân hóa (Otsu)',
            '5_cropped': 'Cắt vùng chữ',
            '6_resized': 'Resize giữ tỉ lệ',
            '7_padded': 'Padding về 256x64'
        }
        
        for key in sorted(steps.keys()):
            if key in step_info and isinstance(steps[key], np.ndarray):
                try:
                    img = steps[key]
                    processing_steps.append({
                        'name': step_info[key],
                        'image': image_to_base64(img),
                        'shape': list(img.shape) if hasattr(img, 'shape') else None
                    })
                except Exception as e:
                    print(f"⚠️ Warning: Could not convert step {key} to base64: {e}")

        return jsonify({
            'mode': 'single',
            'text': pred_text,
            'confidence': confidence,
            'processing_steps': processing_steps
        })

    except Exception as e:
        import traceback
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # host='0.0.0.0' allows access from other devices on the network
    app.run(host='0.0.0.0', port=5000, debug=True)
