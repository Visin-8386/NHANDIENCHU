import os
import sys
import time
import torch
import cv2
import numpy as np
from PIL import Image

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Fix encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

from src.models.handwriting_model import load_handwriting_model, decode_sequence, idx_to_char, SOS_IDX, EOS_IDX
from src.data.handwriting_preprocessing import preprocess_batch
from src.data.segmentation import TextSegmenter

def test_token_speed():
    model_path = "models/iam_p4/best_encoder_decoder.pth"
    image_path = r"D:\WEB_AI\data\authentic_digits_test\z7252775403934_e9f6a68fd84e0fe4f1fd3cad7ab31201.jpg"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading model on {device}...")
    model = load_handwriting_model(model_path, device=device)
    model.eval()

    # Load image
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return

    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Segment
    print("Searching image...")
    segmenter = TextSegmenter()
    segments = segmenter.segment_full_image(gray)
    print(f"Found {len(segments)} word segments")

    if len(segments) == 0:
        print("No segments found.")
        return

    # Preprocess
    word_images = [seg[0] for seg in segments]
    batch_tensor = preprocess_batch(word_images, device=device)

    # Benchmark
    print(f"Benchmarking token generation speed for {len(segments)} words...")
    
    # Warmup
    with torch.no_grad():
        _ = model.generate(batch_tensor, SOS_IDX, EOS_IDX, max_len=27)

    # Actual test
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    
    with torch.no_grad():
        # result is a tuple (tgt_tokens, confidences) because return_confidence=True in app.py's logic
        # but here we can just use defaults
        result = model.generate(batch_tensor, SOS_IDX, EOS_IDX, max_len=27, return_confidence=True)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()
    
    total_time = end_time - start_time
    
    tgt_tokens, confidences = result
    
    # Calculate tokens
    total_tokens = 0
    words_decoded = []
    for i in range(len(tgt_tokens)):
        tokens = tgt_tokens[i]
        # Count non-SOS, non-PAD tokens.
        # SOS is at index 0. We count from index 1 until EOS or end.
        count = 0
        for t in tokens[1:]:
            count += 1
            if t == EOS_IDX:
                break
        total_tokens += count
        
        text = decode_sequence(tokens, idx_to_char)
        words_decoded.append(text)

    tps = total_tokens / total_time
    latency_per_word = (total_time / len(segments)) * 1000
    ms_per_token = (total_time / total_tokens) * 1000

    print("\n" + "="*50)
    print(f"📊 RESULTS for {len(segments)} words:")
    print(f"📝 Text: {' '.join(words_decoded)}")
    print("-" * 50)
    print(f"Total Tokens Generated: {total_tokens}")
    print(f"Total Time:           {total_time:.4f} s")
    print(f"Tokens Per Second:     {tps:.2f} tokens/s")
    print(f"Avg Latency per Word:  {latency_per_word:.2f} ms")
    print(f"Avg Time per Token:    {ms_per_token:.2f} ms")
    print("="*50)

if __name__ == "__main__":
    test_token_speed()
