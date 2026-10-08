/* ============================================================
   HTR — Handwritten Text Recognition
   Canvas drawing engine + predict flow + theme + UI helpers
   ============================================================ */

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d', { willReadFrequently: true });

let isDrawing = false;
let currentMode = 'single'; // 'single' or 'multi'
let undoStack = [];
let isEraserMode = false;
let isPredicting = false;
let lastResultText = '';

// Brush settings
let brushSize = 3;
let brushColor = '#000000';

// ---------- Theme ----------
const themeToggle = document.getElementById('theme-toggle');

function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    document.querySelector('meta[name="theme-color"]')
        .setAttribute('content', theme === 'dark' ? '#09090b' : '#fafafa');
    try {
        localStorage.setItem('htr-theme', theme);
    } catch (e) { /* private mode */ }
}

function initTheme() {
    let saved = null;
    try {
        saved = localStorage.getItem('htr-theme');
    } catch (e) { /* ignore */ }
    if (saved === 'light' || saved === 'dark') {
        applyTheme(saved);
    } else if (window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches) {
        applyTheme('light');
    } else {
        applyTheme('dark');
    }
}

themeToggle.addEventListener('click', () => {
    const current = document.documentElement.getAttribute('data-theme');
    applyTheme(current === 'dark' ? 'light' : 'dark');
});

// ---------- Toast ----------
const toastEl = document.getElementById('toast');
let toastTimer = null;

function showToast(message, isError = false) {
    toastEl.textContent = message;
    toastEl.classList.toggle('toast-error', isError);
    toastEl.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toastEl.classList.remove('show'), 2600);
}

// ---------- Canvas ----------
function initCanvas() {
    adjustCanvasSize();
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    updateCanvasContext();
    saveCanvasState();
}

function adjustCanvasSize() {
    const oldWidth = canvas.width;
    const oldHeight = canvas.height;
    let imageData = null;

    if (oldWidth > 0 && oldHeight > 0) {
        try {
            imageData = ctx.getImageData(0, 0, oldWidth, oldHeight);
        } catch (e) {
            console.log('Could not save canvas state');
        }
    }

    const containerWidth = Math.min(window.innerWidth - 40, 900);

    if (window.innerWidth <= 480) {
        canvas.width = Math.min(containerWidth, 350);
        canvas.height = Math.round(canvas.width * 0.6);
    } else if (window.innerWidth <= 768) {
        canvas.width = Math.min(containerWidth, 500);
        canvas.height = Math.round(canvas.width * 0.6);
    } else {
        canvas.width = 900;
        canvas.height = 600;
    }

    updateCanvasContext();

    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    if (imageData) {
        try {
            ctx.putImageData(imageData, 0, 0);
        } catch (e) {
            console.log('Could not restore canvas state');
        }
    }
}

function updateCanvasContext() {
    ctx.fillStyle = 'white';
    ctx.strokeStyle = isEraserMode ? 'white' : brushColor;
    ctx.lineWidth = brushSize;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
}

function getCoordinates(e) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    if (e.touches) {
        return {
            x: (e.touches[0].clientX - rect.left) * scaleX,
            y: (e.touches[0].clientY - rect.top) * scaleY
        };
    }
    return {
        x: (e.clientX - rect.left) * scaleX,
        y: (e.clientY - rect.top) * scaleY
    };
}

function startDrawing(e) {
    e.preventDefault();
    isDrawing = true;
    const coords = getCoordinates(e);
    ctx.beginPath();
    ctx.moveTo(coords.x, coords.y);
}

function draw(e) {
    if (!isDrawing) return;
    e.preventDefault();

    const coords = getCoordinates(e);
    ctx.strokeStyle = isEraserMode ? 'white' : brushColor;
    ctx.lineWidth = brushSize;
    ctx.lineTo(coords.x, coords.y);
    ctx.stroke();
}

function stopDrawing(e) {
    if (isDrawing) {
        e.preventDefault();
        isDrawing = false;
        ctx.beginPath();
        saveCanvasState();
    }
}

function saveCanvasState() {
    undoStack.push(ctx.getImageData(0, 0, canvas.width, canvas.height));
    if (undoStack.length > 20) {
        undoStack.shift();
    }
}

function undoStroke() {
    if (undoStack.length > 1) {
        undoStack.pop();
        const previousState = undoStack[undoStack.length - 1];
        ctx.putImageData(previousState, 0, 0);
    }
}

function toggleEraser() {
    isEraserMode = !isEraserMode;
    const eraserBtn = document.getElementById('eraser-btn');
    eraserBtn.classList.toggle('active', isEraserMode);
    canvas.classList.toggle('eraser-mode', isEraserMode);
    eraserBtn.title = isEraserMode ? 'Chuyển sang bút vẽ' : 'Tẩy';
    eraserBtn.setAttribute('aria-label', isEraserMode ? 'Chuyển sang bút vẽ' : 'Tẩy');
}

function clearCanvas() {
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    resetResultPanel();
    undoStack = [];
    saveCanvasState();
}

// ---------- Events ----------
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

canvas.addEventListener('touchstart', startDrawing, { passive: false });
canvas.addEventListener('touchmove', draw, { passive: false });
canvas.addEventListener('touchend', stopDrawing, { passive: false });

document.body.addEventListener('touchstart', function (e) {
    if (e.target === canvas) e.preventDefault();
}, { passive: false });

document.body.addEventListener('touchmove', function (e) {
    if (e.target === canvas) e.preventDefault();
}, { passive: false });

// Ctrl+Z / Cmd+Z undo
document.addEventListener('keydown', function (e) {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z' && !e.shiftKey) {
        const tag = (document.activeElement && document.activeElement.tagName) || '';
        if (tag !== 'INPUT' && tag !== 'TEXTAREA' && tag !== 'SELECT') {
            e.preventDefault();
            undoStroke();
        }
    }
});

// ---------- Settings listeners ----------
document.getElementById('brush-size').addEventListener('input', function (e) {
    brushSize = parseInt(e.target.value);
    document.getElementById('brush-size-value').textContent = brushSize + 'px';
    ctx.lineWidth = brushSize;
});

document.getElementById('brush-color').addEventListener('input', function (e) {
    brushColor = e.target.value;
    if (!isEraserMode) {
        ctx.strokeStyle = brushColor;
    }
});

// Show/hide beam width depending on decode mode
document.getElementById('decode-mode').addEventListener('change', function (e) {
    document.getElementById('beam-row').style.display = e.target.value === 'beam' ? '' : 'none';
});

function updateMode() {
    const selected = document.querySelector('input[name="mode"]:checked').value;
    currentMode = selected;
}

// ---------- Result panel ----------
const EMPTY_RESULT_HTML = `
    <div class="empty-state">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z"/></svg>
        <p>Vẽ chữ và nhấn <strong>Nhận diện</strong> để xem kết quả</p>
    </div>`;

const EMPTY_STEPS_HTML = `
    <div class="empty-state">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M2 12h4l3-8 4 16 3-8h6"/></svg>
        <p>Các bước xử lý sẽ hiển thị ở đây sau khi nhận diện</p>
    </div>`;

function resetResultPanel() {
    document.getElementById('result').innerHTML = EMPTY_RESULT_HTML;
    document.getElementById('processing-steps').innerHTML = EMPTY_STEPS_HTML;
    document.getElementById('copy-btn').hidden = true;
    lastResultText = '';
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function confidenceBarHtml(confidence) {
    const pct = Math.max(0, Math.min(100, confidence * 100));
    return `
        <div class="confidence-block">
            <div class="confidence-header">
                <span>Độ tin cậy</span>
                <strong>${pct.toFixed(1)}%</strong>
            </div>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: 0%"></div>
            </div>
        </div>`;
}

function animateConfidenceBars() {
    requestAnimationFrame(() => {
        document.querySelectorAll('.confidence-fill').forEach(fill => {
            const target = fill.dataset.target;
            if (target) fill.style.width = target;
        });
    });
}

function copyResult() {
    if (!lastResultText) return;
    const done = () => showToast('Đã sao chép kết quả');
    if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(lastResultText).then(done).catch(() => fallbackCopy(done));
    } else {
        fallbackCopy(done);
    }
}

function fallbackCopy(done) {
    const ta = document.createElement('textarea');
    ta.value = lastResultText;
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    try {
        document.execCommand('copy');
        done();
    } catch (e) {
        showToast('Không thể sao chép', true);
    }
    document.body.removeChild(ta);
}

// ---------- Predict ----------
async function predict() {
    if (isPredicting) return;

    const resultDiv = document.getElementById('result');
    const stepsDiv = document.getElementById('processing-steps');
    const predictBtn = document.getElementById('predict-btn');

    isPredicting = true;
    predictBtn.classList.add('loading');
    predictBtn.disabled = true;

    resultDiv.innerHTML = `
        <div class="result-loading">
            <div class="skeleton skeleton-line-lg"></div>
            <div class="skeleton skeleton-line-sm"></div>
            <p class="loading-hint">Đang nhận diện… lần đầu sau khi ứng dụng ngủ có thể mất 1–2 phút</p>
        </div>`;
    stepsDiv.innerHTML = EMPTY_STEPS_HTML;

    // Snapshot canvas to a temp canvas with guaranteed white background
    const tempCanvas = document.createElement('canvas');
    const tempCtx = tempCanvas.getContext('2d');
    tempCanvas.width = canvas.width;
    tempCanvas.height = canvas.height;
    tempCtx.fillStyle = 'white';
    tempCtx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);
    tempCtx.drawImage(canvas, 0, 0);

    const imageData = tempCanvas.toDataURL('image/png');
    const decodeMode = document.getElementById('decode-mode').value;
    const beamWidth = parseInt(document.getElementById('beam-width').value) || 3;
    const spellcheck = document.getElementById('spellcheck').checked;

    const payload = JSON.stringify({
        image: imageData,
        mode: currentMode,
        decode_mode: decodeMode,
        beam_width: beamWidth,
        spellcheck: spellcheck
    });

    // The free-tier service sleeps when idle; the first request after wake can
    // fail while the worker boots. Retry transparently (up to 2 extra attempts).
    const MAX_ATTEMPTS = 3;
    let lastError = null;

    for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
        try {
            const response = await fetch('/predict_handwriting', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: payload
            });

            if (response.status === 502 || response.status === 503 || response.status === 429) {
                throw new Error('server-warming');
            }

            const data = await response.json();

            if (data.error) {
                showError(data.error);
                resetPredictButton();
                return;
            }

            if (currentMode === 'multi' && data.words) {
                displayMultiWordResult(data);
            } else {
                displaySingleWordResult(data);
            }

            if (data.processing_steps) {
                displayProcessingSteps(data.processing_steps);
            }
            resetPredictButton();
            return;
        } catch (error) {
            lastError = error;
            console.error(`Prediction attempt ${attempt} failed:`, error);
            if (attempt < MAX_ATTEMPTS) {
                // Wait for the worker to finish booting before retrying
                await new Promise(r => setTimeout(r, 20000));
            }
        }
    }

    showError('Không nhận được phản hồi từ máy chủ. Vui lòng thử lại sau vài giây.');
    resetPredictButton();
}

function resetPredictButton() {
    const predictBtn = document.getElementById('predict-btn');
    isPredicting = false;
    predictBtn.classList.remove('loading');
    predictBtn.disabled = false;
}

function showError(message) {
    document.getElementById('result').innerHTML = `
        <div class="result-error">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="10"/><path d="M12 8v4"/><path d="M12 16h.01"/></svg>
            <span>${escapeHtml(message)}</span>
        </div>`;
    document.getElementById('copy-btn').hidden = true;
    lastResultText = '';
}

function displaySingleWordResult(data) {
    const resultDiv = document.getElementById('result');
    let html = `<div class="recognized-text">"${escapeHtml(data.text)}"</div>`;

    if (data.confidence !== undefined) {
        const pct = Math.max(0, Math.min(100, data.confidence * 100));
        html += confidenceBarHtml(data.confidence).replace('width: 0%', 'width: 0%');
        html = html.replace(/<div class="confidence-fill" style="width: 0%"><\/div>/,
            `<div class="confidence-fill" data-target="${pct.toFixed(1)}%"></div>`);
    }

    if (data.raw_text && data.raw_text !== data.text) {
        html += `<p class="raw-text">Bản gốc: <span>"${escapeHtml(data.raw_text)}"</span></p>`;
    }

    resultDiv.innerHTML = html;
    lastResultText = data.text || '';
    document.getElementById('copy-btn').hidden = !lastResultText;
    
    // Trigger reflow and add animation
    resultDiv.offsetHeight;
    animateConfidenceBars();
    showToast('Nhận diện thành công');
}

function displayMultiWordResult(data) {
    const resultDiv = document.getElementById('result');
    let html = '';

    if (data.segmentation_image) {
        const imgSrc = data.segmentation_image.startsWith('data:')
            ? data.segmentation_image
            : `data:image/png;base64,${data.segmentation_image}`;
        html += `<img src="${imgSrc}" alt="Segmentation" class="segmentation-image">`;
    }

    html += `<div class="word-list">`;
    data.words.forEach((word, index) => {
        const pct = word.confidence !== undefined
            ? Math.max(0, Math.min(100, word.confidence * 100))
            : null;
        html += `
            <div class="word-result" style="animation-delay: ${index * 0.04}s">
                <div class="word-head">
                    <span class="word-index">Từ ${index + 1}</span>
                    <span class="word-text">"${escapeHtml(word.text)}"</span>
                    ${pct !== null ? `<span class="word-conf">${pct.toFixed(1)}%</span>` : ''}
                </div>
                ${pct !== null ? `
                <div class="confidence-bar">
                    <div class="confidence-fill" data-target="${pct.toFixed(1)}%"></div>
                </div>` : ''}
                ${word.raw_text && word.raw_text !== word.text
                    ? `<p class="raw-text">Bản gốc: <span>"${escapeHtml(word.raw_text)}"</span></p>`
                    : ''}
            </div>`;
    });
    html += `</div>`;

    const fullText = data.words.map(w => w.text).join(' ');
    html += `
        <div class="recognized-text">
            <span class="full-text-label">Toàn bộ văn bản</span>"${escapeHtml(fullText)}"
        </div>`;

    resultDiv.innerHTML = html;
    lastResultText = data.text || fullText;
    document.getElementById('copy-btn').hidden = !lastResultText;
    animateConfidenceBars();
}

function displayProcessingSteps(steps) {
    const stepsDiv = document.getElementById('processing-steps');
    let html = '';

    steps.forEach((step, index) => {
        const imgSrc = step.image.startsWith('data:') ? step.image : `data:image/png;base64,${step.image}`;
        html += `
            <div class="step-card" style="animation-delay: ${index * 0.05}s">
                <img src="${imgSrc}" alt="${escapeHtml(step.name)}" class="step-image">
                <div class="step-title">${escapeHtml(step.name)}</div>
                ${step.shape ? `<div class="step-meta">${step.shape[1]}×${step.shape[0]}</div>` : ''}
            </div>`;
    });

    stepsDiv.innerHTML = html;
}

// ---------- Image upload ----------
function handleImageUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    if (!file.type.startsWith('image/')) {
        showToast('Vui lòng chọn file ảnh!', true);
        event.target.value = '';
        return;
    }

    const reader = new FileReader();
    reader.onload = function (e) {
        const img = new Image();
        img.onload = function () {
            const maxWidth = Math.min(window.innerWidth - 60, 1200);
            const maxHeight = Math.min(window.innerHeight - 300, 800);

            let newWidth = img.width;
            let newHeight = img.height;

            if (newWidth > maxWidth) {
                const ratio = maxWidth / newWidth;
                newWidth = maxWidth;
                newHeight = Math.round(newHeight * ratio);
            }

            if (newHeight > maxHeight) {
                const ratio = maxHeight / newHeight;
                newHeight = maxHeight;
                newWidth = Math.round(newWidth * ratio);
            }

            newWidth = Math.max(newWidth, 200);
            newHeight = Math.max(newHeight, 100);

            canvas.width = newWidth;
            canvas.height = newHeight;
            canvas.style.maxWidth = '100%';
            canvas.style.height = 'auto';

            ctx.fillStyle = 'white';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0, newWidth, newHeight);
            updateCanvasContext();

            undoStack = [];
            saveCanvasState();
            event.target.value = '';
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

function resetCanvasSize() {
    canvas.style.maxWidth = '';
    canvas.style.height = '';
    adjustCanvasSize();
    clearCanvas();
}

// ---------- Init ----------
window.addEventListener('load', function () {
    initTheme();
    initCanvas();
    // Sync beam row visibility with default selection
    document.getElementById('beam-row').style.display =
        document.getElementById('decode-mode').value === 'beam' ? '' : 'none';
    // Warm up the model in the background while the user draws, so the first
    // predict doesn't pay the model-load cost (free tier sleeps when idle)
    fetch('/warmup', { method: 'POST' }).catch(() => {});
});

window.addEventListener('resize', function () {
    // Only auto-resize when canvas is at default size (no uploaded image)
    if (canvas.width <= 900 && canvas.height <= 600) {
        adjustCanvasSize();
    }
});
