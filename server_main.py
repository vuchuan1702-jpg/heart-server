"""
FastAPI server — nhận file WAV từ ESP32, chạy model, trả kết quả
Deploy lên Render: https://render.com

Cấu trúc thư mục:
  server/
  ├── main.py          ← file này
  ├── requirements.txt
  └── model/
      └── heart_model.pkl  ← model train từ Colab (upload lên đây)
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import io
import wave
import struct
import os
import pickle
from scipy.signal import butter, filtfilt, medfilt
import librosa

app = FastAPI(title="Heart Sound Classifier")

# ══════════════════════════════════════════════
# LOAD MODEL (train từ Colab, lưu bằng pickle/joblib)
# ══════════════════════════════════════════════
MODEL_PATH = "model/heart_model.pkl"
model = None

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print(f"[OK] Loaded model từ {MODEL_PATH}")
    else:
        print(f"[WARN] Không tìm thấy model tại {MODEL_PATH} — chạy demo mode")


# ══════════════════════════════════════════════
# PIPELINE XỬ LÝ AUDIO
# ══════════════════════════════════════════════
def read_wav_bytes(data: bytes) -> tuple[np.ndarray, int]:
    """Đọc WAV bytes → numpy array float32 + sample rate."""
    with wave.open(io.BytesIO(data), 'rb') as wf:
        sr        = wf.getframerate()
        n_frames  = wf.getnframes()
        n_ch      = wf.getnchannels()
        raw       = wf.readframes(n_frames)

    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    if n_ch == 2:
        samples = samples[::2]  # lấy kênh trái nếu stereo
    samples /= 32768.0          # chuẩn hóa [-1, 1]
    return samples, sr


def denoise(sig: np.ndarray, sr: int) -> np.ndarray:
    """Lọc nhiễu cơ bản: DC + spike + bandpass."""
    sig = sig - np.mean(sig)
    sig = medfilt(sig.astype(np.float64), kernel_size=5)
    nyq  = sr / 2
    low  = 20   / nyq
    high = min(500, nyq * 0.95) / nyq
    if low < high and len(sig) > 20:
        b, a = butter(4, [low, high], btype='band')
        sig  = filtfilt(b, a, sig)
    return sig.astype(np.float32)


def extract_features(sig: np.ndarray, sr: int) -> np.ndarray:
    """
    Trích xuất features để đưa vào model.
    Điều chỉnh cho khớp với features bạn dùng khi train trên Colab.
    """
    # MFCC — 13 hệ số, lấy mean và std → 26 features
    mfccs      = librosa.feature.mfcc(y=sig, sr=sr, n_mfcc=13)
    mfcc_mean  = np.mean(mfccs, axis=1)
    mfcc_std   = np.std(mfccs,  axis=1)

    # Chroma
    chroma     = librosa.feature.chroma_stft(y=sig, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    # RMS energy
    rms        = librosa.feature.rms(y=sig)
    rms_mean   = np.mean(rms)

    # Zero crossing rate
    zcr        = librosa.feature.zero_crossing_rate(sig)
    zcr_mean   = np.mean(zcr)

    features = np.concatenate([
        mfcc_mean, mfcc_std,    # 26
        chroma_mean,            # 12
        [rms_mean, zcr_mean],   # 2
    ])                          # tổng: 40 features

    return features.astype(np.float32)


# ══════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════
@app.get("/")
def root():
    return {"status": "ok", "message": "Heart Sound Classifier API"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Nhận file WAV từ ESP32, trả về kết quả chẩn đoán.

    Response JSON:
    {
        "label":      "normal" | "abnormal",
        "confidence": 0.92,
        "message":    "Binh thuong"
    }
    """
    # Kiểm tra định dạng
    if not file.filename.endswith(".wav"):
        raise HTTPException(400, "Chỉ chấp nhận file WAV")

    try:
        wav_bytes = await file.read()

        # Đọc + lọc
        sig, sr = read_wav_bytes(wav_bytes)
        sig     = denoise(sig, sr)

        # Demo mode nếu chưa có model
        if model is None:
            return JSONResponse({
                "label":      "demo",
                "confidence": 0.0,
                "message":    "Model chua load. Upload model.pkl vao thu muc model/",
                "duration_s": round(len(sig) / sr, 2),
            })

        # Trích features + predict
        features = extract_features(sig, sr).reshape(1, -1)
        pred     = model.predict(features)[0]

        # Lấy confidence nếu model hỗ trợ predict_proba
        confidence = 0.0
        if hasattr(model, "predict_proba"):
            proba      = model.predict_proba(features)[0]
            confidence = float(np.max(proba))

        # Map nhãn → thông báo
        label_map = {
            0: ("normal",   "Binh thuong"),
            1: ("abnormal", "Bat thuong — can kham bac si"),
            "normal":   ("normal",   "Binh thuong"),
            "abnormal": ("abnormal", "Bat thuong — can kham bac si"),
        }
        label, message = label_map.get(pred, (str(pred), str(pred)))

        return JSONResponse({
            "label":      label,
            "confidence": round(confidence, 3),
            "message":    message,
            "duration_s": round(len(sig) / sr, 2),
        })

    except Exception as e:
        raise HTTPException(500, f"Loi xu ly: {str(e)}")
