# -*- coding: utf-8 -*-
import os
import io
import tempfile
import subprocess
import numpy as np
import soundfile as sf
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Tuple, Optional
from tensorflow.keras.models import load_model
from scipy.fftpack import dct
from sqlalchemy.orm import Session
from datetime import datetime
from passlib.context import CryptContext
from LFCC_pipeline import lfcc as lfcc_ref  # ใช้ LFCC pipeline เดียวกับสคริปต์

# พยายามใช้ librosa ถ้ามี (แนะนำให้ติดตั้ง)
try:
    import librosa
    _HAS_LIBROSA = True
except Exception:
    _HAS_LIBROSA = False

# Database import
from database.database import get_db, SessionLocal
from database.models import Prediction, User

logging.basicConfig(level=logging.INFO)

# ================== Paths (แก้ให้ตรงเครื่องคุณ) ==================
FFMPEG_PATH        = r"D:\NECTEC\automatic speaker verification\venv310\Scripts\ffmpeg.exe"
MODEL_PA_PATH      = r"D:\NECTEC\ASV\backend\models\PA.h5"
MODEL_LA_PATH      = r"D:\NECTEC\ASV\backend\models\LA.h5"
MODEL_LFCC_MMS     = r"D:\NECTEC\ASV\backend\models\lfcc_MMS + Genuine.h5"
MODEL_MFCC_MMS     = r"D:\NECTEC\ASV\backend\models\Mfcc_MMS + Genuine.h5"
# ✅ VAJA รุ่น LFCC / MFCC
MODEL_LFCC_VAJA    = r"D:\NECTEC\ASV\backend\models\lfcc_VAJA + Genuine.h5"
MODEL_MFCC_VAJA    = r"D:\NECTEC\ASV\backend\models\Mfcc_VAJA + Genuine.h5"

# ================== FastAPI & CORS ==================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================== Models ==================
model_pa = None
model_la = None
model_lfcc_mms = None
model_mfcc_mms = None
model_lfcc_vaja = None
model_mfcc_vaja = None

# ----- (A) ResNet34 สำหรับ PA/LA (ต้องรู้ input_shape ตอนสร้างกราฟ) -----
try:
    from classification_models.keras import Classifiers
    ResNet34, _ = Classifiers.get('resnet34')

    if os.path.exists(MODEL_PA_PATH):
        model_pa = ResNet34(input_shape=(57, 600, 1), classes=2)
        model_pa.load_weights(MODEL_PA_PATH)
        logging.info(f"✅ PA Model loaded: {MODEL_PA_PATH}")
    else:
        logging.error(f"❌ PA model file not found: {MODEL_PA_PATH}")

    if os.path.exists(MODEL_LA_PATH):
        model_la = ResNet34(input_shape=(57, 746, 1), classes=2)
        model_la.load_weights(MODEL_LA_PATH)
        logging.info(f"✅ LA Model loaded: {MODEL_LA_PATH}")
    else:
        logging.error(f"❌ LA model file not found: {MODEL_LA_PATH}")
except Exception:
    logging.exception("❌ Model loading error (PA/LA)")

# ----- (B) โมเดล .h5 ที่บันทึกทั้งกราฟ (LFCC/MFCC - MMS/VAJA) -----
def _safe_load_full_model(path: str):
    if not os.path.exists(path):
        logging.error(f"❌ Model file not found: {path}")
        return None
    try:
        m = load_model(path, compile=False)
        logging.info(f"✅ Loaded full model: {path} | input_shape={m.input_shape}")
        return m
    except Exception:
        logging.exception(f"❌ Failed to load full model: {path}")
        return None

# MMS/VAJA
model_lfcc_mms  = _safe_load_full_model(MODEL_LFCC_MMS)    # LFCC_V2
model_mfcc_mms  = _safe_load_full_model(MODEL_MFCC_MMS)    # MFCC
model_lfcc_vaja = _safe_load_full_model(MODEL_LFCC_VAJA)   # LFCC_V2
model_mfcc_vaja = _safe_load_full_model(MODEL_MFCC_VAJA)   # MFCC

# ================== Audio I/O ==================
def read_audio(path: str) -> Tuple[Optional[np.ndarray], Optional[int]]:
    try:
        cmd = [
            FFMPEG_PATH, "-hide_banner", "-loglevel", "error",
            "-i", path, "-f", "wav", "-acodec", "pcm_f32le", "-ac", "1", "pipe:"
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.decode("utf-8", "ignore"))
        buf = io.BytesIO(proc.stdout)
        data, sr = sf.read(buf)
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        return data.astype(np.float32), sr
    except Exception as e:
        logging.error(f"FFmpeg decode failed: {e}")
        return None, None

# ================== Framing ==================
def enframe(sig: np.ndarray, win_len: int, hop: int) -> np.ndarray:
    sig = np.ascontiguousarray(sig, dtype=np.float32)
    if len(sig) < win_len:
        sig = np.pad(sig, (0, win_len - len(sig)))
    num = 1 + (len(sig) - win_len) // hop
    num = max(1, num)
    shape = (num, win_len)
    strides = (hop * sig.strides[0], sig.strides[0])
    return np.lib.stride_tricks.as_strided(sig, shape=shape, strides=strides)

# ================== Utilities ==================
def _pre_emphasis(x: np.ndarray, alpha: float = 0.97) -> np.ndarray:
    if x.size == 0:
        return x.astype(np.float32)
    y = np.empty_like(x, dtype=np.float32)
    y[0] = x[0]
    y[1:] = x[1:] - alpha * x[:-1]
    return y

def head_tail_20_concat(signal: np.ndarray) -> np.ndarray:
    """ตัดเอาช่วงต้น 20% + ท้าย 20% พร้อมกัน (เผื่อ overlap 5%)"""
    n = len(signal)
    H, T, ext = 0.20, 0.20, 0.05
    head = signal[:int(n * H)]
    tail = signal[int(n * (1 - T)):]
    extra = int(len(head) * ext)
    seg1 = signal[max(0, 0 - extra): len(head) + extra]
    pivot = int(n * (1 - T))
    seg2 = signal[max(0, pivot - extra): min(n, pivot + len(tail) + extra)]
    return np.concatenate((seg1, seg2))

def deltas(x: np.ndarray, width: int = 3) -> np.ndarray:
    from scipy.signal import lfilter
    hlen = width // 2
    win = np.arange(hlen, -hlen - 1, -1, dtype=np.float32)
    left = np.repeat(x[:, [0]], hlen, axis=1)
    right = np.repeat(x[:, [-1]], hlen, axis=1)
    xx = np.concatenate([left, x, right], axis=1)
    d = lfilter(win, 1, xx)
    return d[:, hlen * 2 :]

# ================== Feature Backends (3 แบบ) ==================
# ---- 1) LFCC_V1 : สำหรับ PA/LA ----
def lfcc_v1(sig: np.ndarray, sr: int,
            win_ms=30, nfft=1024, nf=70, nc=19, lo=0, hi=8000) -> np.ndarray:
    """
    LFCC_V1 = pipeline ที่ใช้กับ PA/LA → คืน (T, 19)
    """
    sig = np.ascontiguousarray(sig)
    x = np.append(sig[0], sig[1:] - 0.97 * sig[:-1])  # pre-emphasis
    win_len = int(win_ms * sr / 1000)
    hop = max(1, win_len // 2)
    frames = enframe(x, win_len, hop) * np.hamming(win_len)
    mag = np.abs(np.fft.rfft(frames, nfft))
    pw = (mag ** 2) / nfft
    nyq = sr / 2.0
    hi_eff = min(hi, nyq)
    freqs = np.linspace(lo, hi_eff, nf + 2)
    bins = np.floor((nfft + 1) * freqs / sr).astype(int)
    maxbin = nfft // 2
    bins = np.clip(bins, 0, maxbin)
    fb = np.zeros((nf, maxbin + 1), dtype=np.float32)
    for j in range(nf):
        a, b, c = bins[j], bins[j+1], bins[j+2]
        if b > a:
            fb[j, a:b] = np.linspace(0.0, 1.0, b - a, endpoint=False)
        if c > b:
            fb[j, b:c] = np.linspace(1.0, 0.0, c - b, endpoint=False)
    feat = np.log(np.maximum(pw @ fb.T, np.finfo(float).eps))
    ceps = dct(feat, norm='ortho')[:, :nc]
    return ceps  # (T, 19)

# ---- 2) LFCC_V2 : สำหรับ MMS/VAJA (ฝั่ง LFCC) ----
def lfcc_v2(sig: np.ndarray, sr: int,
            win_ms=30, nfft=1024, nf=70, nc=19, lo=0, hi=8000) -> np.ndarray:
    """
    ใช้ LFCC pipeline เดียวกับสคริปต์ (MMS/VAJA)
    """
    # หมายเหตุ: เลือก nc=19 ตาม baseline; ถ้าตอน train ใช้ 20 ให้ปรับตรงนี้
    return lfcc_ref(sig=sig, fs=sr, num_ceps=19, low_freq=0, high_freq=4000)

# ---- 3) MFCC : สำหรับ MMS/VAJA (ฝั่ง MFCC) ----
def _mfcc_librosa(
    y: np.ndarray,
    sr: int,
    num_ceps: int,
    low_freq: int,
    high_freq: int,
    n_mels: int = 70,
    win_ms: float = 30.0,
    hop_ms: float = 15.0,
) -> np.ndarray:
    n_fft = int(round(win_ms * 0.001 * sr))
    hop_length = int(round(hop_ms * 0.001 * sr))
    n_fft = max(n_fft, 256)

    fmin = max(0, low_freq) if low_freq else 0
    fmax = min(high_freq, sr // 2) if (high_freq and high_freq > 0) else None

    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=num_ceps,
        n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, fmin=fmin, fmax=fmax,
        center=False, htk=True
    )  # (num_ceps, T)
    return mfcc.astype(np.float32, copy=False)

def _mfcc_minimal(
    y: np.ndarray,
    sr: int,
    num_ceps: int,
    low_freq: int,
    high_freq: int,
    n_mels: int = 70,
    win_ms: float = 30.0,
    hop_ms: float = 15.0,
) -> np.ndarray:
    n_fft = int(round(win_ms * 0.001 * sr))
    hop = int(round(hop_ms * 0.001 * sr))
    n_fft = max(n_fft, 256)
    win = np.hamming(n_fft).astype(np.float32)

    def _frame_signal(sig, frame_len, hop_len):
        if len(sig) < frame_len:
            sig = np.pad(sig, (0, frame_len - len(sig)), mode="constant")
        num_frames = 1 + (len(sig) - frame_len) // hop_len
        idx = np.tile(np.arange(frame_len), (num_frames, 1)) + np.tile(
            np.arange(0, num_frames * hop_len, hop_len), (frame_len, 1)
        ).T
        frames = sig[idx.astype(np.int32, copy=False)]
        return frames.T  # (frame_len, T)

    frames = _frame_signal(y, n_fft, hop)          # (n_fft, T)
    spec = np.fft.rfft(frames * win[:, None], n=n_fft, axis=0)
    mag = np.abs(spec).astype(np.float32)          # (n_fft//2+1, T)
    pow_spec = (mag ** 2) / float(n_fft)

    def _hz_to_mel(f):  # HTK
        return 2595.0 * np.log10(1.0 + f / 700.0)
    def _mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    fmin = max(0, low_freq) if low_freq else 0
    fmax = min(high_freq, sr // 2) if (high_freq and high_freq > 0) else (sr // 2)

    m_min = _hz_to_mel(fmin); m_max = _hz_to_mel(fmax)
    m_pts = np.linspace(m_min, m_max, n_mels + 2, dtype=np.float32)
    f_pts = _mel_to_hz(m_pts)
    bins = np.floor((n_fft + 1) * f_pts / sr).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)

    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        if c <= l: c = l + 1
        if r <= c: r = c + 1
        fb[m - 1, l:c] = (np.arange(l, c) - l) / max(1, (c - l))
        fb[m - 1, c:r] = (r - np.arange(c, r)) / max(1, (r - c))

    melE = np.maximum(fb @ pow_spec, 1e-10)       # (n_mels, T)
    log_mel = np.log(melE, dtype=np.float32)

    # DCT-II ortho
    M, T = log_mel.shape
    n = np.arange(M, dtype=np.float32)
    k = np.arange(num_ceps, dtype=np.float32)[:, None]
    basis = np.cos(np.pi * (n + 0.5) * k / M).astype(np.float32)
    scale = np.sqrt(2.0 / M).astype(np.float32)
    mfcc = (basis @ log_mel) * scale
    mfcc[0, :] *= np.sqrt(0.5).astype(np.float32)
    return mfcc.astype(np.float32)  # (num_ceps, T)

def mfcc_core(sig: np.ndarray, sr: int, n_mfcc: int = 20) -> np.ndarray:
    """
    คืน MFCC shape (T, n_mfcc) พร้อม pre-emphasis, 30ms/15ms
    """
    x = _pre_emphasis(np.asarray(sig, dtype=np.float32), alpha=0.97)
    if _HAS_LIBROSA:
        mfcc = _mfcc_librosa(x, sr, num_ceps=n_mfcc, low_freq=0, high_freq=4000, n_mels=70)
    else:
        mfcc = _mfcc_minimal(x, sr, num_ceps=n_mfcc, low_freq=0, high_freq=4000, n_mels=70)
    return mfcc.T  # (T, n_mfcc)

# ============== Dispatcher: 3 แบบฟีเจอร์ ==============
def extract_features(signal: np.ndarray, sr: int, mode: str) -> Optional[np.ndarray]:
    """
    mode ∈ {'lfcc_v1', 'lfcc_v2', 'mfcc'}
      - lfcc_v1 → PA/LA (57 = 19×3)
      - lfcc_v2 → MMS/VAJA (57 = 19×3) ใช้ LFCC_pipeline.lfcc
      - mfcc    → MMS/VAJA (60 = 20×3)
    คืน (F, T)
    """
    try:
        # LFCC: ใช้ทั้งสัญญาณ (ให้เหมือนสคริปต์) / MFCC: ใช้ head+tail
        if mode == 'mfcc':
            sig = head_tail_20_concat(signal)
        else:
            sig = np.asarray(signal, dtype=np.float32)

        if mode == 'lfcc_v1':
            c = lfcc_v1(sig, sr, nc=19)     # (T,19)
            static = c.T                    # (19,T)
            d1 = deltas(static); d2 = deltas(d1)
            feats = np.vstack([static, d1, d2])  # (57,T)

        elif mode == 'lfcc_v2':
            c = lfcc_v2(sig, sr, nc=19)     # (T,19) จาก lfcc_ref
            static = c.T
            d1 = deltas(static); d2 = deltas(d1)
            feats = np.vstack([static, d1, d2])  # (57,T)

        elif mode == 'mfcc':
            c = mfcc_core(sig, sr, n_mfcc=20)  # (T,20)
            static = c.T
            d1 = deltas(static); d2 = deltas(d1)
            feats = np.vstack([static, d1, d2])  # (60,T)

        else:
            raise ValueError(f"Unknown feature mode: {mode}")

        return feats
    except Exception:
        logging.exception(f"{mode.upper()} extraction failed")
        return None

def fix_input_shape_to_model(
    feats: np.ndarray,
    model,
    prefer_feature_bins: Optional[int] = None
) -> np.ndarray:
    """
    ปรับ (F,T) ให้เป็น (1, H, W, 1) เข้ากับ model.input_shape
    - รองรับได้ทั้งกรณี H=features หรือ H=time
    """
    if feats is None:
        raise ValueError("features is None")
    if not hasattr(model, "input_shape") or model.input_shape is None:
        raise ValueError("Model has no input_shape; cannot infer target size.")

    _, H, W, C = model.input_shape
    if C != 1:
        raise ValueError(f"Model expects C=1, got C={C}")

    F, T = feats.shape

    def _fit_len(arr_2d: np.ndarray, target: int, axis: int) -> np.ndarray:
        cur = arr_2d.shape[axis]
        if cur > target:
            if axis == 0: arr_2d = arr_2d[:target, :]
            else:         arr_2d = arr_2d[:, :target]
        elif cur < target:
            pad = target - cur
            if axis == 0: arr_2d = np.pad(arr_2d, ((0, pad), (0, 0)), mode='constant')
            else:         arr_2d = np.pad(arr_2d, ((0, 0), (0, pad)), mode='constant')
        return arr_2d

    can_map_F_to_H = (F == H) or (prefer_feature_bins == H)
    can_map_F_to_W = (F == W) or (prefer_feature_bins == W)

    if can_map_F_to_H and not can_map_F_to_W:
        chosen = 'FH'
    elif can_map_F_to_W and not can_map_F_to_H:
        chosen = 'FW'
    elif can_map_F_to_H and can_map_F_to_W:
        chosen = 'FH' if abs(T - W) <= abs(T - H) else 'FW'
    else:
        if H >= 256 and W <= 128:
            chosen = 'FW'
        elif W >= 256 and H <= 128:
            chosen = 'FH'
        else:
            cost_FH = abs(F - H) + abs(T - W)
            cost_FW = abs(F - W) + abs(T - H)
            chosen = 'FH' if cost_FH <= cost_FW else 'FW'

    if chosen == 'FH':
        x2d = feats.copy()
        x2d = _fit_len(x2d, H, axis=0)
        x2d = _fit_len(x2d, W, axis=1)
    else:
        x2d = feats.T.copy()
        x2d = _fit_len(x2d, H, axis=0)
        x2d = _fit_len(x2d, W, axis=1)

    x = x2d[np.newaxis, ..., np.newaxis]
    return x

def _predict_core(file: UploadFile, model, mode: str, label_prefix: str, prefer_feature_bins: Optional[int] = None):
    """อ่านไฟล์ → สกัดฟีเจอร์(3 แบบ) → ปรับรูป → predict → บันทึก DB"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    try:
        signal, sr = read_audio(tmp_path)
        if signal is None or sr is None:
            return {"filename": file.filename, "error": "Failed to read audio"}

        feats = extract_features(signal, sr, mode=mode)  # (F,T)
        if feats is None:
            return {"filename": file.filename, "error": "Feature extraction failed"}

        x = fix_input_shape_to_model(feats, model, prefer_feature_bins=prefer_feature_bins)
        probs = model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))
        label = "Real" if idx == 0 else "Fake"
        conf_prob = float(probs[idx])          # 0..1
        conf_pct  = float(probs[idx] * 100.0)  # 0..100

        with SessionLocal() as db:
            db.add(Prediction(
                filename=file.filename,
                label=f"{label_prefix} - {label}",
                confidence=conf_pct,
                timestamp=datetime.now()
            ))
            db.commit()

        return {
            "filename": file.filename,
            "label": label,
            "confidence": conf_prob,
            "confidence_pct": conf_pct
        }
    except Exception:
        logging.exception(f"{label_prefix} prediction failed")
        return {"filename": file.filename, "error": "Prediction failed"}
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

# ================== Endpoints ==================
# --- PA (LFCC_V1: 57×600) ---
@app.post("/predict_pa")
async def predict_pa(files: List[UploadFile] = File(...)):
    if model_pa is None:
        raise HTTPException(status_code=500, detail="PA model not loaded")
    results = []
    for f in files:
        f.file.seek(0)
        res = _predict_core(f, model_pa, mode='lfcc_v1', label_prefix="PA", prefer_feature_bins=57)
        results.append(res)
    return results

# --- LA (LFCC_V1: 57×746) ---
@app.post("/predict_la")
async def predict_la(files: List[UploadFile] = File(...)):
    if model_la is None:
        raise HTTPException(status_code=500, detail="LA model not loaded")
    results = []
    for f in files:
        f.file.seek(0)
        res = _predict_core(f, model_la, mode='lfcc_v1', label_prefix="LA", prefer_feature_bins=57)
        results.append(res)
    return results

# --- MMS: LFCC_V2 ---
@app.post("/predict_lfcc_mms")
async def predict_lfcc_mms(files: List[UploadFile] = File(...)):
    if model_lfcc_mms is None:
        raise HTTPException(status_code=500, detail="LFCC-MMS model not loaded")
    results = []
    for f in files:
        f.file.seek(0)
        res = _predict_core(f, model_lfcc_mms, mode='lfcc_v2', label_prefix="LFCC_MMS", prefer_feature_bins=57)
        results.append(res)
    return results

# --- MMS: MFCC ---
@app.post("/predict_mfcc_mms")
async def predict_mfcc_mms(files: List[UploadFile] = File(...)):
    if model_mfcc_mms is None:
        raise HTTPException(status_code=500, detail="MFCC-MMS model not loaded")
    results = []
    for f in files:
        f.file.seek(0)
        res = _predict_core(f, model_mfcc_mms, mode='mfcc', label_prefix="MFCC_MMS", prefer_feature_bins=60)
        results.append(res)
    return results

# --- VAJA: LFCC_V2 ---
@app.post("/predict_lfcc_vaja")
async def predict_lfcc_vaja(files: List[UploadFile] = File(...)):
    if model_lfcc_vaja is None:
        raise HTTPException(status_code=500, detail="LFCC-VAJA model not loaded")
    results = []
    for f in files:
        f.file.seek(0)
        res = _predict_core(f, model_lfcc_vaja, mode='lfcc_v2', label_prefix="LFCC_VAJA", prefer_feature_bins=57)
        results.append(res)
    return results

# --- VAJA: MFCC ---
@app.post("/predict_mfcc_vaja")
async def predict_mfcc_vaja(files: List[UploadFile] = File(...)):
    if model_mfcc_vaja is None:
        raise HTTPException(status_code=500, detail="MFCC-VAJA model not loaded")
    results = []
    for f in files:
        f.file.seek(0)
        res = _predict_core(f, model_mfcc_vaja, mode='mfcc', label_prefix="MFCC_VAJA", prefer_feature_bins=60)
        results.append(res)
    return results

# --- History / Delete ---
@app.get("/history")
def get_prediction_history(db: Session = Depends(get_db)):
    results = db.query(Prediction).order_by(Prediction.id.desc()).all()
    return [{
        "id": r.id,
        "filename": r.filename,
        "label": r.label,
        "confidence": f"{r.confidence:.2f}%",
        "timestamp": r.timestamp.isoformat()
    } for r in results]

@app.delete("/history/{prediction_id}")
def delete_prediction(prediction_id: int, db: Session = Depends(get_db)):
    prediction = db.query(Prediction).filter(Prediction.id == prediction_id).first()
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    db.delete(prediction)
    db.commit()
    return {"message": f"Prediction {prediction_id} deleted successfully"}

# (optional auth context)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
