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
    allow_origins=["http://localhost:4200"],  # ปรับ origin ให้ตรงกับ frontend
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

# MMS
model_lfcc_mms  = _safe_load_full_model(MODEL_LFCC_MMS)   # คาดว่าใช้ LFCC
model_mfcc_mms  = _safe_load_full_model(MODEL_MFCC_MMS)   # คาดว่าใช้ MFCC
# ✅ VAJA
model_lfcc_vaja = _safe_load_full_model(MODEL_LFCC_VAJA)  # LFCC-VAJA
model_mfcc_vaja = _safe_load_full_model(MODEL_MFCC_VAJA)  # MFCC-VAJA

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

# ================== LFCC backend ==================
def lfcc_bp(sig: np.ndarray, sr: int, win_ms=30, nfft=1024, nf=70, nc=19, lo=0, hi=8000):
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
    feat = np.log(np.maximum(pw @ fb.T, np.finfo(float).eps))   # (frames, nf)
    ceps = dct(feat, norm='ortho')[:, :nc]                      # (frames, nc)
    return ceps  # (T, 19)

# ================== MFCC backend (อัปเดตตามที่ขอ) ==================
def hz_to_mel(f: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + (f / 700.0))

def mel_to_hz(m: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

from scipy.fft import dct  # คงไว้

def mfcc_bp(
    sig: np.ndarray,
    sr: int,
    win_ms: float = 25.0,   # ✅ 25 ms
    hop_ms: float = 10.0,   # ✅ 10 ms
    nfft: Optional[int] = None,
    n_mels: int = 70,       # ✅ ตามสคริปต์
    n_mfcc: int = 20,       # ✅ 20 ตัว
    fmin: float = 0.0,
    fmax: Optional[float] = 4000.0,  # ✅ ตัดที่ 4 kHz
    htk: bool = True                 # ✅ HTK mel
) -> np.ndarray:
    """
    คืน (T, n_mfcc) โดยทำ pre-emphasis, STFT(center=False), HTK mel, log, DCT-II (ortho)
    """
    # pre-emphasis
    x = np.ascontiguousarray(sig, dtype=np.float32)
    x = np.append(x[0], x[1:] - 0.97 * x[:-1]).astype(np.float32)

    # กำหนด n_fft ตามหน้าต่าง ถ้าไม่ระบุ
    if nfft is None:
        nfft = int(round(win_ms * 0.001 * sr))
        nfft = max(nfft, 256)

    # framing: center=False → ใช้ enframe ของเรา
    win_len = int(round(win_ms * sr / 1000.0))
    hop     = int(round(hop_ms * sr / 1000.0))
    frames  = enframe(x, win_len, hop)
    frames  = frames * np.hamming(win_len).astype(np.float32)

    # power spectrum
    spec = np.fft.rfft(frames, nfft, axis=1)
    mag  = np.abs(spec).astype(np.float32)
    pw   = (mag ** 2) / float(nfft)  # (frames, nfft//2+1)

    # HTK mel fbanks
    def _hz_to_mel_htk(f):  # HTK
        return 2595.0 * np.log10(1.0 + (f / 700.0))
    def _mel_to_hz_htk(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    if fmax is None:
        fmax = sr / 2.0
    fmax = min(fmax, sr / 2.0)

    m_min = _hz_to_mel_htk(fmin) if htk else 1127.0*np.log1p(fmin/700.0)
    m_max = _hz_to_mel_htk(fmax) if htk else 1127.0*np.log1p(fmax/700.0)
    m_pts = np.linspace(m_min, m_max, n_mels + 2, dtype=np.float32)
    f_pts = _mel_to_hz_htk(m_pts) if htk else 700.0*(np.expm1(m_pts/1127.0))
    bins  = np.floor((nfft + 1) * f_pts / sr).astype(int)

    maxbin = nfft // 2
    bins = np.clip(bins, 0, maxbin)

    fb = np.zeros((n_mels, maxbin + 1), dtype=np.float32)
    for j in range(n_mels):
        a, b, c = bins[j], bins[j + 1], bins[j + 2]
        if b > a:
            fb[j, a:b] = np.linspace(0.0, 1.0, b - a, endpoint=False, dtype=np.float32)
        if c > b:
            fb[j, b:c] = np.linspace(1.0, 0.0, c - b, endpoint=False, dtype=np.float32)

    # (frames, n_mels)
    melE = np.maximum(pw @ fb.T, np.finfo(np.float32).eps)
    logE = np.log(melE).astype(np.float32)

    # DCT-II (ortho) → เก็บ 20 ตัวแรก
    ceps = dct(logE, type=2, norm='ortho', axis=1).astype(np.float32)[:, :n_mfcc]  # (frames, 20)
    return ceps

# ================== Utilities ==================
def head_tail_20_concat(signal: np.ndarray) -> np.ndarray:
    """Head 20% + Tail 20% (+5% overlap buffer)"""
    n = len(signal)
    H, T, ext = 0.20, 0.20, 0.05
    head = signal[:int(n * H)]
    tail = signal[int(n * (1 - T)):]
    extra = int(len(head) * ext)
    seg1 = signal[max(0, 0 - extra): len(head) + extra]
    pivot = int(n * (1 - T))
    seg2 = signal[max(0, pivot - extra): min(n, pivot + len(tail) + extra)]
    return np.concatenate((seg1, seg2))

from scipy.signal import lfilter

def _deltas_filter(x: np.ndarray, width: int = 3) -> np.ndarray:
    """
    x: (C, T) คืนค่า (C, T) ของ delta โดยใช้ฟิลเตอร์เชิงเส้นแบบเดียวกับสคริปต์อ้างอิง
    """
    assert width % 2 == 1 and width >= 3, "width ต้องเป็นคี่ ≥3"
    hlen = width // 2
    win = np.arange(hlen, -hlen - 1, -1, dtype=np.float32)  # [1,0,-1] ถ้า width=3
    left  = np.repeat(x[:, [0]], hlen, axis=1)
    right = np.repeat(x[:, [-1]], hlen, axis=1)
    xx = np.concatenate([left, x, right], axis=1)
    d = lfilter(win, 1, xx)
    return d[:, hlen * 2:]

def stack_deltas(static_fts: np.ndarray, width: int = 3) -> np.ndarray:
    """
    static_fts: (F_static, T) -> concat [static, Δ, ΔΔ] = (F_static*3, T)
    ใช้เดลต้าแบบฟิลเตอร์ให้เหมือนสคริปต์ที่คุณส่งมา
    """
    delta  = _deltas_filter(static_fts, width=width)
    deltad = _deltas_filter(delta,      width=width)
    return np.concatenate([static_fts, delta, deltad], axis=0)

def extract_features(signal: np.ndarray, sr: int, mode: str) -> Optional[np.ndarray]:
    """
    'lfcc' -> 20*3=60  | 'mfcc' -> 20*3=60
    """
    try:
        sig = head_tail_20_concat(signal)  # เหมือนเดิม
        if mode == 'lfcc':
            c = lfcc_bp(sig, sr, nc=20)                  # (T, 20)
        elif mode == 'mfcc':
            c = mfcc_bp(sig, sr, win_ms=25.0, hop_ms=10.0,  # ✅ พารามิเตอร์ตามต้นฉบับ
                        n_mels=70, n_mfcc=20, fmin=0.0, fmax=4000.0, htk=True)  # (T, 20)
        else:
            raise ValueError(f"Unknown feature mode: {mode}")

        static = c.T                                     # (F_static=20, T)
        feats  = stack_deltas(static, width=3)           # (60, T) Δ/ΔΔ แบบฟิลเตอร์
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
    ปรับฟีเจอร์ (F,T) ให้เข้ากับ model.input_shape = (None, H, W, 1)
    - รองรับทั้งกรณี H=features,W=time และ H=time,W=features
    - ถ้ามี prefer_feature_bins ให้ใช้เป็นตัวช่วยชี้ว่าแกนไหนคือฟีเจอร์
    """
    if feats is None:
        raise ValueError("features is None")

    if not hasattr(model, "input_shape") or model.input_shape is None:
        raise ValueError("Model has no input_shape; cannot infer target size.")

    _, H, W, C = model.input_shape  # (batch, H, W, C)
    if C != 1:
        raise ValueError(f"Model expects C=1, got C={C}")

    F, T = feats.shape

    # helper: pad/crop along axis to target
    def _fit_len(arr_2d: np.ndarray, target: int, axis: int) -> np.ndarray:
        cur = arr_2d.shape[axis]
        if cur > target:
            if axis == 0:
                arr_2d = arr_2d[:target, :]
            else:
                arr_2d = arr_2d[:, :target]
        elif cur < target:
            pad = target - cur
            if axis == 0:
                arr_2d = np.pad(arr_2d, ((0, pad), (0, 0)), mode='constant')
            else:
                arr_2d = np.pad(arr_2d, ((0, 0), (0, pad)), mode='constant')
        return arr_2d

    # ========== decide mapping ==========
    can_map_F_to_H = (F == H) or (prefer_feature_bins == H)
    can_map_F_to_W = (F == W) or (prefer_feature_bins == W)

    if can_map_F_to_H and not can_map_F_to_W:
        chosen = 'FH'
    elif can_map_F_to_W and not can_map_F_to_H:
        chosen = 'FW'
    elif can_map_F_to_H and can_map_F_to_W:
        # เลือกอันที่เวลา T ใกล้ target มากกว่า
        chosen = 'FH' if abs(T - W) <= abs(T - H) else 'FW'
    else:
        # heuristic
        if H >= 256 and W <= 128:
            chosen = 'FW'  # เวลา→H, ฟีเจอร์→W
        elif W >= 256 and H <= 128:
            chosen = 'FH'
        else:
            cost_FH = abs(F - H) + abs(T - W)
            cost_FW = abs(F - W) + abs(T - H)
            chosen = 'FH' if cost_FH <= cost_FW else 'FW'

    # ========== shape according to mapping ==========
    if chosen == 'FH':
        # F -> H, T -> W
        x2d = feats.copy()
        x2d = _fit_len(x2d, H, axis=0)  # fit features to H
        x2d = _fit_len(x2d, W, axis=1)  # fit time to W
    else:
        # 'FW': F -> W, T -> H  => transpose ก่อน
        x2d = feats.T.copy()            # (T,F)
        x2d = _fit_len(x2d, H, axis=0)  # fit time (axis0) to H
        x2d = _fit_len(x2d, W, axis=1)  # fit features (axis1) to W

    x = x2d[np.newaxis, ..., np.newaxis]  # (1, H, W, 1)
    return x

def _predict_core(file: UploadFile, model, mode: str, label_prefix: str, prefer_feature_bins: Optional[int] = None):
    """อ่านไฟล์ → สกัดฟีเจอร์ → ปรับรูป → predict → บันทึก DB"""
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
        conf_prob = float(probs[idx])         # 0..1
        conf_pct  = float(probs[idx] * 100.0) # 0..100

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
# --- เดิม: PA (LFCC 57×600) ---
@app.post("/predict_pa")
async def predict_pa(files: List[UploadFile] = File(...)):
    if model_pa is None:
        raise HTTPException(status_code=500, detail="PA model not loaded")
    results = []
    for f in files:
        f.file.seek(0)
        res = _predict_core(f, model_pa, mode='lfcc', label_prefix="PA", prefer_feature_bins=57)
        results.append(res)
    return results

# --- เดิม: LA (LFCC 57×746) ---
@app.post("/predict_la")
async def predict_la(files: List[UploadFile] = File(...)):
    if model_la is None:
        raise HTTPException(status_code=500, detail="LA model not loaded")
    results = []
    for f in files:
        f.file.seek(0)
        res = _predict_core(f, model_la, mode='lfcc', label_prefix="LA", prefer_feature_bins=57)
        results.append(res)
    return results

# --- MMS: LFCC/MFCC ---
@app.post("/predict_lfcc_mms")
async def predict_lfcc_mms(files: List[UploadFile] = File(...)):
    if model_lfcc_mms is None:
        raise HTTPException(status_code=500, detail="LFCC-MMS model not loaded")
    results = []
    for f in files:
        f.file.seek(0)
        res = _predict_core(f, model_lfcc_mms, mode='lfcc', label_prefix="LFCC_MMS", prefer_feature_bins=57)
        results.append(res)
    return results

@app.post("/predict_mfcc_mms")
async def predict_mfcc_mms(files: List[UploadFile] = File(...)):
    if model_mfcc_mms is None:
        raise HTTPException(status_code=500, detail="MFCC-MMS model not loaded")
    results = []
    for f in files:
        f.file.seek(0)
        # ✅ บอกชัดว่าแกนฟีเจอร์ = 60 (20 MFCC × 3)
        res = _predict_core(f, model_mfcc_mms, mode='mfcc', label_prefix="MFCC_MMS", prefer_feature_bins=60)
        results.append(res)
    return results

# ✅ VAJA — LFCC/MFCC
@app.post("/predict_lfcc_vaja")
async def predict_lfcc_vaja(files: List[UploadFile] = File(...)):
    if model_lfcc_vaja is None:
        raise HTTPException(status_code=500, detail="LFCC-VAJA model not loaded")
    results = []
    for f in files:
        f.file.seek(0)
        res = _predict_core(f, model_lfcc_vaja, mode='lfcc', label_prefix="LFCC_VAJA", prefer_feature_bins=57)
        results.append(res)
    return results

@app.post("/predict_mfcc_vaja")
async def predict_mfcc_vaja(files: List[UploadFile] = File(...)):
    if model_mfcc_vaja is None:
        raise HTTPException(status_code=500, detail="MFCC-VAJA model not loaded")
    results = []
    for f in files:
        f.file.seek(0)
        # ✅ ชี้แกนฟีเจอร์ = 60 ให้ตรงกับชุด train MFCC
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
