#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PA_extractor_head_tail20.py — LFCC extraction using head & tail 20%, saving .mat as v7.3 via hdf5storage.
ROOT hard-coded, audio loading via pysoundfile + ffmpeg fallback (Windows path), no librosa.
"""

import os
import sys
import io
import subprocess
import numpy as np
import multiprocessing as mp
from scipy.fftpack import dct
import soundfile as sf
from tqdm import tqdm

# ensure hdf5storage is installed
try:
    import hdf5storage
except ImportError:
    print("Missing hdf5storage: pip install hdf5storage")
    sys.exit(1)

# ------------------ Configuration ------------------
# Hard-code root directory of ASVspoof2019 PA data
ROOT = r"D:\NECTEC\automatic speaker verification\PA\PA"
# Subsampling protocol
DS = {"bonafide": 1, "spoof": 1}
# LFCC parameters
WIN_MS, NFFT = 30, 1024
NF, NC, LO, HI = 70, 19, 0, 8000
# Full path to ffmpeg executable on Windows
FFMPEG = r"D:\NECTEC\automatic speaker verification\venv310\Scripts\ffmpeg.exe"

# ------------------ Helpers ------------------

def read_audio(path):
    """
    Read audio via soundfile; fallback to ffmpeg pipe if SF decoder fails.
    """
    try:
        data, sr = sf.read(path)
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        return data.astype(np.float32), sr
    except Exception:
        # fallback via ffmpeg decode to WAV in-memory
        cmd = [
            FFMPEG, "-hide_banner", "-loglevel", "error",
            "-i", path,
            "-f", "wav", "-acodec", "pcm_f32le", "-ac", "1", "pipe:"
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            print(f"FFmpeg decode failed for {path}: {proc.stderr.decode().strip()}")
            return None, None
        buf = io.BytesIO(proc.stdout)
        data, sr = sf.read(buf)
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        return data.astype(np.float32), sr


def enframe(sig, win_len, hop):
    n = len(sig)
    if n < win_len:
        sig = np.pad(sig, (0, win_len-n), 'constant')
        n = win_len
    num = 1 + (n - win_len) // hop
    shape = (num, win_len)
    strides = (hop * sig.strides[0], sig.strides[0])
    return np.lib.stride_tricks.as_strided(sig, shape=shape, strides=strides)


def lfcc_bp(sig, sr, win_ms, nfft, nf, nc, lo, hi):
    x = np.append(sig[0], sig[1:] - 0.97 * sig[:-1])
    win_len = int(win_ms * sr / 1000)
    hop = win_len // 2
    frames = enframe(x, win_len, hop) * np.hamming(win_len)
    mag = np.abs(np.fft.rfft(frames, nfft))
    pw = mag**2 / nfft

    freqs = np.linspace(lo, hi, nf + 2)
    bins = np.floor((nfft + 1) * freqs / sr).astype(int)
    fb = np.zeros((nf, nfft // 2 + 1))
    for j in range(nf):
        fb[j, bins[j]: bins[j+1]] = np.linspace(0, 1, bins[j+1] - bins[j])
        fb[j, bins[j+1]: bins[j+2]] = np.linspace(1, 0, bins[j+2] - bins[j+1])

    feat = np.log(np.maximum(pw.dot(fb.T), np.finfo(float).eps))
    ceps = dct(feat, norm='ortho')[:, :nc]

    def delta(x, N=2):
        den = 2 * sum(i*i for i in range(1, N+1))
        pad = np.pad(x, ((N,N),(0,0)), 'edge')
        out = np.zeros_like(x)
        for t in range(len(x)):
            out[t] = sum(n * (pad[t+N+n] - pad[t+N-n]) for n in range(1, N+1)) / den
        return out

    return ceps, delta(ceps), delta(delta(ceps))


def process_file(task):
    fid, subset, label = task
    folder = os.path.join(ROOT, f"ASVspoof2019_PA_{subset}", "flac")
    path = os.path.join(folder, f"{fid}.flac")
    if not os.path.isfile(path):
        return None

    audio, sr = read_audio(path)
    if audio is None:
        return None

    n = len(audio)
    H, T, ext = 0.20, 0.20, 0.05
    head = audio[:int(n*H)]
    tail = audio[int(n*(1-T)):]
    extra = int(len(head)*ext)
    seg1 = audio[max(0, 0-extra): len(head)+extra]
    seg2 = audio[int(n*(1-T))-extra: int(n*(1-T))+len(tail)+extra]
    sig = np.concatenate((seg1, seg2))

    c, d1, d2 = lfcc_bp(sig, sr, WIN_MS, NFFT, NF, NC, LO, HI)
    return fid, label, np.vstack((c, d1, d2))


def main():
    for subset in ("train", "dev", "eval"):
        proto_dir = os.path.join(ROOT, "ASVspoof2019_PA_cm_protocols")

        # เปลี่ยนชื่อไฟล์โปรโตคอลให้ตรงกับไฟล์จริงในโฟลเดอร์ของคุณ
        proto_map = {
            "train": "ASVspoof2019.PA.cm.train.trn.txt",
            "dev":   "ASVspoof2019.PA.cm.dev.trl.txt",
            "eval":  "ASVspoof2019.PA.cm.eval.trl.txt",
        }
        proto_fn = proto_map[subset]
        proto_path = os.path.join(proto_dir, proto_fn)

        if not os.path.isfile(proto_path):
            print(f"Missing protocol: {proto_path}")
            continue

        with open(proto_path) as f:
            lines = [ln.split() for ln in f]
        _, ids, *_, labels = zip(*lines)

        bon = [i for i, l in enumerate(labels) if l == "bonafide"][::DS["bonafide"]]
        spo = [i for i, l in enumerate(labels) if l == "spoof"][::DS["spoof"]]
        tasks = [(ids[i], subset, labels[i]) for i in bon + spo]

        results = []
        with mp.Pool(mp.cpu_count()) as pool:
            for res in tqdm(pool.imap_unordered(process_file, tasks), total=len(tasks), desc=f"Proc {subset}"):
                if res:
                    results.append(res)

        bona  = [feat for _, lbl, feat in results if lbl == "bonafide"]
        spoof = [feat for _, lbl, feat in results if lbl == "spoof"]

        hdf5storage.savemat(f"LFCC_PA_bonafide_{subset}.mat", {"genuineFeatureCell": bona}, format="7.3")
        hdf5storage.savemat(f"LFCC_PA_spoof_{subset}.mat",    {"spoofFeatureCell": spoof}, format="7.3")
        print(f"✔ {subset}: {len(bona)} bona, {len(spoof)} spoof")

if __name__ == '__main__':
    main()
