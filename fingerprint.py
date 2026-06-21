import librosa
import numpy as np
import scipy.ndimage as ndimage
import pickle
import os

# ── Constants (must match how you built the database) ──────────
N_FFT        = 2048
HOP_LENGTH   = N_FFT // 4   # 512
NEIGHBORHOOD = (20, 20)
PERCENTILE   = 75
FAN          = 5
MAX_DT       = 2.0

def load_audio(path, sr=None):
    y, sr = librosa.load(path, sr=sr, mono=True)
    return y, sr

def compute_spectrogram(y, sr):
    D    = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    return S_db

def get_peaks(S_db, sr):
    local_max   = ndimage.maximum_filter(S_db, size=NEIGHBORHOOD) == S_db
    loud_enough = S_db > np.percentile(S_db, PERCENTILE)
    peak_fi, peak_ti = np.where(local_max & loud_enough)
    times = librosa.frames_to_time(peak_ti, sr=sr, hop_length=HOP_LENGTH)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)[peak_fi]
    return times, freqs

def build_hashes(peak_times, peak_freqs):
    peaks_sorted = sorted(zip(peak_times, peak_freqs), key=lambda x: x[0])
    hashes = []
    for i, (t1, f1) in enumerate(peaks_sorted):
        for j in range(1, FAN + 1):
            if i + j < len(peaks_sorted):
                t2, f2 = peaks_sorted[i + j]
                dt = round(t2 - t1, 3)
                if 0 < dt <= MAX_DT:
                    hashes.append(((round(f1,1), round(f2,1), dt), t1))
    return hashes

def identify(query_path, db_path="database/song_db.pkl"):
    # Load database
    with open(db_path, 'rb') as f:
        song_database = pickle.load(f)

    # Fingerprint query
    y, sr    = load_audio(query_path)
    S_db     = compute_spectrogram(y, sr)
    p_times, p_freqs = get_peaks(S_db, sr)
    hashes   = build_hashes(p_times, p_freqs)

    # Match hashes
    offset_tallies = {}
    for hk, q_time in hashes:
        if hk in song_database:
            for song_name, db_time in song_database[hk]:
                offset = round(db_time - q_time, 2)
                offset_tallies.setdefault(song_name, {})
                offset_tallies[song_name][offset] = \
                    offset_tallies[song_name].get(offset, 0) + 1

    # Find best match
    best_song, best_count = "Unknown", 0
    all_scores = {}
    for song, offsets in offset_tallies.items():
        top = max(offsets.values())
        all_scores[song] = top
        if top > best_count:
            best_count, best_song = top, song

    return best_song, S_db, p_times, p_freqs, all_scores, offset_tallies.get(best_song, {})
