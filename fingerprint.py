import librosa
import numpy as np

def load_audio(filepath, sr=22050):
    y, sr = librosa.load(filepath, sr=sr, mono=True)
    return y, sr

import librosa.display
import matplotlib.pyplot as plt

def compute_spectrogram(y, sr, n_fft=4096, hop_length=512):
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    return S_db, sr, hop_length

from scipy.ndimage import maximum_filter

def get_peaks(S_db, threshold_db=-40, neighborhood=20):
    # Local maxima using a sliding max filter
    local_max = maximum_filter(S_db, size=neighborhood)
    peaks = (S_db == local_max) & (S_db > threshold_db)
    freq_idx, time_idx = np.where(peaks)
    return list(zip(time_idx, freq_idx))   # (time, freq) pairs

def build_hashes(peaks, fan_value=15, time_delta_max=200):
    peaks_sorted = sorted(peaks, key=lambda x: x[0])  # sort by time
    hashes = []
    for i, (t1, f1) in enumerate(peaks_sorted):
        for j in range(1, fan_value + 1):
            if i + j < len(peaks_sorted):
                t2, f2 = peaks_sorted[i + j]
                dt = t2 - t1
                if 0 < dt <= time_delta_max:
                    h = hash((f1, f2, dt))
                    hashes.append((h, t1))   # (hash_value, time_offset)
    return hashes

import os, pickle

def build_database(songs_folder, db_path="database/song_db.pkl"):
    db = {}   # { hash_value: [(song_name, time_offset), ...] }
    for filename in os.listdir(songs_folder):
        if filename.endswith(('.mp3', '.wav', '.flac')):
            path = os.path.join(songs_folder, filename)
            y, sr = load_audio(path)
            S_db, _, _ = compute_spectrogram(y, sr)
            peaks = get_peaks(S_db)
            hashes = build_hashes(peaks)
            song_name = os.path.splitext(filename)[0]
            for h, t in hashes:
                db.setdefault(h, []).append((song_name, t))
    with open(db_path, 'wb') as f:
        pickle.dump(db, f)
    print(f"Database saved with {len(db)} hashes.")

from collections import defaultdict

def identify(query_path, db_path="database/song_db.pkl"):
    with open(db_path, 'rb') as f:
        db = pickle.load(f)

    y, sr = load_audio(query_path)
    S_db, _, hop = compute_spectrogram(y, sr)
    peaks = get_peaks(S_db)
    hashes = build_hashes(peaks)

    # Tally offset matches per song
    matches = defaultdict(lambda: defaultdict(int))
    for h, t_query in hashes:
        if h in db:
            for song_name, t_db in db[h]:
                offset = t_db - t_query
                matches[song_name][offset] += 1

    # Find the song with the most aligned offsets
    best_song, best_count = None, 0
    all_scores = {}
    for song, offsets in matches.items():
        top = max(offsets.values())
        all_scores[song] = top
        if top > best_count:
            best_count = top
            best_song = song

    return best_song, S_db, peaks, all_scores
