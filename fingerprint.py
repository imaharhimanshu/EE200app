import numpy as np
import librosa
import librosa.display
import pickle
import os
import scipy.ndimage as ndimage


def load_audio(filepath, sr=22050):
    y, sr = librosa.load(filepath, sr=sr, mono=True)
    return y, sr


def compute_spectrogram(y, sr, n_fft=2048):
    hop_length = n_fft // 4
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    times = librosa.times_like(S_db, sr=sr, hop_length=hop_length)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    
    return S_db, times, freqs, hop_length

def find_peaks(S_db, sr, hop_length, threshold_percentile=75):
    neighborhood_size = (20, 20)
    local_max = ndimage.maximum_filter(S_db, size=neighborhood_size) == S_db
    threshold = np.percentile(S_db, threshold_percentile)
    loud_enough = S_db > threshold
    peaks = local_max & loud_enough
    freq_idx, time_idx = np.where(peaks)
    peak_times = librosa.frames_to_time(time_idx, sr=sr, hop_length=hop_length)
    peak_freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)[freq_idx]

    return list(zip(time_idx.tolist(), freq_idx.tolist())), peak_times, peak_freqs


def generate_hashes(peaks_idx, song_name, fan_out=15):
    hashes = {}
    peaks_idx.sort(key=lambda x: x[0])
    for i in range(len(peaks_idx)):
        for j in range(1, fan_out + 1):
            if i + j >= len(peaks_idx):
                break
            t1, f1 = peaks_idx[i]
            t2, f2 = peaks_idx[i + j]
            delta_t = t2 - t1
            if delta_t <= 0:
                continue
            key = (f1, f2, delta_t)
            if key not in hashes:
                hashes[key] = []
            hashes[key].append((song_name, t1))
    return hashes

# ─────────────────────────────────────────
# 5. BUILD DATABASE
# ─────────────────────────────────────────
def build_database(songs_folder, save_path="database.pkl"):
    database = {}
    song_names = []
    files = [f for f in os.listdir(songs_folder)
             if f.lower().endswith((".mp3", ".wav"))]
    for filename in files:
        filepath = os.path.join(songs_folder, filename)
        song_name = os.path.splitext(filename)[0]
        song_names.append(song_name)
        print(f"Indexing: {song_name}")
        try:
            y, sr = load_audio(filepath)
            S_db, times, freqs, hop_length = compute_spectrogram(y, sr)
            peaks_idx, _, _ = find_peaks(S_db, sr, hop_length)
            hashes = generate_hashes(peaks_idx, song_name)
            for key, val in hashes.items():
                if key not in database:
                    database[key] = []
                database[key].extend(val)
        except Exception as e:
            print(f"  ERROR with {filename}: {e}")
    with open(save_path, "wb") as f:
        pickle.dump({"hashes": database, "songs": song_names}, f)
    print(f"\n Done: {len(song_names)} songs, {len(database)} hashes")
    return database, song_names


def identify_clip(query_path, database):
    y, sr = load_audio(query_path)
    S_db, times, freqs, hop_length = compute_spectrogram(y, sr)
    peaks_idx, peak_times, peak_freqs = find_peaks(S_db, sr, hop_length)
    query_hashes = generate_hashes(peaks_idx, "query")
    db_hashes = database["hashes"]
    song_list = database["songs"]
    offset_counts = {song: {} for song in song_list}
    for key, query_entries in query_hashes.items():
        if key in db_hashes:
            for (db_song, db_t1) in db_hashes[key]:
                for (_, q_t1) in query_entries:
                    offset = db_t1 - q_t1
                    offset_counts[db_song][offset] = \
                        offset_counts[db_song].get(offset, 0) + 1
    best_per_song = {
        song: max(offsets.values()) if offsets else 0
        for song, offsets in offset_counts.items()
    }
    matched_song = max(best_per_song, key=best_per_song.get)

    total_query_hashes = len(query_hashes)

    matched_hash_counts = {song: 0 for song in song_list}
    for key, query_entries in query_hashes.items():
        if key in db_hashes:
            for (db_song, db_t1) in db_hashes[key]:
                matched_hash_counts[db_song] += len(query_entries)

    return matched_song, best_per_song, S_db, times, freqs, \
           hop_length, sr, peak_times, peak_freqs, \
           total_query_hashes, matched_hash_counts


def load_database(path="database.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)
