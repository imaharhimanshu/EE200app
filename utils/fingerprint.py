# utils/fingerprint.py

import numpy as np
import librosa
import scipy.ndimage as ndimage


# ==========================================
# Spectrogram Generation
# ==========================================

def generate_spectrogram(
    audio_path,
    n_fft=2048,
    hop_length=512
):
    """
    Load audio and compute spectrogram.
    """

    y, sr = librosa.load(audio_path, mono=True)

    D = librosa.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_length
    )

    S = np.abs(D)

    S_db = librosa.amplitude_to_db(
        S,
        ref=np.max
    )

    return y, sr, S_db


# ==========================================
# Peak Detection (Constellation Map)
# ==========================================

def find_peaks(
    S_db,
    sr,
    n_fft=2048,
    hop_length=512,
    neighborhood_size=(20, 20),
    percentile_threshold=75
):
    """
    Extract strong local maxima from spectrogram.
    """

    local_max = (
        ndimage.maximum_filter(
            S_db,
            size=neighborhood_size
        ) == S_db
    )

    threshold = np.percentile(
        S_db,
        percentile_threshold
    )

    loud_enough = S_db > threshold

    peaks = local_max & loud_enough

    freq_idx, time_idx = np.where(peaks)

    peak_times = librosa.frames_to_time(
        time_idx,
        sr=sr,
        hop_length=hop_length
    )

    peak_freqs = librosa.fft_frequencies(
        sr=sr,
        n_fft=n_fft
    )[freq_idx]

    return peak_times, peak_freqs


# ==========================================
# Hash Generation
# ==========================================

def generate_hashes(
    peak_times,
    peak_freqs,
    target_zone_size=5
):
    """
    Create paired fingerprints.

    Hash format:
    (freq1, freq2, delta_time)
    """

    hashes = []

    for i in range(len(peak_times)):

        for j in range(
            i + 1,
            min(
                i + target_zone_size,
                len(peak_times)
            )
        ):

            f1 = int(peak_freqs[i])
            f2 = int(peak_freqs[j])

            delta_t = round(
                peak_times[j] - peak_times[i],
                2
            )

            if delta_t <= 0:
                continue

            hash_key = (
                f1,
                f2,
                delta_t
            )

            hashes.append(
                (
                    hash_key,
                    peak_times[i]
                )
            )

    return hashes


# ==========================================
# Full Fingerprint Pipeline
# ==========================================

def fingerprint_audio(audio_path):
    """
    Complete fingerprint extraction.
    """

    y, sr, S_db = generate_spectrogram(
        audio_path
    )

    peak_times, peak_freqs = find_peaks(
        S_db,
        sr
    )

    hashes = generate_hashes(
        peak_times,
        peak_freqs
    )

    return {
        "spectrogram": S_db,
        "sample_rate": sr,
        "peak_times": peak_times,
        "peak_freqs": peak_freqs,
        "hashes": hashes
    }
