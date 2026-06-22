# utils/visualizations.py

import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np


# ==========================================
# Spectrogram Plot
# ==========================================

def plot_spectrogram(
    S_db,
    sr,
    hop_length=512
):
    """
    Returns a matplotlib figure
    containing the spectrogram.
    """

    fig, ax = plt.subplots(figsize=(10, 5))

    img = librosa.display.specshow(
        S_db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="hz",
        cmap="magma",
        ax=ax
    )

    ax.set_title("Spectrogram")

    fig.colorbar(
        img,
        ax=ax,
        format="%+2.0f dB"
    )

    plt.tight_layout()

    return fig


# ==========================================
# Constellation Plot
# ==========================================

def plot_constellation(
    S_db,
    sr,
    peak_times,
    peak_freqs,
    hop_length=512
):
    """
    Spectrogram with detected peaks.
    """

    fig, ax = plt.subplots(figsize=(10, 5))

    img = librosa.display.specshow(
        S_db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="hz",
        cmap="magma",
        ax=ax
    )

    ax.scatter(
        peak_times,
        peak_freqs,
        facecolors="none",
        edgecolors="cyan",
        s=25,
        linewidths=1,
        label="Peaks"
    )

    ax.set_title("Constellation Map")
    ax.legend()

    fig.colorbar(
        img,
        ax=ax,
        format="%+2.0f dB"
    )

    plt.tight_layout()

    return fig


# ==========================================
# Offset Histogram
# ==========================================

def plot_offset_histogram(histogram):
    """
    Plot offset voting histogram.
    """

    fig, ax = plt.subplots(figsize=(10, 4))

    if histogram is None or len(histogram) == 0:

        ax.text(
            0.5,
            0.5,
            "No Matches Found",
            ha="center",
            va="center",
            fontsize=14
        )

        ax.set_title("Offset Histogram")

        return fig

    offsets = list(histogram.keys())
    counts = list(histogram.values())

    ax.bar(
        offsets,
        counts
    )

    ax.set_title("Offset Histogram")
    ax.set_xlabel("Offset (seconds)")
    ax.set_ylabel("Match Count")

    plt.tight_layout()

    return fig


# ==========================================
# Optional DFT Plot
# ==========================================

def plot_dft(audio_path):
    """
    Useful for Q3A report.
    """

    y, sr = librosa.load(
        audio_path,
        mono=True
    )

    N = len(y)

    dft = np.fft.rfft(y)

    freqs = np.fft.rfftfreq(
        N,
        d=1.0 / sr
    )

    magnitude = np.abs(dft)

    fig, ax = plt.subplots(
        figsize=(10, 4)
    )

    ax.plot(
        freqs,
        magnitude,
        linewidth=0.5
    )

    ax.set_title(
        "DFT Magnitude of Entire Song"
    )

    ax.set_xlabel(
        "Frequency (Hz)"
    )

    ax.set_ylabel(
        "Magnitude"
    )

    ax.set_xlim(
        0,
        sr / 2
    )

    plt.tight_layout()

    return fig
