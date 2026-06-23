import os
import streamlit as st
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import pandas as pd
import tempfile

from fingerprint import (
    load_database, build_database,
    identify_clip, compute_spectrogram,
    find_peaks, load_audio
)

st.set_page_config(
    page_title="EE200: MUSIC IDENTIFIER",
    layout="wide"
)

DB_PATH = "database.pkl"

@st.cache_resource
def get_database():
    return load_database(DB_PATH)

database = get_database()

st.sidebar.title("EE200: MUSIC IDENTIFIER")
st.sidebar.markdown("---")

mode = st.sidebar.radio("Choose Mode", ["Single Clip", "Batch Mode"])

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Songs in database:** {len(database['songs'])}")
st.sidebar.markdown("**Indexed songs:**")
for s in sorted(database["songs"]):
    st.sidebar.markdown(f"- {s}")

def draw_spectrogram(S_db, sr, hop_length, title="Spectrogram"):
    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.specshow(
        S_db, sr=sr, hop_length=hop_length,
        x_axis='time', y_axis='hz',
        cmap='magma', ax=ax
    )
    plt.colorbar(ax.collections[0], ax=ax, format='%+2.0f dB')
    ax.set_title(title)
    ax.set_ylim(0, 5000)
    plt.tight_layout()
    return fig

def draw_constellation(S_db, sr, hop_length, peak_times, peak_freqs):
    fig = plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        S_db, sr=sr, hop_length=hop_length,
        x_axis='time', y_axis='hz', cmap='magma'
    )
    plt.plot(
        peak_times, peak_freqs,
        linestyle='None',
        marker='o',
        markerfacecolor='none',
        markeredgecolor='cyan',
        markersize=5,
        label='Constellation Peaks'
    )
    plt.ylim(0, 3500)
    plt.xlim(0, 25)
    plt.title('Spectrogram with Constellation Peaks')
    plt.colorbar(format='%+2.0f dB')
    plt.legend(loc='upper right')
    plt.tight_layout()
    return fig

def draw_histogram(best_per_song, matched_song):
    fig, ax = plt.subplots(figsize=(10, 4))
    songs = list(best_per_song.keys())
    scores = [best_per_song[s] for s in songs]
    colors = ['#00cc66' if s == matched_song else '#4466ff' for s in songs]
    ax.bar(songs, scores, color=colors)
    ax.set_title("Offset Histogram - Match Scores per Song")
    ax.set_ylabel("Peak Offset Count")
    ax.set_xlabel("Song")
    plt.xticks(rotation=90, ha='right', fontsize=7)
    plt.tight_layout()
    return fig


if mode == "Single Clip":
    st.title("Single Clip Identifier")
    st.markdown("Upload a short audio clip and the app will identify which song it belongs to.")

    uploaded = st.file_uploader("Upload a query clip (.mp3 or .wav)", type=["mp3", "wav"])

    if uploaded is not None:
        suffix = ".mp3" if uploaded.name.endswith(".mp3") else ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        st.audio(uploaded)

        with st.spinner("Analysing..."):
            (matched_song, best_per_song, S_db, times, freqs,
             hop_length, sr, peak_times, peak_freqs,
             total_query_hashes, matched_hash_counts) = identify_clip(tmp_path, database)

        os.unlink(tmp_path)

        # Result
        st.success(f"Matched Song: **{matched_song}**")

        # Hash match stats
        matched = matched_hash_counts[matched_song]
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Query Hashes", total_query_hashes)
        col2.metric("Hashes Matched", matched)
        col3.metric("Match Rate", f"{100 * matched / max(total_query_hashes, 1):.1f}%")

        st.markdown("---")


        st.subheader("Step 1 - Spectrogram")
        st.markdown("Time on X-axis, frequency on Y-axis, brightness shows strength of that frequency at that moment.")
        fig1 = draw_spectrogram(S_db, sr, hop_length)
        st.pyplot(fig1)
        plt.close(fig1)


        st.subheader("Step 2 - Constellation of Peaks")
        st.markdown("Only the loudest local-maximum points are kept as the song fingerprint.")
        fig2 = draw_constellation(S_db, sr, hop_length, peak_times, peak_freqs)
        st.pyplot(fig2)
        plt.close(fig2)


        st.subheader("Step 3 - Offset Histogram")
        st.markdown("The green bar is the winning song. Its hashes all align at one time offset, while wrong songs give scattered matches.")
        fig3 = draw_histogram(best_per_song, matched_song)
        st.pyplot(fig3)
        plt.close(fig3)


elif mode == "Batch Mode":
    st.title("Batch Identifier")
    st.markdown("Upload multiple clips. The app will return a `results.csv` with `filename` and `prediction` columns.")

    uploaded_files = st.file_uploader(
        "Upload query clips (.mp3 or .wav)",
        type=["mp3", "wav"],
        accept_multiple_files=True
    )

    if uploaded_files:
        results = []
        progress = st.progress(0)
        status = st.empty()

        for i, f in enumerate(uploaded_files):
            status.text(f"Processing {f.name} ...")
            suffix = ".mp3" if f.name.endswith(".mp3") else ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(f.read())
                tmp_path = tmp.name

            try:
                (matched_song, _, _, _, _, _, _, _, _, _, _) = identify_clip(tmp_path, database)
            except Exception as e:
                matched_song = f"ERROR: {e}"

            os.unlink(tmp_path)

            results.append({
                "filename": os.path.splitext(f.name)[0],
                "prediction": matched_song
            })
            progress.progress((i + 1) / len(uploaded_files))

        status.text("Done!")

        df = pd.DataFrame(results)[["filename", "prediction"]]
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False)
        st.download_button(
            label="Download results.csv",
            data=csv,
            file_name="results.csv",
            mime="text/csv"
        )
