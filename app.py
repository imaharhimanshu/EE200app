import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tempfile, os
from fingerprint import identify, compute_spectrogram, get_peaks, load_audio

st.set_page_config(page_title="🎵 Music Identifier", layout="wide")
st.title("🎵 Sonic Signature — Music Identifier")

mode = st.sidebar.radio("Mode", ["Single Clip", "Batch Mode"])

# ── SINGLE CLIP MODE ──────────────────────────────────────────
if mode == "Single Clip":
    uploaded = st.file_uploader("Upload a query audio clip", type=["mp3","wav","flac"])

    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        st.audio(uploaded)

        with st.spinner("Identifying..."):
            best_song, S_db, peaks, all_scores = identify(tmp_path)

        st.success(f"✅ Matched Song: **{best_song}**")

        col1, col2, col3 = st.columns(3)

        # Plot 1: Spectrogram
        with col1:
            st.subheader("Spectrogram")
            fig, ax = plt.subplots()
            ax.imshow(S_db, origin='lower', aspect='auto', cmap='inferno')
            ax.set_xlabel("Time frames")
            ax.set_ylabel("Frequency bins")
            st.pyplot(fig)

        # Plot 2: Constellation
        with col2:
            st.subheader("Constellation (Peaks)")
            fig2, ax2 = plt.subplots()
            ax2.imshow(S_db, origin='lower', aspect='auto', cmap='gray')
            times = [p[0] for p in peaks]
            freqs = [p[1] for p in peaks]
            ax2.scatter(times, freqs, c='cyan', s=5, alpha=0.7)
            ax2.set_xlabel("Time frames")
            ax2.set_ylabel("Frequency bins")
            st.pyplot(fig2)

        # Plot 3: Offset histogram
        with col3:
            st.subheader("Match Scores (Offset Histogram)")
            fig3, ax3 = plt.subplots()
            songs = list(all_scores.keys())
            scores = list(all_scores.values())
            ax3.barh(songs, scores, color='steelblue')
            ax3.set_xlabel("Max aligned matches")
            ax3.axvline(x=max(scores), color='red', linestyle='--')
            st.pyplot(fig3)

        os.unlink(tmp_path)

# ── BATCH MODE ────────────────────────────────────────────────
elif mode == "Batch Mode":
    uploaded_files = st.file_uploader(
        "Upload multiple query clips",
        type=["mp3","wav","flac"],
        accept_multiple_files=True
    )

    if uploaded_files and st.button("Run Batch Identification"):
        results = []
        progress = st.progress(0)

        for i, f in enumerate(uploaded_files):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                tmp.write(f.read())
                tmp_path = tmp.name
            best_song, _, _, _ = identify(tmp_path)
            results.append({"filename": f.name, "prediction": best_song})
            os.unlink(tmp_path)
            progress.progress((i+1)/len(uploaded_files))

        df = pd.DataFrame(results)[["filename", "prediction"]]
        st.dataframe(df)
        csv = df.to_csv(index=False)
        st.download_button("📥 Download results.csv", csv, "results.csv", "text/csv")
