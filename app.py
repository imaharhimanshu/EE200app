import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import librosa
import librosa.display
import numpy as np
import pandas as pd
import tempfile, os
from fingerprint import identify, compute_spectrogram, get_peaks, load_audio

# ── Page config ────────────────────────────────────────────────
st.set_page_config(page_title="🎵 Music Identifier", layout="wide")
st.title("🎵 Sonic Signature — Music Identifier")

DB_PATH = "database/song_db.pkl"

# ── Sidebar ────────────────────────────────────────────────────
mode = st.sidebar.radio("Select Mode", ["🎵 Single Clip", "📂 Batch Mode"])

# ══════════════════════════════════════════════════════════════
# SINGLE CLIP MODE
# ══════════════════════════════════════════════════════════════
if mode == "🎵 Single Clip":
    st.header("Single Clip Identification")
    uploaded = st.file_uploader("Upload a query audio clip",
                                type=["mp3", "wav", "flac"])

    if uploaded is not None:
        # Save to temp file
        suffix = os.path.splitext(uploaded.name)[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        st.audio(uploaded)

        with st.spinner("Fingerprinting and matching..."):
            try:
                best_song, S_db, p_times, p_freqs, all_scores, best_offsets = \
                    identify(tmp_path, DB_PATH)

                # ── Result banner ──────────────────────────────
                st.success(f"✅ Matched Song:  **{best_song}**")

                # ── Three plots ────────────────────────────────
                col1, col2, col3 = st.columns(3)

                # Plot 1 — Spectrogram
                with col1:
                    st.subheader("① Spectrogram")
                    y, sr = load_audio(tmp_path)
                    fig1, ax1 = plt.subplots(figsize=(6, 4))
                    librosa.display.specshow(S_db, sr=sr,
                                             hop_length=512,
                                             x_axis='time',
                                             y_axis='hz',
                                             cmap='magma',
                                             ax=ax1)
                    ax1.set_ylim(0, 5000)
                    ax1.set_title("Spectrogram")
                    fig1.colorbar(ax1.collections[0],
                                  ax=ax1, format='%+2.0f dB')
                    st.pyplot(fig1)
                    plt.close(fig1)

                # Plot 2 — Constellation
                with col2:
                    st.subheader("② Constellation Peaks")
                    y, sr = load_audio(tmp_path)
                    fig2, ax2 = plt.subplots(figsize=(6, 4))
                    librosa.display.specshow(S_db, sr=sr,
                                             hop_length=512,
                                             x_axis='time',
                                             y_axis='hz',
                                             cmap='magma',
                                             ax=ax2)
                    ax2.scatter(p_times, p_freqs,
                                facecolors='none',
                                edgecolors='cyan',
                                s=20, linewidths=0.6,
                                label=f'{len(p_times)} peaks')
                    ax2.set_ylim(0, 5000)
                    ax2.set_title("Constellation")
                    ax2.legend(fontsize=8)
                    st.pyplot(fig2)
                    plt.close(fig2)

                # Plot 3 — Offset Histogram
                with col3:
                    st.subheader("③ Offset Histogram")
                    fig3, ax3 = plt.subplots(figsize=(6, 4))
                    if best_offsets:
                        ax3.bar(list(best_offsets.keys()),
                                list(best_offsets.values()),
                                width=0.05, color='steelblue')
                        peak_offset = max(best_offsets,
                                          key=best_offsets.get)
                        ax3.axvline(x=peak_offset, color='red',
                                    linestyle='--',
                                    label=f'Peak @ {peak_offset}s')
                        ax3.legend(fontsize=8)
                    ax3.set_title(f'Offsets — {best_song}')
                    ax3.set_xlabel("Time offset (s)")
                    ax3.set_ylabel("Hash matches")
                    st.pyplot(fig3)
                    plt.close(fig3)

                # ── All songs score table ──────────────────────
                st.subheader("All Song Scores")
                scores_sorted = sorted(all_scores.items(),
                                       key=lambda x: -x[1])
                df_scores = pd.DataFrame(scores_sorted,
                                         columns=["Song", "Max Aligned Matches"])
                st.dataframe(df_scores, use_container_width=True)

            except Exception as e:
                st.error(f"Error: {e}")

        os.unlink(tmp_path)

# ══════════════════════════════════════════════════════════════
# BATCH MODE
# ══════════════════════════════════════════════════════════════
elif mode == "📂 Batch Mode":
    st.header("Batch Identification")
    st.info("Upload multiple clips — a results.csv will be generated.")

    uploaded_files = st.file_uploader(
        "Upload query clips",
        type=["mp3", "wav", "flac"],
        accept_multiple_files=True
    )

    if uploaded_files and st.button("▶ Run Batch"):
        results  = []
        progress = st.progress(0)
        status   = st.empty()

        for i, f in enumerate(uploaded_files):
            status.text(f"Processing {f.name} ...")
            suffix = os.path.splitext(f.name)[-1]
            with tempfile.NamedTemporaryFile(delete=False,
                                             suffix=suffix) as tmp:
                tmp.write(f.read())
                tmp_path = tmp.name
            try:
                best_song, _, _, _, _, _ = identify(tmp_path, DB_PATH)
            except Exception:
                best_song = "error"
            results.append({
                "filename"  : f.name,
                "prediction": best_song
            })
            os.unlink(tmp_path)
            progress.progress((i + 1) / len(uploaded_files))

        status.text("Done!")
        df = pd.DataFrame(results)[["filename", "prediction"]]
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False)
        st.download_button("📥 Download results.csv",
                           csv,
                           file_name="results.csv",
                           mime="text/csv")
