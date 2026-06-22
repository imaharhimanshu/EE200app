import streamlit as st
import pandas as pd
import tempfile
from pathlib import Path

from utils.fingerprint import fingerprint_audio
from utils.matcher import (
    build_song_database,
    identify_song
)
from utils.visualizations import (
    plot_spectrogram,
    plot_constellation,
    plot_offset_histogram
)

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Sonic Signatures",
    layout="wide"
)

st.title("🎵 Sonic Signatures")
st.write("Audio Fingerprinting Music Identifier")


# ==========================================
# BUILD DATABASE
# ==========================================

@st.cache_resource
def load_database():

    songs_folder = Path("songs")

    song_fingerprints = {}

    supported_formats = [
        "*.mp3",
        "*.wav",
        "*.flac"
    ]

    song_files = []

    for ext in supported_formats:
        song_files.extend(
            songs_folder.glob(ext)
        )

    for song_path in song_files:

        song_name = song_path.stem

        fp = fingerprint_audio(
            str(song_path)
        )

        song_fingerprints[song_name] = fp["hashes"]

    database = build_song_database(
        song_fingerprints
    )

    return database, len(song_files)


database, total_songs = load_database()

st.sidebar.success(
    f"Indexed Songs: {total_songs}"
)


# ==========================================
# MODE SELECTION
# ==========================================

mode = st.sidebar.radio(
    "Select Mode",
    [
        "Single Clip Recognition",
        "Batch Recognition"
    ]
)


# ==========================================
# SINGLE CLIP MODE
# ==========================================

if mode == "Single Clip Recognition":

    st.header("Single Clip Recognition")

    uploaded_file = st.file_uploader(
        "Upload Audio File",
        type=["mp3", "wav", "flac"]
    )

    if uploaded_file:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        ) as tmp:

            tmp.write(uploaded_file.read())

            temp_path = tmp.name

        query_fp = fingerprint_audio(
            temp_path
        )

        song_name, confidence, histogram = identify_song(
            query_fp["hashes"],
            database
        )

        st.success(
            f"Matched Song: {song_name}"
        )

        st.info(
            f"Confidence: {confidence:.2f}%"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Spectrogram")

            fig = plot_spectrogram(
                query_fp["spectrogram"],
                query_fp["sample_rate"]
            )

            st.pyplot(fig)

        with col2:

            st.subheader("Constellation Map")

            fig = plot_constellation(
                query_fp["spectrogram"],
                query_fp["sample_rate"],
                query_fp["peak_times"],
                query_fp["peak_freqs"]
            )

            st.pyplot(fig)

        st.subheader("Offset Histogram")

        fig = plot_offset_histogram(
            histogram
        )

        st.pyplot(fig)


# ==========================================
# BATCH MODE
# ==========================================

else:

    st.header("Batch Recognition")

    uploaded_files = st.file_uploader(
        "Upload Query Clips",
        type=["mp3", "wav", "flac"],
        accept_multiple_files=True
    )

    if uploaded_files:

        results = []

        progress = st.progress(0)

        for idx, file in enumerate(uploaded_files):

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".wav"
            ) as tmp:

                tmp.write(file.read())

                temp_path = tmp.name

            query_fp = fingerprint_audio(
                temp_path
            )

            song_name, _, _ = identify_song(
                query_fp["hashes"],
                database
            )

            results.append(
                {
                    "filename": file.name,
                    "prediction": song_name
                }
            )

            progress.progress(
                (idx + 1) / len(uploaded_files)
            )

        results_df = pd.DataFrame(
            results
        )

        st.subheader("Results")

        st.dataframe(
            results_df,
            use_container_width=True
        )

        csv = results_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Download results.csv",
            data=csv,
            file_name="results.csv",
            mime="text/csv"
        )
