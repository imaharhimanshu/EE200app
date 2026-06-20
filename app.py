import streamlit as st
import librosa
import librosa.display
import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import glob
import os

# Set up the visual layout of the webpage
st.set_page_config(page_title="Audio Fingerprinting", layout="wide")

# ==========================================
# 1. CORE ALGORITHM FUNCTIONS
# ==========================================

def extract_peaks(y, sr):
    """Generates a spectrogram and extracts the constellation peaks."""
    n_fft_standard = 2048
    hop_length_standard = n_fft_standard // 4
    D = librosa.stft(y, n_fft=n_fft_standard, hop_length=hop_length_standard)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    neighborhood_size = (20, 20)
    local_max = ndimage.maximum_filter(S_db, size=neighborhood_size) == S_db
    threshold = np.percentile(S_db, 75)
    loud_enough = S_db > threshold
    peaks = local_max & loud_enough

    peak_freq_indices, peak_time_indices = np.where(peaks)
    peak_times = librosa.frames_to_time(peak_time_indices, sr=sr, hop_length=hop_length_standard)
    peak_freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft_standard)[peak_freq_indices]

    return peak_times, peak_freqs, S_db, hop_length_standard

def add_song_to_database(song_database, peak_times, peak_freqs, song_name):
    """Pairs peaks into hashes and stores them in the dictionary."""
    target_zone_size = 5
    for i in range(len(peak_times)):
        anchor_time = peak_times[i]
        anchor_freq = peak_freqs[i]
        for j in range(1, target_zone_size + 1):
            if i + j < len(peak_times):
                target_time = peak_times[i + j]
                target_freq = peak_freqs[i + j]
                delta_t = round(target_time - anchor_time, 3)
                if delta_t > 0:
                    hash_key = (anchor_freq, target_freq, delta_t)
                    if hash_key not in song_database:
                        song_database[hash_key] = []
                    song_database[hash_key].append((song_name, anchor_time))

def identify_query_clip(query_peak_times, query_peak_freqs, song_database):
    """Matches a query clip against the database and finds the highest time offset alignment."""
    query_hashes = []
    target_zone_size = 5
    
    # Hash the query
    for i in range(len(query_peak_times)):
        anchor_time = query_peak_times[i]
        anchor_freq = query_peak_freqs[i]
        for j in range(1, target_zone_size + 1):
            if i + j < len(query_peak_times):
                delta_t = round(query_peak_times[i + j] - anchor_time, 3)
                if delta_t > 0:
                    hash_key = (anchor_freq, query_peak_freqs[i + j], delta_t)
                    query_hashes.append((hash_key, anchor_time))

    # Compare against database
    offset_tallies = {}
    for hash_key, query_time in query_hashes:
        if hash_key in song_database:
            for db_song_name, db_time in song_database[hash_key]:
                offset = round(db_time - query_time, 2)
                if db_song_name not in offset_tallies:
                    offset_tallies[db_song_name] = {}
                if offset not in offset_tallies[db_song_name]:
                    offset_tallies[db_song_name][offset] = 0
                offset_tallies[db_song_name][offset] += 1

    # Find the winner
    best_song = "Unknown"
    highest_matches = 0
    for song_name, offsets in offset_tallies.items():
        if offsets:
            max_matches = max(offsets.values())
            if max_matches > highest_matches:
                highest_matches = max_matches
                best_song = song_name
                
    return best_song, offset_tallies

# ==========================================
# 2. DATABASE INITIALIZATION
# ==========================================

@st.cache_resource
def build_database():
    """Loads all songs in the database_songs folder and indexes them. Runs ONCE."""
    song_database = {}
    # Make sure this path matches the folder name in your GitHub repo exactly
    folder_path = "./database_songs/*.mp3" 
    files = glob.glob(folder_path)
    
    for file_path in files:
        song_name = os.path.basename(file_path)
        y, sr = librosa.load(file_path, sr=None)
        peak_times, peak_freqs, _, _ = extract_peaks(y, sr)
        add_song_to_database(song_database, peak_times, peak_freqs, song_name)
        
    return song_database

# ==========================================
# 3. STREAMLIT USER INTERFACE
# ==========================================

st.title("EE200: Audio Fingerprinting")
st.markdown("Upload a short clip, and the system will identify the song based on its spectrogram constellation.")

# This spinner will show up while the server builds the DB the very first time
with st.spinner("Indexing library... (This may take a minute)"):
    song_database = build_database()
st.success(f"Library Indexed! Total unique hashes stored: {len(song_database)}")

st.divider()

# File Uploader
uploaded_file = st.file_uploader("Upload a Query Clip", type=["mp3", "wav"])

if uploaded_file is not None:
    st.audio(uploaded_file)
    
    with st.spinner("Extracting features and searching database..."):
        # Load the uploaded file
        y_query, sr_query = librosa.load(uploaded_file, sr=None)
        
        # 1. Extract peaks
        query_peak_times, query_peak_freqs, S_db_query, hop_length = extract_peaks(y_query, sr_query)
        
        # 2. Plot Spectrogram with Constellation
        st.subheader("Step 1: Feature Extraction (Constellation)")
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        librosa.display.specshow(S_db_query, sr=sr_query, hop_length=hop_length, x_axis='time', y_axis='hz', cmap='magma', ax=ax1)
        ax1.scatter(query_peak_times, query_peak_freqs, facecolors='none', edgecolors='cyan', s=30, label='Query Peaks')
        ax1.set_ylim(0, 3500)
        ax1.legend()
        st.pyplot(fig1)

        # 3. Run Matching Engine
        best_song, offset_tallies = identify_query_clip(query_peak_times, query_peak_freqs, song_database)

        st.subheader("Step 2: Match Results")
        if best_song != "Unknown":
            st.success(f"🎉 **Match Found!** The system identified the song as: **{best_song}**")
            
            # 4. Plot the Offset Histogram for the winning song
            st.subheader("Step 3: The Proof (Alignment Spike)")
            winning_offsets = offset_tallies[best_song]
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            ax2.bar(list(winning_offsets.keys()), list(winning_offsets.values()), width=0.2, color='orange')
            ax2.set_xlabel("Time Offset (seconds)")
            ax2.set_ylabel("Number of Aligned Hashes")
            ax2.set_title(f"Histogram of Time Offsets for {best_song}")
            st.pyplot(fig2)
            
        else:
            st.error("No match found in the database. The clip might be too noisy or pitched shifted.")
