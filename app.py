import streamlit as st
import librosa
import librosa.display
import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import glob
import os

# --- PAGE SETUP ---
st.set_page_config(page_title="Sonic Signatures", layout="wide")
st.title("🎵 Sonic Signatures Identifier")
st.write("Upload a query clip to identify it from our database of 50 songs!")

# ==========================================
# 1. CORE FUNCTIONS
# ==========================================

def get_peaks(y, sr):
    """Extracts spectrogram and constellation peaks from audio."""
    n_fft_standard = 2048
    hop_length_standard = n_fft_standard // 4
    
    D_standard = librosa.stft(y, n_fft=n_fft_standard, hop_length=hop_length_standard)
    S_db_standard = librosa.amplitude_to_db(np.abs(D_standard), ref=np.max)
    
    neighborhood_size = (20, 20)
    local_max = ndimage.maximum_filter(S_db_standard, size=neighborhood_size) == S_db_standard
    
    threshold = np.percentile(S_db_standard, 75)
    loud_enough = S_db_standard > threshold
    peaks = local_max & loud_enough
    
    peak_freq_indices, peak_time_indices = np.where(peaks)
    peak_times = librosa.frames_to_time(peak_time_indices, sr=sr, hop_length=hop_length_standard)
    peak_freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft_standard)[peak_freq_indices]
    
    return peak_times, peak_freqs, S_db_standard

def add_song_to_database(peak_times, peak_freqs, song_name, db):
    """Creates hashes and adds them to the database dictionary."""
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
                    if hash_key not in db:
                        db[hash_key] = []
                    db[hash_key].append((song_name, anchor_time))

def identify_query_clip(query_peak_times, query_peak_freqs, song_database):
    """Matches query hashes against the database and finds the best song."""
    query_hashes = []
    target_zone_size = 5

    # 1. Generate hashes for the query
    for i in range(len(query_peak_times)):
        anchor_time = query_peak_times[i]
        anchor_freq = query_peak_freqs[i]
        for j in range(1, target_zone_size + 1):
            if i + j < len(query_peak_times):
                delta_t = round(query_peak_times[i + j] - anchor_time, 3)
                if delta_t > 0:
                    hash_key = (anchor_freq, query_peak_freqs[i + j], delta_t)
                    query_hashes.append((hash_key, anchor_time))

    # 2. Compare against database
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

    # 3. Find the winner
    best_song = "Unknown"
    highest_matches = 0
    best_offsets = []

    for song_name, offsets in offset_tallies.items():
        if offsets:
            max_matches = max(offsets.values())
            if max_matches > highest_matches:
                highest_matches = max_matches
                best_song = song_name
                # Expand the dictionary into a flat list for the histogram
                best_offsets = [off for off, count in offsets.items() for _ in range(count)]

    return best_song, highest_matches, best_offsets

# ==========================================
# 2. BUILD DATABASE (Runs once on startup)
# ==========================================
@st.cache_resource(show_spinner=False)
def load_database():
    db = {}
    # Looking for a folder named "database_songs" in the GitHub repo
    folder_path = "database_songs/*.mp3" 
    files = glob.glob(folder_path)
    
    if len(files) == 0:
        return None # No songs found
        
    for file_path in files:
        song_name = os.path.basename(file_path)
        y, sr = librosa.load(file_path, sr=None)
        p_times, p_freqs, _ = get_peaks(y, sr)
        add_song_to_database(p_times, p_freqs, song_name, db)
    return db

with st.spinner("Building database from 50 songs... (This may take a minute)"):
    song_database = load_database()

# ==========================================
# 3. USER INTERFACE (The App)
# ==========================================

if song_database is None:
    st.error("⚠️ Could not find the 'database_songs' folder. Make sure you uploaded your 50 MP3s to GitHub!")
else:
    st.success(f"✅ Database active! Indexed {len(song_database)} unique frequency hashes.")
    
    # Drag and drop file uploader
    uploaded_file = st.file_uploader("Upload your mystery audio clip", type=['mp3', 'wav'])

    if uploaded_file is not None:
        st.write("---")
        st.subheader("Processing Query Clip...")
        
        # Load audio
        y_query, sr_query = librosa.load(uploaded_file, sr=None)
        
        # Get peaks and spectrogram
        query_times, query_freqs, S_db_query = get_peaks(y_query, sr_query)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Step 1: Extracted Constellation**")
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            librosa.display.specshow(S_db_query, sr=sr_query, hop_length=(2048//4), x_axis='time', y_axis='hz', cmap='magma', ax=ax1)
            ax1.scatter(query_times, query_freqs, facecolors='none', edgecolors='cyan', s=30, label='Peaks')
            ax1.set_ylim(0, 3500)
            ax1.legend(loc='upper right')
            st.pyplot(fig1)

        with col2:
            st.write("**Step 2: Database Match Result**")
            # Run identification
            best_match, max_matches, offset_list = identify_query_clip(query_times, query_freqs, song_database)
            
            st.success(f"### 🏆 Matched Song: {best_match}")
            st.write(f"Highest aligned hashes: **{max_matches}**")
            
            # Plot Histogram
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            ax2.hist(offset_list, bins=50, color='lightgreen', edgecolor='black')
            ax2.set_title("Time Offset Distribution")
            ax2.set_xlabel("Time Offset (seconds)")
            ax2.set_ylabel("Matches")
            st.pyplot(fig2)
            
            st.balloons()
