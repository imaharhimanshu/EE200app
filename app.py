import streamlit as st
import librosa
import librosa.display
import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
import time
import tempfile

# ==========================================
# PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(page_title="EE200: Audio Fingerprinting", layout="wide")
st.title("EE200: Audio Fingerprinting")
st.caption("Signals, Systems & Networks - Project Demo")

# Dark theme for matplotlib to match the constellation aesthetic
plt.style.use('dark_background')

# ==========================================
# CORE ALGORITHM FUNCTIONS
# ==========================================
def extract_peaks(y, sr):
    """Generates a spectrogram and extracts constellation peaks."""
    n_fft = 2048
    hop_length = n_fft // 4
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    neighborhood_size = (20, 20)
    local_max = ndimage.maximum_filter(S_db, size=neighborhood_size) == S_db
    threshold = np.percentile(S_db, 75)
    loud_enough = S_db > threshold
    peaks = local_max & loud_enough

    peak_freq_indices, peak_time_indices = np.where(peaks)
    peak_times = librosa.frames_to_time(peak_time_indices, sr=sr, hop_length=hop_length)
    peak_freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)[peak_freq_indices]

    return peak_times, peak_freqs, S_db, hop_length

def build_hashes(peak_times, peak_freqs):
    """Pairs peaks into hashes."""
    hashes = []
    target_zone_size = 5
    for i in range(len(peak_times)):
        anchor_time = peak_times[i]
        anchor_freq = peak_freqs[i]
        for j in range(1, target_zone_size + 1):
            if i + j < len(peak_times):
                delta_t = round(peak_times[i + j] - anchor_time, 3)
                if delta_t > 0:
                    hash_key = (anchor_freq, peak_freqs[i + j], delta_t)
                    hashes.append((hash_key, anchor_time))
    return hashes

@st.cache_resource
def build_database():
    """Indexes library. Returns the hash DB and metadata for the Library tab."""
    song_database = {}
    db_metadata = {} # Stores hashes count and constellation points for visual index
    
    folder_path = "./database_songs/*.mp3" # Ensure this folder exists!
    files = glob.glob(folder_path)
    
    for file_path in files:
        song_name = os.path.basename(file_path)
        try:
            y, sr = librosa.load(file_path, sr=None)
            peak_times, peak_freqs, _, _ = extract_peaks(y, sr)
            hashes = build_hashes(peak_times, peak_freqs)
            
            db_metadata[song_name] = {
                "hash_count": len(hashes),
                "peak_times": peak_times,
                "peak_freqs": peak_freqs
            }
            
            for hash_key, anchor_time in hashes:
                if hash_key not in song_database:
                    song_database[hash_key] = []
                song_database[hash_key].append((song_name, anchor_time))
        except Exception as e:
            print(f"Skipping {song_name} due to read error: {e}")
            
    return song_database, db_metadata

# Boot up the database
with st.spinner("Indexing database..."):
    song_database, db_metadata = build_database()

# ==========================================
# APP LAYOUT (TABS)
# ==========================================
tab_library, tab_identify, tab_batch = st.tabs(["LIBRARY", "IDENTIFY", "BATCH"])

# ------------------------------------------
# TAB 1: LIBRARY DASHBOARD
# ------------------------------------------
with tab_library:
    st.markdown("### Indexed Database")
    st.write("Song indexing is managed by the admin. Drop a clip in the Identify tab to test the library.")
    
    if not db_metadata:
        st.warning("No songs found in the database. Ensure 'database_songs' folder has mp3 files.")
    else:
        # Create a grid layout (4 columns)
        cols = st.columns(4)
        col_idx = 0
        for song_name, data in db_metadata.items():
            with cols[col_idx % 4]:
                st.markdown(f"**{song_name}**")
                st.caption(f"{data['hash_count']:,} hashes")
                
                # Mini constellation plot
                fig, ax = plt.subplots(figsize=(3, 2))
                ax.scatter(data['peak_times'], data['peak_freqs'], s=1, color='cyan', alpha=0.6)
                ax.axis('off') # Hide axes for clean thumbnail look
                st.pyplot(fig, clear_figure=True)
                
            col_idx += 1

# ------------------------------------------
# TAB 2: IDENTIFY (SINGLE CLIP)
# ------------------------------------------
with tab_identify:
    st.markdown("### Identify a clip")
    query_file = st.file_uploader("Upload a short query clip", type=["mp3", "wav", "flac", "m4a"], key="single_upload")
    
    if query_file is not None:
        st.audio(query_file)
        
        # Track execution times for live metrics
        time_metrics = {}
        start_total = time.time()
        
        # We save the file temporarily to prevent librosa buffer crash
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(query_file.getvalue())
            tmp_path = tmp_file.name

        with st.spinner("Processing..."):
            # Step 1: Spectrogram & Peaks
            t0 = time.time()
            y_q, sr_q = librosa.load(tmp_path, sr=None)
            peak_times_q, peak_freqs_q, S_db_q, hop_length_q = extract_peaks(y_q, sr_q)
            time_metrics['Spectrogram/Peaks'] = int((time.time() - t0) * 1000)
            
            # Step 2: Hashing
            t0 = time.time()
            query_hashes = build_hashes(peak_times_q, peak_freqs_q)
            time_metrics['Hashing'] = int((time.time() - t0) * 1000)
            
            # Step 3: DB Lookup & Scoring
            t0 = time.time()
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
            
            candidate_scores = {}
            for song, offsets in offset_tallies.items():
                candidate_scores[song] = max(offsets.values()) if offsets else 0
            
            time_metrics['DB Lookup & Scoring'] = int((time.time() - t0) * 1000)
            time_metrics['Total'] = int((time.time() - start_total) * 1000)
        
        os.remove(tmp_path) # Clean up temp file
        
        # --- DISPLAY RESULTS ---
        st.divider()
        
        # Live Performance Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Spectrogram & Peaks", f"{time_metrics['Spectrogram/Peaks']} ms")
        m2.metric("Hashing", f"{time_metrics['Hashing']} ms")
        m3.metric("DB Lookup & Score", f"{time_metrics['DB Lookup & Scoring']} ms")
        m4.metric("TOTAL TIME", f"{time_metrics['Total']} ms")
        
        if not candidate_scores:
            st.error("No matches found. Try a cleaner clip.")
        else:
            # Sort candidates
            sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
            winner = sorted_candidates[0]
            
            st.success(f"### Match Found: {winner[0]}")
            
            # Candidate Scoreboard
            st.markdown("**CANDIDATE SCORES**")
            score_df = pd.DataFrame(sorted_candidates, columns=["Track", "Score (Aligned Hashes)"])
            st.dataframe(score_df, hide_index=True, use_container_width=True)
            
            # Step 1: Feature Extraction Plot
            st.markdown("#### Step 1 - Feature Extraction")
            st.write(f"From spectrogram to constellation. Kept **{len(peak_times_q)} prominent peaks**.")
            fig1, ax1 = plt.subplots(figsize=(10, 3))
            librosa.display.specshow(S_db_q, sr=sr_q, hop_length=hop_length_q, x_axis='time', y_axis='hz', cmap='magma', ax=ax1)
            ax1.scatter(peak_times_q, peak_freqs_q, s=20, edgecolor='cyan', facecolor='none')
            ax1.set_ylim(0, 4000)
            st.pyplot(fig1, clear_figure=True)
            
            # Step 3: The Proof Plot
            st.markdown("#### Step 2 - The Proof (Alignment Spike)")
            st.write(f"Every matched hash votes for a time offset. **{winner[1]} hashes** agreed on a single offset.")
            winning_offsets = offset_tallies[winner[0]]
            fig3, ax3 = plt.subplots(figsize=(10, 3))
            ax3.bar(list(winning_offsets.keys()), list(winning_offsets.values()), width=0.1, color='orange')
            ax3.set_xlabel("Time offset (Database frame - query frame)")
            ax3.set_ylabel("# Hashes")
            st.pyplot(fig3, clear_figure=True)

# ------------------------------------------
# TAB 3: BATCH PROCESSING
# ------------------------------------------
with tab_batch:
    st.markdown("### Identify many clips at once")
    st.write("Upload a set of query clips. Each is identified against the database, and results are written to a CSV.")
    
    batch_files = st.file_uploader("Upload Batch", type=["mp3", "wav"], accept_multiple_files=True, key="batch_upload")
    
    if batch_files and st.button("Run Batch"):
        results = []
        progress_bar = st.progress(0)
        
        for idx, file in enumerate(batch_files):
            # Save temporary file safely
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(file.getvalue())
                tmp_path = tmp_file.name
                
            try:
                y_q, sr_q = librosa.load(tmp_path, sr=None)
                p_times, p_freqs, _, _ = extract_peaks(y_q, sr_q)
                q_hashes = build_hashes(p_times, p_freqs)
                
                # Fast lookup
                batch_tallies = {}
                for h_key, q_time in q_hashes:
                    if h_key in song_database:
                        for db_song, db_time in song_database[h_key]:
                            offset = round(db_time - q_time, 1)
                            if db_song not in batch_tallies:
                                batch_tallies[db_song] = {}
                            batch_tallies[db_song][offset] = batch_tallies[db_song].get(offset, 0) + 1
                
                best_song = "None"
                best_score = 0
                for song, offsets in batch_tallies.items():
                    if offsets and max(offsets.values()) > best_score:
                        best_score = max(offsets.values())
                        best_song = song
                        
                results.append({"Filename": file.name, "Prediction": best_song, "Score": best_score})
            except Exception as e:
                results.append({"Filename": file.name, "Prediction": "Error processing", "Score": 0})
            
            os.remove(tmp_path)
            progress_bar.progress((idx + 1) / len(batch_files))
            
        st.success(f"{len(batch_files)} clips matched.")
        
        # Display DataFrame and CSV download
        results_df = pd.DataFrame(results)
        st.dataframe(results_df, use_container_width=True)
        
        csv_data = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download results.csv",
            data=csv_data,
            file_name='results.csv',
            mime='text/csv',
        )
