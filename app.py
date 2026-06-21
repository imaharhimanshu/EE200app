import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import pandas as pd
import tempfile, os, pickle

from fingerprint import identify, compute_spectrogram, get_peaks, load_audio, N_FFT, HOP_LENGTH

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sonic Signature",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark background */
    .stApp { background-color: #0e1117; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #161b22;
        border-right: 1px solid #30363d;
    }

    /* Hero banner */
    .hero {
        background: linear-gradient(135deg, #1a1f2e 0%, #16213e 50%, #0f3460 100%);
        border: 1px solid #30363d;
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
    }
    .hero h1 { font-size: 2.2rem; margin: 0; color: #e6edf3; }
    .hero p  { color: #8b949e; margin: 0.4rem 0 0; font-size: 1rem; }

    /* Result card */
    .result-card {
        background: linear-gradient(135deg, #0d2137, #0a3d1f);
        border: 1px solid #238636;
        border-radius: 12px;
        padding: 1.2rem 1.8rem;
        margin: 1rem 0;
    }
    .result-card h2 { color: #3fb950; margin: 0; font-size: 1.6rem; }
    .result-card p  { color: #8b949e; margin: 0.2rem 0 0; font-size: 0.9rem; }

    /* Song library card */
    .song-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.8rem;
    }

    /* Step labels */
    .step-label {
        background: #1f6feb;
        color: white;
        border-radius: 6px;
        padding: 0.2rem 0.6rem;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 0.4rem;
    }

    /* Metric box */
    .metric-box {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .metric-box .val { font-size: 1.8rem; font-weight: 700; color: #58a6ff; }
    .metric-box .lbl { font-size: 0.8rem; color: #8b949e; }

    /* Section divider */
    .divider {
        border: none;
        border-top: 1px solid #30363d;
        margin: 1.5rem 0;
    }

    /* Hide default streamlit header */
    header[data-testid="stHeader"] { display: none; }

    /* Plot background */
    .stPlot { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
DB_PATH    = "database/song_db.pkl"
SONGS_DIR  = "songs"

# ── Helper: load song list from database ──────────────────────────────────────
@st.cache_data
def get_indexed_songs(db_path):
    if not os.path.exists(db_path):
        return []
    try:
        # Try normal load first
        with open(db_path, 'rb') as f:
            db = pickle.load(f)
    except Exception:
        try:
            # Fallback: try with encoding fix for cross-version pickle
            with open(db_path, 'rb') as f:
                db = pickle.load(f, encoding='latin1')
        except Exception as e:
            st.error(f"Could not load database: {e}")
            return []
    songs = set()
    for entries in db.values():
        for song_name, _ in entries:
            songs.add(song_name)
    return sorted(songs)
# ── Helper: plot spectrogram ──────────────────────────────────────────────────
def plot_spectrogram(S_db, sr, title="Spectrogram"):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    img = librosa.display.specshow(
        S_db, sr=sr, hop_length=HOP_LENGTH,
        x_axis='time', y_axis='hz',
        cmap='magma', ax=ax
    )
    ax.set_ylim(0, 5000)
    ax.set_title(title, color='#e6edf3', fontsize=11, pad=8)
    ax.set_xlabel("Time (s)", color='#8b949e', fontsize=9)
    ax.set_ylabel("Frequency (Hz)", color='#8b949e', fontsize=9)
    ax.tick_params(colors='#8b949e', labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')
    cbar = fig.colorbar(img, ax=ax, format='%+2.0f dB')
    cbar.ax.yaxis.set_tick_params(color='#8b949e', labelsize=7)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#8b949e')
    cbar.outline.set_edgecolor('#30363d')
    plt.tight_layout(pad=0.5)
    return fig

# ── Helper: plot constellation ────────────────────────────────────────────────
def plot_constellation(S_db, sr, p_times, p_freqs, title="Constellation Peaks"):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    librosa.display.specshow(
        S_db, sr=sr, hop_length=HOP_LENGTH,
        x_axis='time', y_axis='hz',
        cmap='magma', ax=ax
    )
    ax.scatter(p_times, p_freqs,
               facecolors='none', edgecolors='#00d4ff',
               s=25, linewidths=0.7,
               label=f'{len(p_times)} peaks', zorder=5)
    ax.set_ylim(0, 5000)
    ax.set_title(title, color='#e6edf3', fontsize=11, pad=8)
    ax.set_xlabel("Time (s)", color='#8b949e', fontsize=9)
    ax.set_ylabel("Frequency (Hz)", color='#8b949e', fontsize=9)
    ax.tick_params(colors='#8b949e', labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')
    leg = ax.legend(fontsize=8, loc='upper right')
    leg.get_frame().set_facecolor('#161b22')
    leg.get_frame().set_edgecolor('#30363d')
    for t in leg.get_texts():
        t.set_color('#e6edf3')
    plt.tight_layout(pad=0.5)
    return fig

# ── Helper: plot offset histogram ─────────────────────────────────────────────
def plot_offset_histogram(best_offsets, best_song):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    if best_offsets:
        offsets = list(best_offsets.keys())
        counts  = list(best_offsets.values())
        ax.bar(offsets, counts, width=0.05,
               color='#1f6feb', alpha=0.85, zorder=3)
        peak_off = max(best_offsets, key=best_offsets.get)
        ax.axvline(x=peak_off, color='#3fb950',
                   linestyle='--', linewidth=1.5,
                   label=f'Peak @ {peak_off:.2f}s', zorder=4)
        leg = ax.legend(fontsize=8)
        leg.get_frame().set_facecolor('#161b22')
        leg.get_frame().set_edgecolor('#30363d')
        for t in leg.get_texts():
            t.set_color('#e6edf3')
    ax.set_title(f'Offset Histogram — {best_song}', color='#e6edf3', fontsize=11, pad=8)
    ax.set_xlabel("Time Offset (s)", color='#8b949e', fontsize=9)
    ax.set_ylabel("Hash Matches", color='#8b949e', fontsize=9)
    ax.tick_params(colors='#8b949e', labelsize=8)
    ax.grid(axis='y', color='#30363d', linewidth=0.5, zorder=0)
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')
    plt.tight_layout(pad=0.5)
    return fig

# ── Helper: all songs bar chart ───────────────────────────────────────────────
def plot_scores_bar(all_scores, best_song):
    scores_sorted = sorted(all_scores.items(), key=lambda x: -x[1])[:10]
    names  = [s[0] for s in scores_sorted]
    scores = [s[1] for s in scores_sorted]
    colors = ['#3fb950' if n == best_song else '#1f6feb' for n in names]
    fig, ax = plt.subplots(figsize=(7, max(3, len(names) * 0.45)))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    bars = ax.barh(names, scores, color=colors, alpha=0.85)
    ax.set_xlabel("Max Aligned Hash Matches", color='#8b949e', fontsize=9)
    ax.set_title("Match Scores — All Songs", color='#e6edf3', fontsize=11, pad=8)
    ax.tick_params(colors='#8b949e', labelsize=8)
    ax.grid(axis='x', color='#30363d', linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')
    # Value labels
    for bar, score in zip(bars, scores):
        ax.text(score + max(scores)*0.01, bar.get_y() + bar.get_height()/2,
                str(score), va='center', color='#8b949e', fontsize=8)
    plt.tight_layout(pad=0.5)
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎵 Sonic Signature")
    st.markdown("<hr style='border-color:#30363d'>", unsafe_allow_html=True)

    mode = st.radio("Select Mode", ["🎵 Single Clip", "📂 Batch Mode", "📚 Song Library"])

    st.markdown("<hr style='border-color:#30363d'>", unsafe_allow_html=True)

    # Database status
    indexed_songs = get_indexed_songs(DB_PATH)
    if indexed_songs:
        st.success(f"✅ Database ready\n\n**{len(indexed_songs)} songs** indexed")
    else:
        st.error("❌ Database not found\n\nCheck `database/song_db.pkl`")

    st.markdown("<hr style='border-color:#30363d'>", unsafe_allow_html=True)
    st.caption("Built for EE200 · Sonic Signatures")

# ══════════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <h1>🎵 Sonic Signature — Music Identifier</h1>
    <p>Fingerprint-based music recognition · Spectrogram · Constellation · Hash Matching</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MODE: SINGLE CLIP
# ══════════════════════════════════════════════════════════════════════════════
if mode == "🎵 Single Clip":
    st.markdown("### Single Clip Identification")
    st.markdown("Upload any short audio clip and the system will identify it from the database.")

    uploaded = st.file_uploader(
        "Drop your query clip here",
        type=["mp3", "wav", "flac"],
        help="Upload a clip (even a few seconds works)"
    )

    if uploaded is not None:
        st.audio(uploaded)

        suffix = os.path.splitext(uploaded.name)[-1] or ".mp3"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        with st.spinner("🔍 Fingerprinting and matching..."):
            try:
                best_song, S_db, p_times, p_freqs, all_scores, best_offsets = \
                    identify(tmp_path, DB_PATH)

                y, sr = load_audio(tmp_path)

                # ── Result banner ──────────────────────────────────────────
                st.markdown(f"""
                <div class="result-card">
                    <h2>✅ {best_song}</h2>
                    <p>Matched with {max(best_offsets.values()) if best_offsets else 0} aligned hash pairs</p>
                </div>
                """, unsafe_allow_html=True)

                # ── Quick metrics ──────────────────────────────────────────
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.markdown(f"""<div class="metric-box">
                        <div class="val">{len(p_times)}</div>
                        <div class="lbl">Peaks Found</div></div>""",
                        unsafe_allow_html=True)
                with m2:
                    st.markdown(f"""<div class="metric-box">
                        <div class="val">{max(best_offsets.values()) if best_offsets else 0}</div>
                        <div class="lbl">Aligned Matches</div></div>""",
                        unsafe_allow_html=True)
                with m3:
                    st.markdown(f"""<div class="metric-box">
                        <div class="val">{len(all_scores)}</div>
                        <div class="lbl">Songs Compared</div></div>""",
                        unsafe_allow_html=True)
                with m4:
                    dur = round(len(y)/sr, 1)
                    st.markdown(f"""<div class="metric-box">
                        <div class="val">{dur}s</div>
                        <div class="lbl">Clip Duration</div></div>""",
                        unsafe_allow_html=True)

                st.markdown("<hr class='divider'>", unsafe_allow_html=True)

                # ── Three intermediate step plots ──────────────────────────
                st.markdown("#### Intermediate Steps")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown('<span class="step-label">STEP 1</span>', unsafe_allow_html=True)
                    st.markdown("**Spectrogram**")
                    st.caption("Short-Time Fourier Transform showing frequency content over time")
                    fig1 = plot_spectrogram(S_db, sr)
                    st.pyplot(fig1, use_container_width=True)
                    plt.close(fig1)

                with col2:
                    st.markdown('<span class="step-label">STEP 2</span>', unsafe_allow_html=True)
                    st.markdown("**Constellation Peaks**")
                    st.caption("Local maxima — the sparse fingerprint of the song")
                    fig2 = plot_constellation(S_db, sr, p_times, p_freqs)
                    st.pyplot(fig2, use_container_width=True)
                    plt.close(fig2)

                with col3:
                    st.markdown('<span class="step-label">STEP 3</span>', unsafe_allow_html=True)
                    st.markdown("**Offset Histogram**")
                    st.caption("Hash matches aligned at one time offset → correct match")
                    fig3 = plot_offset_histogram(best_offsets, best_song)
                    st.pyplot(fig3, use_container_width=True)
                    plt.close(fig3)

                st.markdown("<hr class='divider'>", unsafe_allow_html=True)

                # ── All songs comparison ───────────────────────────────────
                st.markdown("#### All Song Match Scores")
                col_chart, col_table = st.columns([3, 2])

                with col_chart:
                    fig4 = plot_scores_bar(all_scores, best_song)
                    st.pyplot(fig4, use_container_width=True)
                    plt.close(fig4)

                with col_table:
                    scores_df = pd.DataFrame(
                        sorted(all_scores.items(), key=lambda x: -x[1]),
                        columns=["Song", "Max Matches"]
                    )
                    scores_df.index += 1
                    st.dataframe(scores_df, use_container_width=True, height=300)

            except Exception as e:
                st.error(f"**Error during identification:** {e}")
                st.info("Make sure `database/song_db.pkl` exists and `fingerprint.py` is present.")

        os.unlink(tmp_path)

# ══════════════════════════════════════════════════════════════════════════════
# MODE: BATCH
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "📂 Batch Mode":
    st.markdown("### Batch Identification")
    st.markdown("Upload multiple clips — results will be exported as `results.csv` with exact format required.")

    st.info("📋 Output format: `filename, prediction` — prediction is the matched song name **without extension**")

    uploaded_files = st.file_uploader(
        "Upload multiple query clips",
        type=["mp3", "wav", "flac"],
        accept_multiple_files=True,
        help="Select all clips at once"
    )

    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} file(s) selected:**")
        for f in uploaded_files:
            st.caption(f"• {f.name}")

        if st.button("▶ Run Batch Identification", type="primary"):
            results  = []
            progress = st.progress(0)
            status   = st.empty()
            log_area = st.empty()
            log_lines = []

            for i, f in enumerate(uploaded_files):
                status.markdown(f"⏳ Processing **{f.name}** ({i+1}/{len(uploaded_files)})")
                suffix = os.path.splitext(f.name)[-1] or ".mp3"

                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(f.read())
                    tmp_path = tmp.name

                try:
                    best_song, _, _, _, _, _ = identify(tmp_path, DB_PATH)
                    # prediction = filename without extension
                    prediction = best_song
                    log_lines.append(f"✅ `{f.name}` → **{prediction}**")
                except Exception as e:
                    prediction = "error"
                    log_lines.append(f"❌ `{f.name}` → error: {e}")

                results.append({
                    "filename"  : f.name,
                    "prediction": prediction
                })
                os.unlink(tmp_path)
                progress.progress((i + 1) / len(uploaded_files))
                log_area.markdown("\n\n".join(log_lines))

            status.markdown("✅ **Batch complete!**")

            # Build exact CSV format
            df = pd.DataFrame(results)[["filename", "prediction"]]

            st.markdown("<hr class='divider'>", unsafe_allow_html=True)
            st.markdown("#### Results")
            st.dataframe(df, use_container_width=True)

            csv_str = df.to_csv(index=False)
            st.download_button(
                label="📥 Download results.csv",
                data=csv_str,
                file_name="results.csv",
                mime="text/csv",
                type="primary"
            )

            st.code(csv_str, language="csv")

# ══════════════════════════════════════════════════════════════════════════════
# MODE: SONG LIBRARY
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "📚 Song Library":
    st.markdown("### Song Library")
    st.markdown("All songs currently indexed in the database.")

    indexed_songs = get_indexed_songs(DB_PATH)

    if not indexed_songs:
        st.error("No songs found in database. Check `database/song_db.pkl`.")
    else:
        # Summary
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""<div class="metric-box">
                <div class="val">{len(indexed_songs)}</div>
                <div class="lbl">Total Songs Indexed</div></div>""",
                unsafe_allow_html=True)
        with c2:
            # Count total hashes
            with open(DB_PATH, 'rb') as f:
                db = pickle.load(f)
            st.markdown(f"""<div class="metric-box">
                <div class="val">{len(db):,}</div>
                <div class="lbl">Total Hash Entries</div></div>""",
                unsafe_allow_html=True)

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        # Search box
        search = st.text_input("🔍 Search songs", placeholder="Type to filter...")

        filtered = [s for s in indexed_songs
                    if search.lower() in s.lower()] if search else indexed_songs

        st.markdown(f"Showing **{len(filtered)}** of **{len(indexed_songs)}** songs")
        st.markdown("")

        # Song cards
        for i, song in enumerate(filtered, 1):
            # Check if actual file exists in songs folder
            file_exists = any(
                os.path.exists(os.path.join(SONGS_DIR, f"{song}{ext}"))
                for ext in ['.mp3', '.wav', '.flac', '.MP3', '.WAV']
            )
            icon = "🎵" if file_exists else "💾"

            st.markdown(f"""
            <div class="song-card">
                <span style="font-size:1.3rem">{icon}</span>
                <span style="color:#e6edf3; font-size:0.95rem; flex:1">{song}</span>
                <span style="color:#8b949e; font-size:0.8rem">#{i}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.caption("🎵 = audio file present in songs/   💾 = indexed in database only")
