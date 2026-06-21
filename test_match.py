from fingerprint import identify

best_song, S_db, p_times, p_freqs, all_scores, _ = identify("songs/Bohemian Rhapsody.mp3")
print("Matched:", best_song)
print(sorted(all_scores.items(), key=lambda x: -x[1])[:5])