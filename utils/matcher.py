# utils/matcher.py

from collections import defaultdict, Counter


# ==========================================
# Build Database
# ==========================================

def build_song_database(song_fingerprints):
    """
    song_fingerprints format:

    {
        "song_name": hashes
    }

    Creates:

    {
        hash_key: [
            (song_name, song_time),
            ...
        ]
    }
    """

    database = defaultdict(list)

    for song_name, hashes in song_fingerprints.items():

        for hash_key, song_time in hashes:

            database[hash_key].append(
                (
                    song_name,
                    song_time
                )
            )

    return database


# ==========================================
# Match Query Against Database
# ==========================================

def match_query(query_hashes, database):
    """
    Returns best matching song using
    offset histogram voting.
    """

    offset_votes = defaultdict(list)

    for hash_key, query_time in query_hashes:

        if hash_key not in database:
            continue

        matches = database[hash_key]

        for song_name, song_time in matches:

            offset = round(
                song_time - query_time,
                2
            )

            offset_votes[song_name].append(offset)

    best_song = None
    best_score = 0
    best_histogram = None

    for song_name, offsets in offset_votes.items():

        histogram = Counter(offsets)

        peak_count = max(histogram.values())

        if peak_count > best_score:
            best_score = peak_count
            best_song = song_name
            best_histogram = histogram

    return best_song, best_score, best_histogram


# ==========================================
# Confidence Score
# ==========================================

def calculate_confidence(score, total_hashes):
    """
    Convert vote count into percentage.
    """

    if total_hashes == 0:
        return 0

    confidence = (
        score / total_hashes
    ) * 100

    return round(confidence, 2)


# ==========================================
# Complete Recognition Pipeline
# ==========================================

def identify_song(
    query_hashes,
    database
):
    """
    Returns:

    song_name
    confidence
    histogram
    """

    song_name, score, histogram = match_query(
        query_hashes,
        database
    )

    confidence = calculate_confidence(
        score,
        len(query_hashes)
    )

    return (
        song_name,
        confidence,
        histogram
    )


# ==========================================
# Batch Recognition
# ==========================================

def batch_identify(
    fingerprints,
    database
):
    """
    fingerprints:

    [
        ("clip1.wav", hashes),
        ("clip2.wav", hashes),
        ...
    ]
    """

    results = []

    for filename, hashes in fingerprints:

        song_name, _, _ = identify_song(
            hashes,
            database
        )

        results.append(
            (
                filename,
                song_name
            )
        )

    return results
