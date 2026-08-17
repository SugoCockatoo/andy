"""
build_reference_table.py

Turns a single recording of "all the natural notes of the Bandola" into a
labeled reference table: {note_name: measured_frequency_hz}.

Workflow:
    1. Run this script on your recording -> it segments notes and guesses
       a label for each, exporting everything to a CSV.
    2. Open the CSV, listen back to the recording alongside it, and correct
       any wrong/missing labels in the "note_name" column (you know the
       order you played them in -- that's your ground truth).
    3. Run load_reference_table() to load the corrected CSV wherever you
       need expected_hz values (e.g. in compare_note() from before).

Dependencies:
    pip install aubio soundfile numpy
"""

import csv
import numpy as np
import soundfile as sf
import aubio

SR = 22050


def load_mono(path, sr=SR):
    """Load an audio file as mono float32, resampled to sr."""
    y, file_sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)  # downmix to mono if stereo
    if file_sr != sr:
        # simple resample via aubio's built-in resampler would need extra setup;
        # easiest correct path is to just record at SR to begin with.
        raise ValueError(f"File is {file_sr} Hz, expected {sr} Hz. Re-record or resample first.")
    return y


def detect_onsets(y, sr=SR, hop_size=512):
    """Return onset sample positions using aubio's onset detector."""
    onset_detector = aubio.onset("default", 1024, hop_size, sr)
    onsets = []
    total = len(y)
    pos = 0
    while pos + hop_size <= total:
        chunk = y[pos:pos + hop_size]
        if onset_detector(chunk):
            onsets.append(onset_detector.get_last())
        pos += hop_size
    return onsets


def segment_by_onsets(y, onsets, sr=SR, min_len_sec=0.15):
    """Split audio into segments between consecutive onsets."""
    bounds = onsets + [len(y)]
    segments = []
    for i in range(len(bounds) - 1):
        start, end = bounds[i], bounds[i + 1]
        if (end - start) / sr >= min_len_sec:
            segments.append((start, end, y[start:end]))
    return segments


def estimate_pitch_hz(segment, sr=SR, hop_size=512):
    """Median pitch estimate for one segment using aubio's YIN implementation."""
    pitch_detector = aubio.pitch("yinfft", 2048, hop_size, sr)
    pitch_detector.set_unit("Hz")
    pitch_detector.set_tolerance(0.8)

    estimates = []
    pos = 0
    while pos + hop_size <= len(segment):
        chunk = segment[pos:pos + hop_size]
        pitch = pitch_detector(chunk)[0]
        confidence = pitch_detector.get_confidence()
        if pitch > 0 and confidence > 0.6:
            estimates.append(pitch)
        pos += hop_size

    if not estimates:
        return None
    return float(np.median(estimates))


# --- Nearest equal-tempered note name, just as a labeling AID (not ground truth) ---

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def nearest_note_name(hz, a4=440.0):
    if hz is None or hz <= 0:
        return "?"
    semitones_from_a4 = 12 * np.log2(hz / a4)
    note_index = int(round(semitones_from_a4)) + 57  # A4 is MIDI note 69; C0 offset = 12
    octave = note_index // 12
    name = NOTE_NAMES[note_index % 12]
    return f"{name}{octave}"


def build_table(audio_path, out_csv="bandola_reference.csv"):
    y = load_mono(audio_path)
    onset_samples = detect_onsets(y)
    segments = segment_by_onsets(y, onset_samples)

    rows = []
    for i, (start, end, seg) in enumerate(segments):
        hz = estimate_pitch_hz(seg)
        guess = nearest_note_name(hz)
        rows.append({
            "index": i,
            "start_sec": round(start / SR, 3),
            "end_sec": round(end / SR, 3),
            "measured_hz": round(hz, 2) if hz else "",
            "note_name": guess,  # <-- REVIEW AND CORRECT THIS COLUMN BY EAR
        })

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "start_sec", "end_sec", "measured_hz", "note_name"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Found {len(rows)} note segments. Exported to {out_csv}.")
    print("Listen back to the recording and correct the 'note_name' column before using this as ground truth.")
    return rows


def load_reference_table(csv_path="bandola_reference.csv"):
    """Load a (manually corrected) CSV into {note_name: hz} for use elsewhere."""
    table = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["note_name"] and row["measured_hz"]:
                table[row["note_name"]] = float(row["measured_hz"])
    return table


if __name__ == "__main__":
    build_table("bandola_all_notes.wav")