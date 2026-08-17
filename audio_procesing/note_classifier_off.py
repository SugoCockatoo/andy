#!/usr/bin/env python3
"""
Offline note + string detector for Andy (prerecorded audio file).
Single-file, self-contained version -- no local imports needed.

Given a .wav file of playing, this:
  1. Builds a reference table from your labeled sample folder (finds the
     dominant/"highest point" frequency in each sample via FFT peak-picking).
  2. Detects individual note onsets in the target recording via an RMS
     energy envelope.
  3. Estimates the pitch of each detected note with YIN and matches it to
     the closest reference entry.
  4. Prints (and optionally CSV-exports) the note+string sequence with
     timestamps.

Usage:
    python note_detector_offline.py path\\to\\recording.wav
    python note_detector_offline.py path\\to\\recording.wav --csv out.csv

Dependencies:
    pip install numpy scipy

CAVEATS (read before trusting the output):

1. "Highest point in the spectrum" for the reference samples = the loudest
   frequency bin, not necessarily the true fundamental. Plucked strings
   often have overtones louder than the fundamental. If a sample's
   detected frequency looks like an exact multiple/half of a neighboring
   entry, that's a sign the peak-picker locked onto a harmonic instead of
   the fundamental -- this script warns about those automatically.

2. Some notes will genuinely share (near-)identical frequency across
   different strings -- that's physics, not a bug. Those are flagged as
   AMBIGUOUS; pitch alone can't resolve them.

3. Onset detection here is a simple RMS-envelope threshold: it calls a new
   note whenever energy jumps sharply out of near-silence. This works well
   for cleanly separated plucked notes. It will struggle with legato
   passages, very quiet notes, or ornaments faster than ~150ms apart. If
   you need it robust for real playing later, swap detect_onsets() for a
   proper spectral-flux detector (aubio or librosa.onset.onset_detect).
"""

import os
import re
import sys
import argparse
import numpy as np
from scipy.io import wavfile

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SAMPLES_FOLDER = r"C:\Users\betas\OneDrive\Documentos\Elite_Grup\bogota_wro\audio_procesing\bandola_samples"

FMIN_HZ = 80.0     # lowest plausible bandola fundamental to search for
FMAX_HZ = 2000.0   # highest plausible bandola fundamental to search for

FILENAME_RE = re.compile(
    r"^([a-gA-G](?:sharp)?)(\d+)(?:\(\d+\))?\.wav$", re.IGNORECASE
)

FRAME_SIZE = 1024
HOP_SIZE = 256
SILENCE_RMS = 0.01          # below this = definitely silence
ONSET_FACTOR = 3.0          # onset threshold = noise_floor * ONSET_FACTOR
MIN_NOTE_GAP_SEC = 0.15     # ignore re-triggers closer together than this
ANALYSIS_SKIP_SEC = 0.03    # skip attack transient before estimating pitch
ANALYSIS_WINDOW_SEC = 0.30  # how much steady-state audio to analyze per note


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------
def load_mono(filepath):
    """Load a wav file, downmix to mono, normalize to [-1, 1]. Returns (sr, samples)."""
    sr, data = wavfile.read(filepath)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float64)
    peak = np.max(np.abs(data)) or 1.0
    data = data / peak
    return sr, data


# ---------------------------------------------------------------------------
# Reference table (built from labeled sample files)
# ---------------------------------------------------------------------------
def dominant_frequency(sr, samples, fmin=FMIN_HZ, fmax=FMAX_HZ):
    """Find the loudest frequency bin ('highest point') in the spectrum,
    with parabolic interpolation for sub-bin precision. Skips the first
    ~40ms (attack transient) and analyzes a steady-state window instead."""
    skip = int(sr * 0.04)
    window_len = min(len(samples) - skip, int(sr * 0.4))
    if window_len <= 0:
        skip = 0
        window_len = len(samples)
    segment = samples[skip:skip + window_len]
    if len(segment) < 32:
        return None

    windowed = segment * np.hanning(len(segment))
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(windowed), d=1.0 / sr)

    band = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(band):
        return None
    band_idx = np.where(band)[0]
    peak_local = np.argmax(spectrum[band_idx])
    peak_idx = band_idx[peak_local]

    if 1 <= peak_idx < len(spectrum) - 1:
        a, b, c = spectrum[peak_idx - 1], spectrum[peak_idx], spectrum[peak_idx + 1]
        denom = (a - 2 * b + c)
        shift = 0.5 * (a - c) / denom if denom != 0 else 0.0
    else:
        shift = 0.0

    bin_width = sr / len(windowed)
    return freqs[peak_idx] + shift * bin_width


def build_reference_from_folder(folder):
    if not os.path.isdir(folder):
        print(f"ERROR: folder not found: {folder}", file=sys.stderr)
        sys.exit(1)

    entries = []
    skipped = []
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith(".wav"):
            continue
        m = FILENAME_RE.match(fname)
        if not m:
            skipped.append(fname)
            continue
        note_raw, string_raw = m.groups()
        note = note_raw.lower()
        string = int(string_raw)

        filepath = os.path.join(folder, fname)
        try:
            sr, samples = load_mono(filepath)
            freq = dominant_frequency(sr, samples)
        except Exception as exc:
            print(f"  WARNING: failed to analyze {fname}: {exc}", file=sys.stderr)
            continue
        if freq is None:
            print(f"  WARNING: no clear peak found in {fname}", file=sys.stderr)
            continue

        pretty_note = note.replace("sharp", "#").upper()
        label = f"{pretty_note} / string {string}"
        entries.append({
            "note": note, "string": string, "freq": freq,
            "label": label, "filename": fname,
        })

    if skipped:
        print(f"Skipped {len(skipped)} file(s) that didn't match the naming "
              f"pattern <note><string>.wav: {skipped}")

    entries.sort(key=lambda e: e["freq"])
    return entries


def flag_octave_suspects(entries, tolerance_cents=15.0):
    """Warn about pairs that look like one entry mistakenly locked onto a
    harmonic (roughly 2x, 3x, 0.5x, etc. of another entry's frequency)."""
    ratios_to_check = [2.0, 3.0, 0.5, 1.0 / 3.0]
    warnings = []
    for i, a in enumerate(entries):
        for b in entries[i + 1:]:
            for r in ratios_to_check:
                expected = a["freq"] * r
                if expected <= 0:
                    continue
                cents = abs(1200 * np.log2(b["freq"] / expected))
                if cents <= tolerance_cents:
                    warnings.append(
                        f"  {b['filename']} ({b['freq']:.2f} Hz) looks like "
                        f"~{r:g}x {a['filename']} ({a['freq']:.2f} Hz) -- "
                        f"possible harmonic lock, worth re-checking manually."
                    )
    return warnings


def flag_ambiguous_groups(entries, cents_tolerance=5.0):
    """Mark entries that share (near-)identical frequency with another
    entry on a *different* string -- these cannot be told apart by pitch."""
    for e in entries:
        e["ambiguous_with"] = []
    for i, a in enumerate(entries):
        for b in entries[i + 1:]:
            cents = 0.0 if abs(a["freq"] - b["freq"]) < 0.01 else \
                abs(1200 * np.log2(a["freq"] / b["freq"]))
            if cents <= cents_tolerance and a["string"] != b["string"]:
                a["ambiguous_with"].append(b["label"])
                b["ambiguous_with"].append(a["label"])
    return entries


def load_reference_table(folder=SAMPLES_FOLDER, verbose=True):
    """Convenience wrapper: build + flag + return (entries, freqs_array)."""
    if verbose:
        print(f"Scanning {folder} ...")
    entries = build_reference_from_folder(folder)
    if not entries:
        print("No usable samples found -- check the folder path and filenames.")
        sys.exit(1)

    if verbose:
        print(f"\nBuilt reference table from {len(entries)} sample(s):")
        for e in entries:
            print(f"  {e['filename']:24s} {e['freq']:9.2f} Hz   {e['label']}")

    octave_warnings = flag_octave_suspects(entries)
    if octave_warnings and verbose:
        print("\nPossible harmonic-lock warnings:")
        for w in octave_warnings:
            print(w)

    entries = flag_ambiguous_groups(entries)
    ambiguous = [e for e in entries if e["ambiguous_with"]]
    if ambiguous and verbose:
        print(f"\n{len(ambiguous)} entries share pitch with another string "
              f"(will be reported as ambiguous):")
        seen = set()
        for e in ambiguous:
            key = tuple(sorted([e["label"]] + e["ambiguous_with"]))
            if key not in seen:
                seen.add(key)
                print(f"  - {' == '.join(key)}")

    freqs_array = np.array([e["freq"] for e in entries])
    return entries, freqs_array


def match_note(freq_hz, reference, freqs_array, max_cents=60):
    if freq_hz is None or len(reference) == 0:
        return None, None
    cents = 1200 * np.log2(freqs_array / freq_hz)
    idx = int(np.argmin(np.abs(cents)))
    off = cents[idx]
    if abs(off) > max_cents:
        return None, None
    return reference[idx], off


# ---------------------------------------------------------------------------
# YIN pitch detection
# ---------------------------------------------------------------------------
def yin_pitch(signal, sr, fmin=FMIN_HZ, fmax=FMAX_HZ, threshold=0.12):
    signal = signal.astype(np.float64)
    signal = signal - np.mean(signal)

    tau_min = max(1, int(sr / fmax))
    tau_max = int(sr / fmin)
    tau_max = min(tau_max, len(signal) - 1)
    if tau_max <= tau_min:
        return None

    n = len(signal)
    size = 1
    while size < 2 * n:
        size *= 2
    fft_sig = np.fft.rfft(signal, size)
    acf = np.fft.irfft(fft_sig * np.conj(fft_sig))[:tau_max + 1]

    energy = np.cumsum(signal ** 2)
    energy = np.concatenate(([0.0], energy))
    total_energy = energy[n] - energy[0]

    d = np.zeros(tau_max + 1)
    for tau in range(1, tau_max + 1):
        d[tau] = 2 * total_energy - 2 * acf[tau]
    d[0] = 0.0

    cmnd = np.ones(tau_max + 1)
    running_sum = 0.0
    for tau in range(1, tau_max + 1):
        running_sum += d[tau]
        cmnd[tau] = d[tau] * tau / running_sum if running_sum > 0 else 1.0

    tau_est = None
    for tau in range(tau_min, tau_max + 1):
        if cmnd[tau] < threshold:
            while tau + 1 <= tau_max and cmnd[tau + 1] < cmnd[tau]:
                tau += 1
            tau_est = tau
            break

    if tau_est is None:
        tau_est = tau_min + int(np.argmin(cmnd[tau_min:tau_max + 1]))
        if cmnd[tau_est] > 0.4:
            return None

    if 1 <= tau_est < tau_max:
        s0, s1, s2 = cmnd[tau_est - 1], cmnd[tau_est], cmnd[tau_est + 1]
        denom = (s0 - 2 * s1 + s2)
        shift = 0.5 * (s0 - s2) / denom if denom != 0 else 0.0
    else:
        shift = 0.0

    tau_refined = tau_est + shift
    if tau_refined <= 0:
        return None
    return sr / tau_refined


# ---------------------------------------------------------------------------
# Onset detection + per-note analysis
# ---------------------------------------------------------------------------
def compute_rms_envelope(samples, sr, frame_size=FRAME_SIZE, hop=HOP_SIZE):
    n_frames = max(0, (len(samples) - frame_size) // hop + 1)
    envelope = np.zeros(n_frames)
    times = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop
        frame = samples[start:start + frame_size]
        envelope[i] = np.sqrt(np.mean(frame ** 2))
        times[i] = start / sr
    return times, envelope


def detect_onsets(samples, sr):
    times, envelope = compute_rms_envelope(samples, sr)
    if len(envelope) == 0:
        return []

    noise_floor = max(np.percentile(envelope, 20), 1e-6)
    onset_threshold = max(SILENCE_RMS, noise_floor * ONSET_FACTOR)

    onsets = []
    is_active = False
    last_onset_time = -1e9
    for t, e in zip(times, envelope):
        if not is_active and e > onset_threshold:
            if t - last_onset_time >= MIN_NOTE_GAP_SEC:
                onsets.append(t)
                last_onset_time = t
            is_active = True
        elif is_active and e < onset_threshold * 0.4:
            is_active = False
    return onsets


def analyze_recording(filepath, reference, freqs_array):
    sr, samples = load_mono(filepath)
    onsets = detect_onsets(samples, sr)

    results = []
    for i, onset_t in enumerate(onsets):
        end_t = onsets[i + 1] if i + 1 < len(onsets) else len(samples) / sr

        analysis_start = onset_t + ANALYSIS_SKIP_SEC
        analysis_end = min(analysis_start + ANALYSIS_WINDOW_SEC, end_t)
        start_idx = int(analysis_start * sr)
        end_idx = int(analysis_end * sr)
        segment = samples[start_idx:end_idx]

        if len(segment) < 256:
            results.append({
                "index": i + 1, "start": onset_t, "end": end_t,
                "freq": None, "entry": None, "cents": None,
            })
            continue

        freq = yin_pitch(segment, sr, FMIN_HZ, FMAX_HZ)
        entry, cents = match_note(freq, reference, freqs_array) if freq else (None, None)

        results.append({
            "index": i + 1, "start": onset_t, "end": end_t,
            "freq": freq, "entry": entry, "cents": cents,
        })
    return results


def print_results(results):
    print(f"\nDetected {len(results)} note(s):\n")
    for r in results:
        idx, start, end = r["index"], r["start"], r["end"]
        if r["entry"] is None:
            status = "no confident match" if r["freq"] else "unclear pitch"
            print(f"  {idx:3d}.  {start:6.2f}s - {end:6.2f}s   ({status})")
            continue

        entry, cents = r["entry"], r["cents"]
        label = entry["label"]
        if entry["ambiguous_with"]:
            label += f"  [AMBIGUOUS also matches: {', '.join(entry['ambiguous_with'])}]"
        print(f"  {idx:3d}.  {start:6.2f}s - {end:6.2f}s   {label:35s} "
              f"({r['freq']:7.2f} Hz, {cents:+.1f} cents)")


def write_csv(results, path):
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "start_sec", "end_sec", "note", "string",
                          "freq_hz", "cents_off", "ambiguous_with"])
        for r in results:
            entry = r["entry"]
            writer.writerow([
                r["index"], f"{r['start']:.3f}", f"{r['end']:.3f}",
                entry["note"] if entry else "", entry["string"] if entry else "",
                f"{r['freq']:.2f}" if r["freq"] else "",
                f"{r['cents']:.1f}" if r["cents"] is not None else "",
                "; ".join(entry["ambiguous_with"]) if entry and entry["ambiguous_with"] else "",
            ])
    print(f"\nWrote {len(results)} row(s) to {path}")


def main():
    parser = argparse.ArgumentParser(description="Detect notes+strings in a prerecorded audio file.")
    parser.add_argument("audio_file", help="Path to the .wav file to analyze")
    parser.add_argument("--samples-folder", default=SAMPLES_FOLDER,
                         help="Folder of labeled reference samples (default: hardcoded Andy folder)")
    parser.add_argument("--csv", default=None, help="Optional path to write results as CSV")
    args = parser.parse_args()

    reference, freqs_array = load_reference_table(args.samples_folder)

    print(f"\nAnalyzing {args.audio_file} ...")
    results = analyze_recording(args.audio_file, reference, freqs_array)
    print_results(results)

    if args.csv:
        write_csv(results, args.csv)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)