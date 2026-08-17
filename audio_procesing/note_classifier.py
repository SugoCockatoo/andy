#!/usr/bin/env python3
"""
Live note + string detector for Andy.

Step 1: Scans a folder of labeled .wav samples (e.g. "gsharp4.wav"), finds
        the dominant ("highest point") frequency in each file's spectrum,
        and builds a note+string reference table from the results.
Step 2: Listens to the microphone live, estimates the fundamental frequency
        with YIN, and matches it against that reference table.

Dependencies:
    pip install sounddevice numpy scipy

IMPORTANT CAVEATS (read before trusting the output):

1. "Highest point in the spectrum" = the loudest frequency bin, not
   necessarily the true fundamental. Plucked strings often have overtones
   louder than the fundamental, especially higher up the neck. If a
   detected frequency looks like an exact multiple/half of a neighboring
   entry (e.g. your old g1 = 787.50 Hz and g1(2) = 1575.00 Hz -- exactly
   2x), that's a sign the peak-picker locked onto a harmonic instead of
   the fundamental on one of the two takes. This script prints a warning
   for any such octave-ish relationships it finds so you can go re-check
   those specific recordings.

2. Some notes will still end up sharing (near-)identical frequencies
   across different strings -- that's physics, not a bug. Those are
   flagged as AMBIGUOUS at both build time and live-match time. Pitch
   alone can't resolve them; you'll need your timbral/spectral classifier
   for that eventually.
"""

import os
import re
import sys
import time
import queue
import numpy as np
import sounddevice as sd
from scipy.io import wavfile

# ---------------------------------------------------------------------------
# 0. Config
# ---------------------------------------------------------------------------
SAMPLES_FOLDER = r"C:\Users\betas\OneDrive\Documentos\Elite_Grup\bogota_wro\audio_procesing\bandola_samples"

FMIN_HZ = 80.0     # lowest plausible bandola fundamental to search for
FMAX_HZ = 2000.0   # highest plausible bandola fundamental to search for

FILENAME_RE = re.compile(
    r"^([a-gA-G](?:sharp)?)(\d+)(?:\(\d+\))?\.wav$", re.IGNORECASE
)


# ---------------------------------------------------------------------------
# 1. Build reference table from the sample folder
# ---------------------------------------------------------------------------
def load_mono(filepath):
    sr, data = wavfile.read(filepath)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float64)
    peak = np.max(np.abs(data)) or 1.0
    data = data / peak
    return sr, data


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
# 2. YIN pitch detection (live input)
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
# 3. Live audio loop
# ---------------------------------------------------------------------------
SAMPLE_RATE = 44100
BLOCK_SIZE = 4096
SILENCE_RMS = 0.01
HOLD_SECONDS = 0.35

audio_q = queue.Queue()


def audio_callback(indata, frames, time_info, status):
    if status:
        print(status, file=sys.stderr)
    audio_q.put(indata[:, 0].copy())


def main():
    print(f"Scanning {SAMPLES_FOLDER} ...")
    reference = build_reference_from_folder(SAMPLES_FOLDER)
    if not reference:
        print("No usable samples found -- check the folder path and filenames.")
        sys.exit(1)

    print(f"\nBuilt reference table from {len(reference)} sample(s):")
    for e in reference:
        print(f"  {e['filename']:24s} {e['freq']:9.2f} Hz   {e['label']}")

    octave_warnings = flag_octave_suspects(reference)
    if octave_warnings:
        print("\nPossible harmonic-lock warnings:")
        for w in octave_warnings:
            print(w)

    reference = flag_ambiguous_groups(reference)
    ambiguous = [e for e in reference if e["ambiguous_with"]]
    if ambiguous:
        print(f"\n{len(ambiguous)} entries share pitch with another string "
              f"(will be reported as ambiguous live):")
        seen = set()
        for e in ambiguous:
            key = tuple(sorted([e["label"]] + e["ambiguous_with"]))
            if key not in seen:
                seen.add(key)
                print(f"  - {' == '.join(key)}")

    freqs_array = np.array([e["freq"] for e in reference])

    print("\nListening... (Ctrl+C to stop)\n")
    last_label = None
    last_time = 0.0

    with sd.InputStream(channels=1, samplerate=SAMPLE_RATE,
                         blocksize=BLOCK_SIZE, callback=audio_callback):
        while True:
            block = audio_q.get()
            rms = np.sqrt(np.mean(block ** 2))
            if rms < SILENCE_RMS:
                continue

            freq = yin_pitch(block, SAMPLE_RATE)
            if freq is None:
                continue

            entry, cents_off = match_note(freq, reference, freqs_array)
            if entry is None:
                continue

            if entry["ambiguous_with"]:
                label = f"{entry['label']}  [AMBIGUOUS also matches: {', '.join(entry['ambiguous_with'])}]"
            else:
                label = entry["label"]

            now = time.time()
            if label != last_label or (now - last_time) > HOLD_SECONDS:
                print(f"{freq:7.2f} Hz  ->  {label}   ({cents_off:+.1f} cents)")
                last_label = label
                last_time = now


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")