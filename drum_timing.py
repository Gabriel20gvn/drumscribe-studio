"""Find the first audible drum beat before the model's first detected hit."""

import numpy as np


def first_drum_beat_origin(y, sr, bpm, beat_times, drum_events):
    """Anchor the score to a clear drum onset, even if ADTOF missed that hit.

    Beat tracking gives quarter notes, but no bar number. We use its beat phase
    and inspect up to four earlier beats in the separated drum stem. A quieter
    transient is accepted only when it stands well above the surrounding audio.
    """
    onsets = [float(t) for times in drum_events.values() for t in times]
    if not onsets:
        return 0.0
    first_hit = min(onsets)
    beats = np.asarray(beat_times, dtype=float)
    prior_beats = beats[beats <= first_hit + 0.10]
    if prior_beats.size:
        base = float(prior_beats[-1])
    elif beats.size:
        base = min(first_hit, float(beats[0]))
    else:
        base = first_hit
    if not np.isfinite(bpm) or bpm <= 0 or sr <= 0:
        return base

    interval = 60.0 / bpm
    candidates = [base - n * interval for n in range(4, -1, -1) if base - n * interval >= 0]
    samples = np.asarray(y, dtype=float)

    def rms(start, end):
        a = max(0, int(start * sr))
        b = min(len(samples), int(end * sr))
        if b <= a:
            return 0.0
        segment = samples[a:b]
        return float(np.sqrt(np.mean(segment * segment)))

    energies = [(rms(t - 0.14, t - 0.05), rms(t - 0.04, t + 0.08)) for t in candidates]
    reference = max((post for _, post in energies), default=0.0)
    if reference <= 0:
        return base
    for t, (before, after) in zip(candidates[:-1], energies[:-1]):
        if after >= 0.18 * reference and after >= 2.5 * before and after - before >= 0.10 * reference:
            return float(t)
    return base
