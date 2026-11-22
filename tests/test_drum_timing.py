import unittest

import numpy as np

from drum_timing import first_drum_beat_origin


class FirstDrumBeatOriginTest(unittest.TestCase):
    def test_missed_first_hit_does_not_move_the_bar(self):
        sr = 8000
        bpm = 128
        first_beat = 8.14125
        next_beat = first_beat + 60 / bpm
        audio = np.zeros(12 * sr, dtype=np.float32)
        audio[int(first_beat * sr):int((first_beat + 0.08) * sr)] = 0.4
        audio[int(next_beat * sr):int((next_beat + 0.08) * sr)] = 0.8
        beats = np.arange(first_beat - 4 * 60 / bpm, next_beat + 60 / bpm, 60 / bpm)

        origin = first_drum_beat_origin(audio, sr, bpm, beats, {"kick": [next_beat]})
        self.assertAlmostEqual(origin, first_beat)

    def test_silence_before_first_hit_keeps_original_start(self):
        sr = 8000
        bpm = 128
        first_beat = 8.14125
        next_beat = first_beat + 60 / bpm
        audio = np.zeros(12 * sr, dtype=np.float32)
        audio[int(next_beat * sr):int((next_beat + 0.08) * sr)] = 0.8
        beats = np.arange(first_beat - 4 * 60 / bpm, next_beat + 60 / bpm, 60 / bpm)

        origin = first_drum_beat_origin(audio, sr, bpm, beats, {"kick": [next_beat]})
        self.assertAlmostEqual(origin, next_beat)


if __name__ == '__main__':
    unittest.main()
