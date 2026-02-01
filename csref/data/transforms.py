import random
from typing import Tuple, Optional, Union
import numpy as np


class RandomAudioSlice:
    """
    Randomly slices the audio and corresponding transcript if the audio exceeds max length.
    """
    def __init__(self, min_length: int, max_length: int, sample_rate: int = 16000):
        self.min_length = min_length
        self.max_length = max_length
        self.sample_rate = sample_rate

    def __call__(self, audio: np.ndarray, transcript: str) -> Tuple[np.ndarray, str]:
        if len(audio) < self.max_length:
            return audio, transcript

        # random cut, select a random length between min and max
        # Check if audio is shorter than min_length (shouldn't happen if logic correct, but safety check)
        actual_min = min(self.min_length, len(audio))
        actual_max = min(self.max_length, len(audio))
        
        if actual_min >= actual_max:
             result_audio_length = actual_max
        else:
            result_audio_length = random.randint(actual_min, actual_max)

        # corresponding transcript length
        if len(audio) > 0:
            result_audio_length_ratio = result_audio_length / len(audio)
            result_transcript_length = max(int(len(transcript) * result_audio_length_ratio), 1)
        else:
            return audio, transcript

        # cut audio and transcript
        start_audio = random.randint(0, len(audio) - result_audio_length)
        result_audio = audio[start_audio: start_audio + result_audio_length]

        start_audio_ratio = start_audio / len(audio)
        start_transcript = int(start_audio_ratio * len(transcript))
        
        # Ensure we don't go out of bounds
        end_transcript = min(start_transcript + result_transcript_length, len(transcript))
        result_transcript = transcript[start_transcript: end_transcript]

        return result_audio, result_transcript
