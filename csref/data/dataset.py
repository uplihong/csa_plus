import os
from typing import List, Optional, Tuple, Callable
import soundfile as sf
import librosa
import numpy as np
from torch.utils.data import Dataset
from ..utils.logging_utils import setup_logger
from ..utils.distributed import is_rank_0

logger = setup_logger(name=__name__)

class LibriSpeechDataset(Dataset):
    def __init__(
        self, 
        root_dir: str, 
        train_split: str = 'train', 
        max_durations: Optional[float] = None,
        use_trim: bool = True,
        target_sample_rate: int = 16000,
        transform: Optional[Callable] = None
    ):
        self.root_dir = root_dir
        self.target_sample_rate = target_sample_rate
        self.use_trim = use_trim
        self.max_durations = max_durations
        self.transform = transform

        if train_split == 'train':
            self.splits = ['train-clean-100', 'train-clean-360', 'train-other-500']
        elif train_split == 'val':
            self.splits = ['dev-clean', 'dev-other']
        elif train_split == 'test':
            self.splits = ['test-clean', 'test-other']
        else:
            raise ValueError(f"Invalid split: {train_split}")

        self.speech_files: List[str] = []
        self.transcripts: List[str] = []

        self._load_data()

        if is_rank_0():
            logger.info(f'====== Dataset {train_split} loaded! ======')
            logger.info(f'Max durations: {max_durations}')
            logger.info(f'is Trimmed: {use_trim}')
            logger.info(f'Target sample rate: {target_sample_rate}')
            logger.info(f'num of samples: {len(self.speech_files)}')

    def _load_data(self):
        for split in self.splits:
            split_path = os.path.join(self.root_dir, split)
            if not os.path.exists(split_path):
                logger.warning(f"Split path does not exist: {split_path}")
                continue

            for speaker in os.listdir(split_path):
                speaker_path = os.path.join(split_path, speaker)
                if not os.path.isdir(speaker_path): continue
                
                for chapter in os.listdir(speaker_path):
                    chapter_path = os.path.join(speaker_path, chapter)
                    if not os.path.isdir(chapter_path): continue

                    transcripts_dict = {}
                    # First pass: read transcripts
                    for file in os.listdir(chapter_path):
                        if file.endswith('.txt'):
                            file_path = os.path.join(chapter_path, file)
                            try:
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    for line in f:
                                        parts = line.strip().split(' ', 1)
                                        if len(parts) == 2:
                                            key, value = parts
                                            transcripts_dict[key] = value
                            except Exception as e:
                                logger.warning(f"Error reading transcript {file_path}: {e}")

                    # Second pass: match audio to transcript
                    for file in os.listdir(chapter_path):
                        if file.endswith('.flac'):
                            file_path = os.path.join(chapter_path, file)
                            file_id = file.split('.')[0]
                            if file_id in transcripts_dict:
                                self.speech_files.append(file_path)
                                self.transcripts.append(transcripts_dict[file_id])

    def load_audio(self, idx: int) -> np.ndarray:
        speech_file = self.speech_files[idx]
        return self.get_audio_by_path(speech_file)

    def get_audio_by_path(self, path: str) -> np.ndarray:
        # sf.read returns (frames, channels) or just frames if mono
        wav, origin_sample_rate = sf.read(path, dtype="float32")
        
        # Ensure mono
        if wav.ndim > 1:
            wav = wav.mean(axis=1)

        # Resample if needed
        if origin_sample_rate != self.target_sample_rate:
            wav = librosa.resample(y=wav, orig_sr=origin_sample_rate, target_sr=self.target_sample_rate)

        # Trim silence
        if self.use_trim:
            wav, _ = librosa.effects.trim(wav)
            
        # Initial max duration cut (hard cut) if specified in init AND not using transform
        # The previous code had this logic. If we use RandomAudioSlice, we might not want this hard cut here
        # The logic was: if max_durations is set, cut to that. 
        # But RandomAudioSlice does a random cut. 
        # I'll keep this check but maybe it's redundant if transform is used.
        if self.max_durations is not None and self.transform is None:
            n_kept_frames = int(self.max_durations * self.target_sample_rate)
            if len(wav) > n_kept_frames:
                wav = wav[0: n_kept_frames]

        return wav

    def __len__(self) -> int:
        return len(self.speech_files)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, str]:
        waveform = self.load_audio(idx)
        transcript = self.transcripts[idx]

        if self.transform:
            waveform, transcript = self.transform(waveform, transcript)

        return waveform, transcript
