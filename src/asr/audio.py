"""Load shared ASR waveforms and slice recordings to the verifier duration limit."""

import logging
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


_slice_warning_logged = False
AUDIO_SUFFIXES = {".wav", ".wave", ".flac", ".ogg", ".oga", ".mp3", ".aif", ".aiff", ".au", ".caf"}


def audio_paths(paths):
    """Yield files and recursively discover audio files inside folders."""
    for path in paths:
        path = Path(path).expanduser()
        if path.is_dir():
            yield from (str(file) for file in path.rglob("*")
                        if file.suffix.lower() in AUDIO_SUFFIXES and file.is_file())
        else:
            yield str(path)


def load_audio(audio_path):
    """Share one mono float32, 16 kHz waveform between both models."""
    audio, sample_rate = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    if sample_rate != 16000:
        divisor = math.gcd(sample_rate, 16000)
        audio = resample_poly(audio, 16000 // divisor, sample_rate // divisor)
    return np.ascontiguousarray(audio, dtype=np.float32)


def audio_segments(paths, verifier_max_audio_seconds=1200):
    """Yield (file/segment metadata, waveform) pairs from files or folders.

    verifier_max_audio_seconds: Maximum duration of each verifier request.
    """
    global _slice_warning_logged
    size = verifier_max_audio_seconds * 16000
    for file_index, path in enumerate(audio_paths(paths)):
        audio = load_audio(path)
        if len(audio) > size and not _slice_warning_logged:
            logging.getLogger(__name__).warning(
                "Audio %s exceeds the verifier limit of %s seconds and will be sliced "
                "at fixed duration boundaries. This can cut speech mid-word and hurt "
                "transcription accuracy and speculative decoding performance. Prepare "
                "audio segments shorter than this limit before running inference, "
                "for example by using a VAD model.",
                path, verifier_max_audio_seconds,
            )
            _slice_warning_logged = True
        for segment_index, start in enumerate(range(0, len(audio), size)):
            yield {"path": path, "file_index": file_index, "segment_index": segment_index}, audio[start:start + size]
