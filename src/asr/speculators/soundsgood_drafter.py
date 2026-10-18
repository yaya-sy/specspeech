from pathlib import Path
from itertools import islice, zip_longest

import numpy as np

from .base_drafter import BaseDrafter


class SoundsGoodDrafter(BaseDrafter):
    """Parakeet TDT or Zipformer transducer through SoundsGoodAI/fast-gpu-asr.

    Load the local TensorRT engine produced by scripts/export.sh.
    Model loading happens before timed inference.
    drafter_max_audio_seconds limits the draft pieces within each verifier request.
    """

    ENGINES = {
        "parakeet": "exported/parakeet-tdt-0.6b-v3",
        "zipformer": "exported/zipformer",
    }

    def __init__(
        self,
        engine_path: str | Path | None = None,
        device_id: int = 0,
        *,
        model: str = "parakeet",
        drafter_max_audio_seconds: int | None = None,
    ):
        self.drafter_max_audio_seconds = drafter_max_audio_seconds if drafter_max_audio_seconds is not None else (
            300 if model == "parakeet" else 40
        )
        try:
            from fast_gpu_asr import ASR
        except ModuleNotFoundError as exc:
            if exc.name != "fast_gpu_asr":
                raise
            raise ImportError(
                "Parakeet and Zipformer require the optional SoundsGood backend "
                "and a CUDA 13-compatible GPU environment. Install the 'soundsgood' "
                "extra with the compatibility wheel as described in scripts/README.md."
            ) from exc

        self.engine_path = Path(engine_path or self.ENGINES[model]).expanduser()
        self.asr = ASR(str(self.engine_path), device_id=device_id)
        self.batch_size = self.asr.encoder.batch_size
        self.encoder_min_samples = self.asr.encoder.min_samples

    def prepare(self, audio: np.ndarray) -> np.ndarray:
        # Both engines receive the same mono 16 kHz waveform.
        return audio

    def transcribe(self, prepared: np.ndarray) -> str:
        return next(self.transcribe_batch([prepared]))[1]

    def transcribe_batch(self, prepared):
        """Join piece transcripts and yield completed requests."""
        parts = [[] for _ in prepared]
        for batch in self.stream_batch(prepared):
            for index, text, last in batch:
                parts[index].append(text)
                if last:
                    yield index, " ".join(parts[index])

    def stream_batch(self, prepared, *, is_finished=lambda index: False):
        """Interleave requests' pieces and publish each completed ASR batch."""
        size = self.drafter_max_audio_seconds * 16000
        requests = [[(index, audio[start:start + size], start + size >= len(audio))
                     for start in range(0, len(audio), size)]
                    for index, audio in enumerate(prepared)]
        pieces = (piece for row in zip_longest(*requests) for piece in row
                  if piece is not None and not is_finished(piece[0]))
        while batch := list(islice(pieces, self.batch_size)):
            audios = [audio for _, audio, _ in batch]
            encoder = self.asr.encoder
            # Round encoder padding to 100 ms; real waveform lengths stay unchanged.
            padded_samples = (max(map(len, audios)) + 1599) // 1600 * 1600
            encoder.min_samples = max(self.encoder_min_samples, min(encoder.max_samples, padded_samples))
            texts, _ = self.asr(audios)
            yield [(index, text, last) for (index, _, last), text in zip(batch, texts)]
