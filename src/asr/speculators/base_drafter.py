from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseDrafter(ABC):
    """Common interface for ASR draft engines used by speculative decoding.

    Model loading, input preparation, and warmup happen outside the benchmark.
    The caller should time only `transcribe(prepared)`.
    """

    batch_size = 1

    @abstractmethod
    def prepare(self, audio: np.ndarray) -> Any:
        """Prepare one request from a mono 16 kHz waveform."""
        raise NotImplementedError

    @abstractmethod
    def transcribe(self, prepared: Any) -> str:
        """Run request-time ASR inference + text decoding."""
        raise NotImplementedError

    def transcribe_batch(self, prepared):
        """Yield (request index, transcript) as each draft becomes ready."""
        for index, audio in enumerate(prepared):
            yield index, self.transcribe(audio)

    def stream_batch(self, prepared, *, is_finished=lambda index: False):
        """Yield batches of (request index, piece transcript, final piece)."""
        for index, audio in enumerate(prepared):
            if not is_finished(index):
                yield [(index, self.transcribe(audio), True)]

    def warmup(self, prepared: Any, n: int = 3) -> None:
        """Warm up the backend."""
        for _ in range(n):
            self.transcribe(prepared)
        self.synchronize()

    def synchronize(self) -> None:
        """Wait for pending device work, if the backend needs it."""
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
