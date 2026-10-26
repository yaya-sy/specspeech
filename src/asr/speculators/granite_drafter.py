"""Granite TurboCTC drafts through the native Transformers implementation."""

from itertools import islice, zip_longest

import torch
from transformers import AutoModelForCTC, AutoProcessor

from .base_drafter import BaseDrafter


class GraniteDrafter(BaseDrafter):
    """Batch mono 16 kHz audio pieces and publish punctuated CTC transcripts."""

    MODEL_ID = "ibm-granite/granite-speech-5.0-470m-turboctc"

    def __init__(self, model_id=MODEL_ID, device="cuda", *, batch_size=8,
                 drafter_max_audio_seconds=40, dtype=torch.bfloat16):
        if batch_size <= 0 or drafter_max_audio_seconds <= 0:
            raise ValueError("Batch size and draft piece duration must be positive")
        self.model_id = model_id
        self.batch_size = batch_size
        self.drafter_max_audio_seconds = drafter_max_audio_seconds
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForCTC.from_pretrained(
            model_id, dtype=dtype, attn_implementation="sdpa",
        ).to(device).eval()

    def prepare(self, audio):
        return audio

    def transcribe(self, prepared):
        return next(self.transcribe_batch([prepared]))[1]

    def transcribe_batch(self, prepared):
        """Join piece transcripts and yield completed requests."""
        parts = [[] for _ in prepared]
        for batch in self.stream_batch(prepared):
            for index, text, last in batch:
                parts[index].append(text)
                if last:
                    yield index, " ".join(parts[index])

    @torch.inference_mode()
    def _decode_batch(self, audios):
        inputs = self.processor(
            audio=audios, sampling_rate=16000, device=self.model.device,
            padding=True, return_attention_mask=True, return_tensors="pt",
        ).to(self.model.device, dtype=self.model.dtype)
        # Native generate masks padded frames before CTC collapse in batch_decode.
        tokens = self.model.generate(**inputs)
        return self.processor.batch_decode(tokens, skip_special_tokens=True)

    def stream_batch(self, prepared, *, is_finished=lambda index: False):
        """Interleave requests' pieces and publish each completed CTC batch."""
        size = self.drafter_max_audio_seconds * 16000
        requests = [[(index, audio[start:start + size], start + size >= len(audio))
                     for start in range(0, len(audio), size)]
                    for index, audio in enumerate(prepared)]
        pieces = (piece for row in zip_longest(*requests) for piece in row
                  if piece is not None and not is_finished(piece[0]))
        while batch := list(islice(pieces, self.batch_size)):
            texts = self._decode_batch([audio for _, audio, _ in batch])
            yield [(index, text, last) for (index, _, last), text in zip(batch, texts)]
