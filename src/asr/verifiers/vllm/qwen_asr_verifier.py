"""Qwen3-ASR prompt and audio preparation for the shared vLLM verifier."""

from vllm.transformers_utils.processors.qwen3_asr import Qwen3ASRProcessor

from .verifier import VLLMVerifier


class QwenASRVerifier(VLLMVerifier):
    MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
    ARCHITECTURE = "Qwen3ASRForConditionalGeneration"
    PROCESSOR_CLASS = Qwen3ASRProcessor

    def prepare(self, audio, language="English"):
        prompt = (
            "<|im_start|>system\n<|im_end|>\n"
            "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n"
            "<|im_start|>assistant\n"
            + (f"language {language}<asr_text>" if language else "")
        )
        return {
            "prompt_token_ids": self.encode(prompt),
            "multi_modal_data": {"audio": audio},
        }
