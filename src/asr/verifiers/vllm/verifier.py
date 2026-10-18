"""Shared vLLM engine and speculative verification for ASR models."""

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import torch
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from vllm import envs
from vllm.utils.system_utils import set_env_var
from vllm.utils.torch_utils import current_stream

from . import register
from .asr_drafter import bind_drafter
from .draft_buffer import DRAFTS, DraftBuffer


class VLLMVerifier(ABC):
    MODEL_ID = None
    ARCHITECTURE = None
    PROCESSOR_CLASS = AutoProcessor

    def __init__(self, model_id=None, *, k=16, max_model_len=4096,
                 max_num_seqs=1, enable_prefix_caching=True,
                 gpu_memory_utilization=0.7, lookup="suffix"):
        if lookup not in ("suffix", "suffix_gpu"):
            raise ValueError("lookup must be 'suffix' or 'suffix_gpu'")
        self.lookup_name, self.lookup = lookup, None
        model_id = model_id or self.MODEL_ID
        self.max_num_seqs = max_num_seqs
        self.processor = self.PROCESSOR_CLASS.from_pretrained(model_id)
        register()
        # Keep the one-GPU worker in this process: it can share draft buffers
        # and the CUDA context with draft ASR without an IPC handoff.
        envs.disable_envs_cache()
        try:
            with set_env_var("VLLM_ENABLE_V1_MULTIPROCESSING", "0"):
                self.engine = LLM(
                    model=model_id, dtype="bfloat16", max_model_len=max_model_len,
                    max_num_seqs=max_num_seqs, block_size=16,
                    gpu_memory_utilization=gpu_memory_utilization,
                    # The multimodal encoder cache follows this token budget.
                    max_num_batched_tokens=max_model_len,
                    distributed_executor_backend="uni",
                    hf_overrides={"architectures": [self.ARCHITECTURE]},
                    speculative_config={
                        "method": "custom_class",
                        "model": "specspeech.asr.verifiers.vllm.asr_drafter.ASRDrafter",
                        "num_speculative_tokens": k,
                    } if k else None,
                    enable_prefix_caching=enable_prefix_caching,
                    async_scheduling=not k,
                    scheduler_cls="specspeech.asr.verifiers.vllm.scheduler.ASRScheduler" if k else None,
                    per_request_spec_decode_metrics="summary" if k else "none",
                )
        finally:
            # vLLM caches environment values during initialization. Restore
            # the caller's values as well as its environment after setup.
            envs.disable_envs_cache()
            envs.enable_envs_cache()
        if k:
            self.engine.collective_rpc(bind_drafter)
        # Isolate speculative work from TensorRT capture; retain vLLM's stream for async inference.
        initial_stream = current_stream()
        self.stream = torch.cuda.Stream() if k else initial_stream
        self.stream.wait_stream(initial_stream)
        if k and lookup == "suffix_gpu":
            from ...lookups.suffix_gpu import SuffixGPULookup

            with torch.cuda.stream(self.stream):
                self.lookup = SuffixGPULookup(
                    self.processor.tokenizer,
                    max(self.engine.llm_engine.model_config.get_vocab_size(),
                        max(self.processor.tokenizer.get_vocab().values()) + 1), self.stream.device,
                )
        self.input_executor = ThreadPoolExecutor(max_workers=1)

    @abstractmethod
    def prepare(self, audio, language="English"):
        """Build the model prompt and multimodal inputs from a shared waveform."""

    def encode(self, text):
        return self.processor.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, tokens):
        return self.processor.tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    def transcribe(self, samples, drafter=None, *, max_pending=32, max_tokens=512, language="English",
                   punctuator=None):
        """Yield transcripts for (metadata, waveform) pairs, in completion order."""
        from .inference import stream_inference

        return stream_inference(
            self, samples, lambda audio: self.prepare(audio, language),
            max_pending=max_pending, max_tokens=max_tokens, drafter=drafter, punctuator=punctuator,
        )

    def verify(self, prepared, draft_tokens, max_tokens=512):
        """Submit one request; vLLM owns token acceptance and KV rollback."""
        return self.verify_batch([prepared], [draft_tokens], max_tokens)[0]

    def verify_batch(self, prepared, drafts, max_tokens=512):
        """Verify token lists or growing DraftBuffers; serialize calls on this adapter."""
        if len(prepared) != len(drafts):
            raise ValueError("Provide one ASR draft per audio request")
        params, keys = [], []
        for draft in drafts:
            key = uuid4().hex
            keys.append(key)
            DRAFTS[key] = draft if isinstance(draft, DraftBuffer) else DraftBuffer(self.decode, draft, finished=True)
            DRAFTS[key].lookup = self.lookup
            params.append(SamplingParams(temperature=0, max_tokens=max_tokens,
                                         extra_args={"asr_draft_key": key}))
        try:
            with torch.cuda.stream(self.stream):
                outputs = self.engine.generate(prepared, params, use_tqdm=False)
        finally:
            for key in keys:
                DRAFTS.pop(key, None)
        return [output.outputs[0] for output in outputs]

    def warmup(self, prepared):
        self.verify(prepared, [], max_tokens=1)

    def close(self):
        self.input_executor.shutdown(wait=True, cancel_futures=True)
        self.engine.llm_engine.engine_core.shutdown()
