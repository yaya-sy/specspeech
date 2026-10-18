"""Local ASR adapters registered with the installed vLLM package."""

from importlib import import_module


ARCHITECTURES = {}
_EXPORTS = {
    "VLLMVerifier": ".verifier",
    "QwenASRVerifier": ".qwen_asr_verifier",
    "Qwen3ASRForConditionalGeneration": "vllm.model_executor.models.qwen3_asr",
}
__all__ = [*_EXPORTS, "register"]


def register():
    """Register local model implementations in vLLM engine/worker processes."""
    from vllm import ModelRegistry

    for architecture, model in ARCHITECTURES.items():
        ModelRegistry.register_model(architecture, model)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
