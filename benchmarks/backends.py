"""HF batches and native vLLM continuous inference backends."""

from itertools import islice

import torch


HF_MODEL = "Qwen/Qwen3-ASR-1.7B-hf"


def make_backend(args, cleanup):
    """Load engines once and return (run_inference, settings).

    run_inference(samples) yields (metadata, transcript) pairs as it is iterated.
    """
    if args.backend in ("hf", "hf-dynamic"):
        from transformers import AutoModelForMultimodalLM, AutoProcessor, CompileConfig

        processor = AutoProcessor.from_pretrained(HF_MODEL)
        model = AutoModelForMultimodalLM.from_pretrained(
            HF_MODEL, dtype=torch.bfloat16, attn_implementation="sdpa",
        ).to("cuda").eval()
        compile_config = CompileConfig(mode="reduce-overhead", fullgraph=True)
        if args.backend == "hf-dynamic":
            model.forward = torch.compile(model.forward)

        def infer_batch(audios):
            inputs = processor.apply_transcription_request(
                audio=audios, language="English", prompt="", sampling_rate=16000,
            ).to("cuda", torch.bfloat16)
            cache_kwargs = {"cache_implementation": "static", "compile_config": compile_config} if args.backend == "hf" else {}
            sequences = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                **cache_kwargs,
            )[:, inputs["input_ids"].shape[1]:].tolist()
            eos = model.generation_config.eos_token_id
            eos = set([eos] if isinstance(eos, int) else eos or [])
            results = []
            for ids in sequences:
                end = next((index + 1 for index, token in enumerate(ids) if token in eos), len(ids))
                ids = ids[:end]
                results.append({
                    "text": processor.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False),
                    "tokens": len(ids),
                    "finish_reason": "stop" if ids and ids[-1] in eos else "length",
                })
            return results

        def run(samples):
            samples = iter(samples)
            while batch := list(islice(samples, args.batch_size)):
                outputs = infer_batch([audio for _, audio in batch])
                if len(outputs) != len(batch):
                    raise RuntimeError("Backend did not return one output per clip")
                yield from zip((sample for sample, _ in batch), outputs)

        return run, {"model": HF_MODEL, "attention": "sdpa",
                     "compile_mode": "reduce-overhead" if args.backend == "hf" else "default",
                     "cache": "static" if args.backend == "hf" else "dynamic"}

    from specspeech.asr.speculators import SoundsGoodDrafter
    from specspeech.asr.verifiers.vllm import QwenASRVerifier

    drafter = SoundsGoodDrafter(
        args.drafter_engine, model=args.drafter, drafter_max_audio_seconds=args.drafter_max_audio_seconds,
    ) if args.backend == "speculative" else None
    verifier = QwenASRVerifier(
        k=args.k if drafter else 0, max_model_len=args.max_model_len,
        enable_prefix_caching=False, max_num_seqs=args.batch_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        lookup=args.lookup,
    )
    cleanup.callback(verifier.close)

    def stream(samples):
        return verifier.transcribe(
            samples, drafter, max_pending=args.max_pending,
            max_tokens=args.max_new_tokens,
        )

    return stream, {"model": verifier.MODEL_ID, "implementation": "vllm.model_executor.models.qwen3_asr",
                    "cuda_graphs_enabled": True, "speculative_tokens": args.k if drafter else 0,
                    "gpu_memory_utilization": args.gpu_memory_utilization,
                    "drafter": args.drafter if drafter else None,
                    "drafter_engine": str(drafter.engine_path) if drafter else None,
                    "drafter_batch_size": drafter.batch_size if drafter else None,
                    "drafter_bucket_seconds": 0.1 if drafter else None,
                    "lookup": verifier.lookup_name if drafter else None,
                    "draft_readiness": "incremental_with_request_pause" if drafter else None}
