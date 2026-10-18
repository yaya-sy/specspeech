"""Command line entry point for speculative speech inference."""

import argparse


def main():
    parser = argparse.ArgumentParser(prog="specspeech")
    commands = parser.add_subparsers(dest="command", required=True)
    asr = commands.add_parser("asr", help="Transcribe audio with speculative decoding")
    asr.add_argument("audio", nargs="+", help="Audio files or folders (searched recursively)")
    asr.add_argument("--drafter", choices=("parakeet", "zipformer"), default="parakeet")
    asr.add_argument("--drafter-engine")
    asr.add_argument("--drafter-max-audio-seconds", type=int,
                     help="Draft piece limit (default: Parakeet 300, Zipformer 40)")
    asr.add_argument("--verifier-max-audio-seconds", type=int, default=1200)
    asr.add_argument("--k", type=int, default=32)
    asr.add_argument("--lookup", choices=("suffix", "suffix_gpu"), default="suffix")
    asr.add_argument("--batch-size", type=int, default=8, help="Maximum active verifier requests")
    asr.add_argument("--max-pending", type=int, default=32)
    asr.add_argument("--max-model-len", type=int, default=32768)
    asr.add_argument("--max-new-tokens", type=int, default=8192)
    asr.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    asr.add_argument("--language", default="English")
    asr.add_argument("--punctuator", help="Optional PCS ONNX model path")
    args = parser.parse_args()
    if min(args.verifier_max_audio_seconds, args.batch_size, args.max_pending,
           args.max_model_len, args.max_new_tokens) <= 0 or args.k < 0:
        parser.error("Durations, batch size and token limits must be positive; k may be zero")
    if args.drafter_max_audio_seconds is not None and args.drafter_max_audio_seconds <= 0:
        parser.error("--drafter-max-audio-seconds must be positive")

    from .asr.audio import audio_segments
    from .asr.punctuator import load_punctuator
    from .asr.speculators import SoundsGoodDrafter
    from .asr.verifiers.vllm import QwenASRVerifier

    drafter = SoundsGoodDrafter(
        model=args.drafter, engine_path=args.drafter_engine,
        drafter_max_audio_seconds=args.drafter_max_audio_seconds,
    ) if args.k else None
    punctuator = load_punctuator(args.punctuator) if args.punctuator else None
    verifier = QwenASRVerifier(
        k=args.k, lookup=args.lookup, max_model_len=args.max_model_len,
        max_num_seqs=args.batch_size, gpu_memory_utilization=args.gpu_memory_utilization,
    )
    results = verifier.transcribe(
        audio_segments(args.audio, verifier_max_audio_seconds=args.verifier_max_audio_seconds),
        drafter, max_pending=args.max_pending, max_tokens=args.max_new_tokens,
        language=args.language, punctuator=punctuator,
    )
    try:
        for sample, result in results:
            print(f"{sample['path']} [segment {sample['segment_index']}]", flush=True)
            print(result["text"], flush=True)
            print(f"Output tokens: {result['tokens']} | Accepted draft tokens: {result['accepted']} "
                  f"| Proposed tokens: {result['proposed']}", flush=True)
    finally:
        results.close()
        verifier.close()
