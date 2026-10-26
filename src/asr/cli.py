"""Command line entry point for speculative ASR inference."""

import argparse
import sys


def add_asr_arguments(asr):
    asr.add_argument("--drafter", choices=("parakeet", "zipformer", "granite"), default="parakeet")
    asr.add_argument("--drafter-engine", help="SoundsGood engine directory (Parakeet or Zipformer)")
    asr.add_argument("--drafter-max-audio-seconds", type=int,
                     help="Draft piece limit (default: Parakeet 300, Zipformer/Granite 40)")
    asr.add_argument("--verifier-max-audio-seconds", type=int, default=1200)
    asr.add_argument("--k", type=int, default=32)
    asr.add_argument("--lookup", choices=("suffix", "suffix_gpu"), default="suffix")
    asr.add_argument("--lookup-window-width", type=int, nargs="+", default=[8], metavar="N",
                     help="Tokens on each side; try increasing widths after a miss (default: 8)")
    asr.add_argument("--max-num-seqs", type=int,
                     help="Maximum verifier sequences per scheduling iteration (default: chosen by vLLM)")
    asr.add_argument("--batch-size", dest="max_num_seqs", type=int,
                     default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    asr.add_argument("--max-pending", type=int, default=32)
    asr.add_argument("--max-model-len", type=int, default=32768)
    asr.add_argument("--max-new-tokens", type=int, default=8192)
    asr.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    asr.add_argument("--language", default="English")
    asr.add_argument("--punctuator", help="Optional PCS ONNX model path")


def validate_arguments(parser, args):
    if args.max_num_seqs is not None and args.max_num_seqs <= 0:
        parser.error("--max-num-seqs must be positive")
    if min(args.verifier_max_audio_seconds, args.max_pending,
           args.max_model_len, args.max_new_tokens) <= 0 or args.k < 0:
        parser.error("Durations and token limits must be positive; k may be zero")
    if args.drafter_max_audio_seconds is not None and args.drafter_max_audio_seconds <= 0:
        parser.error("--drafter-max-audio-seconds must be positive")
    if args.drafter == "granite" and args.drafter_engine:
        parser.error("--drafter-engine applies only to Parakeet and Zipformer")
    if min(args.lookup_window_width) <= 0 or any(left >= right for left, right in
                                                zip(args.lookup_window_width, args.lookup_window_width[1:])):
        parser.error("--lookup-window-width requires positive, strictly increasing integers")


def create_engines(args, **engine_options):
    """Load the shared engines once for file inference or HTTP serving."""
    from .punctuator import load_punctuator
    from .speculators import GraniteDrafter, SoundsGoodDrafter
    from .verifiers.vllm import QwenASRVerifier

    drafter = None
    if args.k:
        if args.drafter == "granite":
            drafter = GraniteDrafter(
                drafter_max_audio_seconds=args.drafter_max_audio_seconds or 40,
            )
        else:
            drafter = SoundsGoodDrafter(
                model=args.drafter, engine_path=args.drafter_engine,
                drafter_max_audio_seconds=args.drafter_max_audio_seconds,
            )
    punctuator = load_punctuator(args.punctuator) if args.punctuator else None
    verifier = QwenASRVerifier(
        k=args.k, lookup=args.lookup, max_model_len=args.max_model_len,
        lookup_window_width=args.lookup_window_width,
        max_num_seqs=args.max_num_seqs, gpu_memory_utilization=args.gpu_memory_utilization,
        **engine_options,
    )
    return verifier, drafter, punctuator


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:2] == ["asr", "endpoint"]:
        from .endpoint import main as endpoint_main

        return endpoint_main(argv[2:])
    parser = argparse.ArgumentParser(prog="specspeech")
    commands = parser.add_subparsers(dest="command", required=True)
    asr = commands.add_parser("asr", help="Transcribe files, or use 'asr endpoint' to serve HTTP")
    asr.add_argument("audio", nargs="+", help="Audio files or folders (searched recursively)")
    add_asr_arguments(asr)
    args = parser.parse_args(argv)
    validate_arguments(parser, args)

    from .audio import audio_segments

    verifier, drafter, punctuator = create_engines(args)
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
