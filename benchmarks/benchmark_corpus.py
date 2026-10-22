#!/usr/bin/env python3
"""Measure pooled ASR throughput on SoundsGood's seven-dataset English suite."""

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import hashlib
from importlib.metadata import PackageNotFoundError, version
from itertools import islice
import json
from pathlib import Path
import time

import torch

from backends import make_backend
from corpus import SUITE, dataset_rows, read_audio


def prefetch_samples(rows, capacity):
    """Read and hash a bounded number of waveforms ahead of inference."""
    def load(row):
        audio = read_audio(row["path"])
        return {**row, "audio_sha256": hashlib.sha256(audio.tobytes()).hexdigest(),
                "audio_seconds": len(audio) / 16000}, audio

    rows = iter(rows)
    with ThreadPoolExecutor(max_workers=1) as loader:
        pending = deque(loader.submit(load, row) for row in islice(rows, capacity))
        while pending:
            yield pending.popleft().result()
            if (row := next(rows, None)) is not None:
                pending.append(loader.submit(load, row))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets_root", type=Path)
    parser.add_argument("--backend", choices=("hf", "hf-dynamic", "vllm-official", "speculative"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for this condition")
    parser.add_argument("--datasets", nargs="+", default=list(SUITE),
                        help="Dataset directories containing references.tsv; defaults to the seven-dataset suite")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Maximum active vLLM requests, or HF inference batch size (default: 8)")
    parser.add_argument("--max-pending", type=int,
                        help="Maximum pending vLLM requests and waveform prefetch count (default: 4 * batch-size)")
    parser.add_argument("--drafter", choices=("parakeet", "zipformer"), default="zipformer")
    parser.add_argument("--drafter-engine")
    parser.add_argument("--drafter-max-audio-seconds", type=int,
                        help="Maximum draft piece duration (default: Parakeet 300, Zipformer 40)")
    parser.add_argument("--verifier-max-audio-seconds", type=int, default=1200)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--lookup", choices=("suffix", "suffix_gpu"), default="suffix")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7,
                        help="GPU memory fraction budgeted for either vLLM backend (default: 0.7); unused by HF")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--warmups", type=int, default=5)
    args = parser.parse_args()
    if args.drafter_max_audio_seconds is None:
        args.drafter_max_audio_seconds = 300 if args.drafter == "parakeet" else 40
    if args.max_pending is None:
        args.max_pending = 4 * args.batch_size
    if min(args.batch_size, args.drafter_max_audio_seconds, args.verifier_max_audio_seconds, args.k,
           args.max_model_len, args.max_new_tokens, args.warmups, args.max_pending) <= 0:
        parser.error("Batch size, durations and token limits must be positive")
    if args.max_pending < args.batch_size:
        parser.error("--max-pending must be at least --batch-size")
    if len(set(args.datasets)) != len(args.datasets):
        parser.error("Each dataset must occur once")
    if not torch.cuda.is_available():
        parser.error("An NVIDIA CUDA GPU is required")
    # Validate the full selection before loading models or collecting timings.
    datasets = {name: dataset_rows(args.datasets_root, name) for name in args.datasets}
    if any(row["duration"] > args.verifier_max_audio_seconds for rows in datasets.values() for row in rows):
        parser.error("A corpus clip exceeds the verifier duration limit; refusing to change corpus segmentation")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    batches = [rows[start:start + args.batch_size] for rows in datasets.values()
               for start in range(0, len(rows), args.batch_size)]
    warmup_batches = sorted(batches, key=lambda batch: max(row["duration"] for row in batch))

    def load(batch):
        return [read_audio(row["path"]) for row in batch]

    with ExitStack() as cleanup:
        run_inference, settings = make_backend(args, cleanup)
        cleanup.enter_context(torch.inference_mode())
        # Real batches spanning the duration range, excluded from timing.
        for index in range(args.warmups):
            batch = warmup_batches[round(index * (len(warmup_batches) - 1) / max(1, args.warmups - 1))]
            audios = load(batch)
            list(run_inference(zip(batch, audios)))
            torch.cuda.synchronize()
            print(f"Warmup {index + 1}/{args.warmups} complete", flush=True)
        dataset_reports = {}
        with (args.output_dir / "transcripts.jsonl").open("w") as transcripts, \
             (args.output_dir / "batches.jsonl").open("w") as timings:
            for name, rows in datasets.items():
                seconds = audio_seconds = 0
                truncated = 0
                fingerprints = {}

                def save(row, digest, output):
                    nonlocal truncated
                    fingerprints[row["id"]] = digest
                    truncated += output.get("finish_reason") == "length"
                    if output.get("finish_reason") == "length":
                        print(f"{name}/{row['id']}: generation limit reached: {output['tokens']} tokens "
                              f"for {row['audio_seconds']:.2f}s audio", flush=True)
                    transcripts.write(json.dumps({"dataset": name, "id": row["id"],
                                                  "audio_sha256": digest, **output}) + "\n")

                torch.cuda.synchronize()
                started = time.perf_counter()
                samples = prefetch_samples(rows, args.max_pending)
                results = run_inference(samples)
                try:
                    # Iteration executes inference and yields each completed request.
                    for completed, (row, output) in enumerate(results, 1):
                        audio_seconds += row["audio_seconds"]
                        save(row, row["audio_sha256"], output)
                        if completed == 1 or completed % (100 * args.batch_size) == 0:
                            elapsed = time.perf_counter() - started
                            print(f"{name}: {completed}/{len(rows)}; {audio_seconds / elapsed:.1f} RTFx", flush=True)
                finally:
                    results.close()
                    samples.close()
                torch.cuda.synchronize()
                seconds = time.perf_counter() - started
                timings.write(json.dumps({"dataset": name, "clips": len(rows),
                                          "audio_seconds": audio_seconds, "seconds": seconds}) + "\n")
                print(f"{name}: {len(rows)}/{len(rows)}; {audio_seconds / seconds:.1f} RTFx", flush=True)
                if len(fingerprints) != len(rows):
                    raise RuntimeError("Backend did not return one output per clip")
                fingerprint = hashlib.sha256()
                for row in rows:
                    fingerprint.update(json.dumps([row["id"], fingerprints[row["id"]]]).encode())
                source = args.datasets_root / name / "source.json"
                dataset_reports[name] = {
                    "clips": len(rows), "audio_seconds": audio_seconds, "seconds": seconds,
                    "rtfx": audio_seconds / seconds, "audio_sha256": fingerprint.hexdigest(),
                    "length_limited_outputs": truncated,
                    "source": json.loads(source.read_text()) if source.exists() else None,
                }
        versions = {}
        for package in ("torch", "transformers", "vllm", "qwen-asr", "specspeech-soundsgood"):
            try:
                versions[package] = version(package)
            except PackageNotFoundError:
                pass
        total_seconds = sum(report["seconds"] for report in dataset_reports.values())
        total_audio = sum(report["audio_seconds"] for report in dataset_reports.values())
        report = {
            "backend": args.backend, "settings": settings, "batch_size": args.batch_size,
            "max_pending": args.max_pending,
            "dtype": "bfloat16", "temperature": 0, "language": "English",
            "gpu": torch.cuda.get_device_name(), "versions": versions,
            "torch_cuda": torch.version.cuda,
            "max_model_len": args.max_model_len, "max_new_tokens": args.max_new_tokens,
            "drafter_max_audio_seconds": args.drafter_max_audio_seconds if args.backend == "speculative" else None,
            "verifier_max_audio_seconds": args.verifier_max_audio_seconds,
            "warmups": args.warmups, "passes": 1, "complete_suite": set(datasets) == set(SUITE),
            "matches_published_clip_counts": set(datasets) == set(SUITE)
            and all(len(rows) == SUITE[name][3] for name, rows in datasets.items()),
            "timing_scope": "synchronized dataset wall time with bounded waveform prefetch; includes loading, resampling, hashing, features, drafting and serialization",
            "datasets": dataset_reports, "clips": sum(item["clips"] for item in dataset_reports.values()),
            "audio_seconds": total_audio, "seconds": total_seconds, "rtfx": total_audio / total_seconds,
        }
        (args.output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"{args.backend}: {report['clips']} clips, {report['rtfx']:.2f} pooled RTFx", flush=True)


if __name__ == "__main__":
    main()
