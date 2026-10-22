#!/usr/bin/env python3
"""Compare pooled corpus throughput and speculative speedup."""

import argparse
import json
from pathlib import Path


def summarize_corpus(reports, root):
    reference = next(iter(reports.values()))
    for report in reports.values():
        if report["timing_scope"] != reference["timing_scope"]:
            raise ValueError("Conditions must use the same timing scope")
        if report.get("max_pending", report.get("request_queue_size")) != reference.get("max_pending", reference.get("request_queue_size")):
            raise ValueError("Conditions must use the same max_pending limit")
        if (report["batch_size"], report["max_model_len"], report["max_new_tokens"], report["gpu"]) != (
            reference["batch_size"], reference["max_model_len"], reference["max_new_tokens"], reference["gpu"]
        ):
            raise ValueError("Corpus conditions must use the same batch, token limits and GPU model")
        if {name: item["audio_sha256"] for name, item in report["datasets"].items()} != {
            name: item["audio_sha256"] for name, item in reference["datasets"].items()
        }:
            raise ValueError("Corpus conditions used different clips or audio")
        if any(item["length_limited_outputs"] for item in report["datasets"].values()):
            raise ValueError("Length-limited transcripts detected; increase --max-new-tokens and rerun all conditions")
    speculative = reports["speculative"]
    results = {name: {"seconds": report["seconds"], "rtfx": report["rtfx"],
                      "speculative_speedup": report["seconds"] / speculative["seconds"]}
               for name, report in reports.items()}
    hf_backends = [name for name in reports if name.startswith("hf")]
    best_hf = min(hf_backends, key=lambda name: reports[name]["seconds"]) if hf_backends else None
    summary = {"complete_suite": reference["complete_suite"], "clips": reference["clips"],
               "matches_published_clip_counts": reference.get("matches_published_clip_counts", False),
               "audio_hours": reference["audio_seconds"] / 3600,
               "timing_scope": reference["timing_scope"],
               "max_pending": reference.get("max_pending", reference.get("request_queue_size")),
               "batch_size": reference["batch_size"], "best_hf": best_hf, "results": results}
    for name, item in results.items():
        print(f"{name:16s} {item['rtfx']:9.2f} RTFx  speculative speedup {item['speculative_speedup']:.2f}x")
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    args = parser.parse_args()
    root = Path(args.directory)
    reports = {p.parent.name: json.loads(p.read_text()) for p in root.glob("*/results.json")}
    if not reports:
        parser.error("No benchmark results found")
    summarize_corpus(reports, root)


if __name__ == "__main__":
    main()
