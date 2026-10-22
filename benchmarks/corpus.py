#!/usr/bin/env python3
"""Prepare the seven public datasets from SoundsGood's speed benchmark."""

import argparse
import csv
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf


# https://github.com/SoundsGoodAI/fast-gpu-asr/blob/main/docs/benchmarks/methodology.md#dataset-suite
LEADERBOARD = "hf-audio/open-asr-leaderboard"
SUITE = {
    "ami_cleaned_test": (LEADERBOARD, "ami_cleaned", "test", 7715),
    "earnings22_cleaned_aa_chunked_test": ("ArtificialAnalysis/Earnings22-Cleaned-AA-chunked", "default", "test", 341),
    "gigaspeech_cleaned_test": (LEADERBOARD, "gigaspeech_cleaned", "test", 18757),
    "librispeech_test.clean": (LEADERBOARD, "librispeech", "test.clean", 2620),
    "librispeech_test.other": (LEADERBOARD, "librispeech", "test.other", 2939),
    "spgispeech_test": (LEADERBOARD, "spgispeech", "test", 39341),
    "voxpopuli_cleaned_aa_test": (LEADERBOARD, "voxpopuli_cleaned_aa", "test", 628),
}


def read_audio(source):
    audio, sample_rate = sf.read(source, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    if sample_rate != 16000:
        divisor = math.gcd(sample_rate, 16000)
        audio = resample_poly(audio, 16000 // divisor, sample_rate // divisor)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError(f"Empty or nonfinite audio: {source}")
    return np.ascontiguousarray(audio, dtype=np.float32)


def dataset_rows(root, name):
    """Read upstream-style manifests and sort whole clips by duration, then ID."""
    folder = root / name
    manifest = folder / "references.tsv"
    with manifest.open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError(f"Empty dataset: {name}")
    seen = set()
    for row in rows:
        path = (folder / row["wav_path"]).resolve()
        path.relative_to(folder.resolve())
        row["path"] = path
        row["id"] = row.get("id") or path.stem
        if row["id"] in seen:
            raise ValueError(f"Duplicate utterance ID in {name}: {row['id']}")
        seen.add(row["id"])
        row["duration"] = sf.info(path).duration
        if row["duration"] <= 0:
            raise ValueError(f"Empty audio: {path}")
    return sorted(rows, key=lambda row: (-row["duration"], row["id"]))


def main():
    from datasets import Audio, load_dataset
    from huggingface_hub import HfApi, hf_hub_download

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--datasets", nargs="+", choices=tuple(SUITE), default=list(SUITE))
    parser.add_argument("--leaderboard-revision", default="main")
    parser.add_argument("--earnings-revision", default="main")
    args = parser.parse_args()
    api = HfApi()
    revisions = {repo: api.dataset_info(repo, revision=args.leaderboard_revision if repo == LEADERBOARD
                                       else args.earnings_revision).sha
                 for repo in {SUITE[name][0] for name in args.datasets}}
    for name in args.datasets:
        repo, config, split, expected = SUITE[name]
        folder = args.root / name
        folder.mkdir(parents=True, exist_ok=False)
        (folder / "soundfiles").mkdir()
        dataset = load_dataset(repo, config, split=split, revision=revisions[repo], streaming=True)
        has_audio = "audio" in dataset.features
        if has_audio:
            dataset = dataset.cast_column("audio", Audio(decode=False))
        total_seconds = 0
        count = 0
        with (folder / "references.tsv").open("w") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["id", "wav_path", "original_text"])
            for count, record in enumerate(dataset, 1):
                if has_audio:
                    encoded = record["audio"]
                    source = BytesIO(encoded["bytes"]) if encoded.get("bytes") is not None else encoded["path"]
                else:
                    # Earnings22's JSONL references separate audio files, not an Audio column.
                    source = hf_hub_download(repo, f"audio/{record['file_name']}",
                                             repo_type="dataset", revision=revisions[repo])
                audio = read_audio(source)
                relative = f"soundfiles/{count:06d}.wav"
                sf.write(folder / relative, audio, 16000, subtype="PCM_16")
                writer.writerow([record.get("id", f"{count:06d}"), relative, record.get("text", "")])
                total_seconds += len(audio) / 16000
                if count % 1000 == 0:
                    print(f"{name}: {count}/{expected}", flush=True)
        if count != expected:
            print(f"{name}: public revision has {count} clips; SoundsGood published {expected}. Keeping every clip.", flush=True)
        metadata = {
            "repository": repo, "config": config, "split": split, "revision": revisions[repo],
            "clips": count, "published_clips": expected, "audio_seconds": total_seconds,
            "manifest_sha256": hashlib.sha256((folder / "references.tsv").read_bytes()).hexdigest(),
        }
        (folder / "source.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"{name}: {count} clips, {total_seconds / 3600:.2f} hours", flush=True)


if __name__ == "__main__":
    main()
