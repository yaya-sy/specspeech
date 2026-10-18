#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/export.sh [parakeet|zipformer] [checkpoint_dir] [engine_dir]
family="${1:-parakeet}"
checkpoint_dir="${2:-checkpoints/$family}"
case "$family" in
  parakeet)
    model="parakeet-tdt-0.6b-v3"
    base="https://huggingface.co/nvidia/$model/resolve/main"
    files=("$model.nemo")
    model_path="$checkpoint_dir/$model.nemo"
    engine_dir="${3:-exported/$model}"
    exporter="fast-gpu-asr-export-parakeet"
    default_max_audio_seconds=300
    ;;
  zipformer)
    base="https://huggingface.co/soundsgoodai/Zipformer-cr-ctc-transducer-XL-290M/resolve/main"
    files=(model.pt config.yaml bpe.model)
    model_path="$checkpoint_dir/model.pt"
    engine_dir="${3:-exported/zipformer}"
    exporter="fast-gpu-asr-export-zipformer"
    default_max_audio_seconds=40
    ;;
  *) echo "Unknown drafter: $family (choose parakeet or zipformer)" >&2; exit 1 ;;
esac

mkdir -p "$checkpoint_dir"
for file in "${files[@]}"; do
  if [[ ! -s "$checkpoint_dir/$file" ]]; then
    curl --fail --location "$base/$file" --output "$checkpoint_dir/$file.partial"
    mv "$checkpoint_dir/$file.partial" "$checkpoint_dir/$file"
  fi
done

# Parakeet uses full attention with five-minute pieces; Zipformer uses 40 seconds.
# The exporter replaces engine_dir; keep it reserved for generated engines.
"$exporter" \
  --model-path "$model_path" \
  --output-dir "$engine_dir" \
  --batch-size "${BATCH_SIZE:-8}" --decoder-type transducer_modified_beam_search --beam 6 \
  --encoder-precision fp16 --decoder-precision fp16 \
  --min-audio-seconds 0.1 --opt-audio-seconds 8 \
  --max-audio-seconds "${MAX_AUDIO_SECONDS:-$default_max_audio_seconds}" \
  --optimization-level 5
