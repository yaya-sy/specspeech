# ASR speed benchmark

```bash
run_dir="results/parakeet-$(date +%Y%m%d-%H%M%S)"
for backend in speculative vllm-official; do
  .venv/bin/python -u benchmarks/benchmark_corpus.py data/asr-benchmark \
    --backend "$backend" --drafter parakeet --drafter-engine exported/parakeet-custom \
    --k 32 --lookup suffix_gpu \
    --drafter-max-audio-seconds 300 --verifier-max-audio-seconds 1200 \
    --max-num-seqs 8 --max-pending 32 --gpu-memory-utilization 0.7 \
    --max-model-len 32768 --max-new-tokens 8192 --warmups 5 \
    --output-dir "$run_dir/$backend" || break
done
python benchmarks/summarize_benchmarks.py "$run_dir"
```

With `--max-num-seqs 1`, both backends run speech inference on one audio at a time,
with one pending inference request. `--audio-prefetch 32` independently loads,
downmixes, resamples and hashes upcoming audio on CPU for both conditions;
use `--audio-prefetch 0` to disable it. Feature extraction and speech-model work
remain inside the timed inference path, and the dataset timer includes loader startup/drain.
The SoundsGood drafter keeps its exported engine's batch capacity (currently 8),
batching only pieces of the current audio alongside verifier prefill/verification.

## Results

Measured on an NVIDIA A40 with Qwen3-ASR-1.7B, Parakeet TDT v3 through
SoundsGood, vLLM 0.30.0 and PyTorch 2.13.0. Both vLLM conditions use CUDA graphs,
BF16 and greedy decoding. Settings: `k=32`, `suffix_gpu`, drafter engine capacity 8,
300-second drafter pieces, 1,200-second verifier segments, `gpu_memory_utilization=0.7`,
`max_model_len=32768`, `max_new_tokens=8192` and five warmups.

Each run covers the same 72,442 clips (157.81 audio hours); audio hashes match
between conditions. RTFx is audio duration divided by processing time, so higher
is faster. The pooled row uses total audio duration divided by total processing
time. Timings include loading, preprocessing, inference and serialization.

### Lookup window widening

Two speculative runs with `--max-num-seqs 8 --max-pending 32`, comparing
`--lookup-window-width 8` against `--lookup-window-width 8 16 32`.
Run: `lookup-windows-8-vs-8-16-32-20261025`.

| Dataset | Fixed 8 RTFx ↑ | Widening 8 → 16 → 32 RTFx ↑ | Widening / fixed |
|---|---:|---:|---:|
| AMI Cleaned† | 117.21 | 108.84 | 0.93× |
| Earnings22 Cleaned (AA chunked) | 377.86 | 329.65 | 0.87× |
| GigaSpeech Cleaned | 261.32 | 270.57 | 1.04× |
| LibriSpeech test-clean | 332.49 | 317.24 | 0.95× |
| LibriSpeech test-other | 279.85 | 282.13 | 1.01× |
| SPGISpeech | 380.28 | 391.70 | 1.03× |
| VoxPopuli Cleaned (AA) | 405.62 | 429.49 | 1.06× |
| **Full suite (pooled)†** | **308.73** | **312.34** | **1.012×** |

Widening improved pooled throughput by **1.17%**, saving 21.27 seconds across
the suite. These are single runs; they do not establish statistical significance.
† One AMI output reached the 8,192-token generation limit in each condition.

### Single-request inference

Both backends use `--max-num-seqs 1 --max-pending 1 --audio-prefetch 32`.
Only the current audio enters speech inference; CPU waveform preparation can
run ahead. The drafter can batch up to eight pieces of that audio, with fixed
lookup window 8. Run: `single-audio-prefetch32-bs1-20261025`.

| Dataset | Native vLLM RTFx ↑ | SpecSpeech RTFx ↑ | Speedup |
|---|---:|---:|---:|
| AMI Cleaned† | 24.23 | 44.24 | 1.83× |
| Earnings22 Cleaned (AA chunked) | 43.23 | 82.07 | 1.90× |
| GigaSpeech Cleaned | 28.01 | 67.65 | 2.42× |
| LibriSpeech test-clean | 31.29 | 82.12 | 2.62× |
| LibriSpeech test-other | 29.83 | 69.03 | 2.31× |
| SPGISpeech | 34.54 | 90.76 | 2.63× |
| VoxPopuli Cleaned (AA) | 38.54 | 94.22 | 2.44× |
| **Full suite (pooled)†** | **32.02** | **79.30** | **2.48×** |

† One native vLLM AMI output reached the 8,192-token generation limit;
no speculative outputs reached it. The table reports observed throughput,
but AMI and pooled ratios include that incomplete output. The summary script
rejected this comparison for that reason; a comparison without capped outputs
requires rerunning both conditions with a higher generation limit.
