# ASR speed benchmark

```bash
run_dir="results/parakeet-$(date +%Y%m%d-%H%M%S)"
for backend in speculative vllm-official; do
  .venv/bin/python -u benchmarks/benchmark_corpus.py data/asr-benchmark \
    --backend "$backend" --drafter parakeet --k 32 --lookup suffix_gpu \
    --drafter-max-audio-seconds 300 --verifier-max-audio-seconds 1200 \
    --batch-size 8 --max-pending 32 --gpu-memory-utilization 0.7 \
    --max-model-len 32768 --max-new-tokens 8192 --warmups 5 \
    --output-dir "$run_dir/$backend" || break
done
python benchmarks/summarize_benchmarks.py "$run_dir"
```
