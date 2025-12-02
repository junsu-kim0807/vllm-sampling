# vllm bench throughput

## JSON CLI Arguments

--8<-- "docs/cli/json_tip.inc.md"

## Options

--8<-- "docs/argparse/bench_throughput.md"

## StarkV SuperPress Backend

When `--backend starkv` is selected, `vllm bench throughput` runs the Hugging Face
`kv-press-text-generation` pipeline with StarkV's SuperPress and SuperCache
implementation instead of the vLLM engine. This mode is useful for measuring the
end-to-end latency and throughput impact of StarkV compression strategies.

Key flags:

- `--starkv-score-fn`, `--starkv-compression-ratio`, and `--starkv-confidence-threshold`
  configure the SuperPress scorer and filtering behaviour.
- `--starkv-batch-sizes` lets you evaluate multiple batch sizes sequentially to
  understand throughput scaling (the CLI reports each batch result as well as the
  single-batch latency).
- `--starkv-offload` toggles the SuperCache offload path, while
  `--starkv-device`/`--starkv-max-context-length` forward device and context
  hints to the underlying transformers pipeline.

The JSON output for this backend includes per-batch metrics alongside the overall
requests-per-second and tokens-per-second statistics.