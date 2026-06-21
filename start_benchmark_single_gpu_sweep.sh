#!/usr/bin/env bash
set -e
RUN="uv run python -m src.bench_single_gpu --config_path configs/baseline_single_gpu_config.yaml"

$RUN --precision fp32 --compile false  --grad_checkpoint_strat none
$RUN --is_custom true --precision fp32 --compile false  --grad_checkpoint_strat none
$RUN --is_custom true --precision fp32 --compile true  --grad_checkpoint_strat none
$RUN --precision fp32 --compile true  --grad_checkpoint_strat none
$RUN --precision fp16 --compile false --grad_checkpoint_strat none
$RUN --precision fp16 --compile true  --grad_checkpoint_strat none
$RUN --precision bf16 --compile false --grad_checkpoint_strat none   
$RUN --precision fp32 --compile false --grad_checkpoint_strat full
$RUN --precision fp16 --compile false --grad_checkpoint_strat full
$RUN --precision fp16 --compile true  --grad_checkpoint_strat full
$RUN --precision fp16 --compile false --grad_checkpoint_strat selective
$RUN --precision fp16 --compile true  --grad_checkpoint_strat selective

uv run python -m src.aggregate_single_gpu_results


# uv run python -m src.run_profiler --precision fp16 --compile false --grad_checkpoint_strat none