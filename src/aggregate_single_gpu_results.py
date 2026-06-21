from __future__ import annotations

import argparse
import json
from pathlib import Path

"""
$RUN --precision fp32 --compile false  --ckpt none
$RUN --is_custom true --precision fp32 --compile false  --ckpt none
$RUN --is_custom true --precision fp32 --compile true  --grad_checkpoint_strat none
$RUN --precision fp32 --compile true  --ckpt none
$RUN --precision fp16 --compile false --ckpt none
$RUN --precision fp16 --compile true  --ckpt none
$RUN --precision bf16 --compile false --ckpt none   
$RUN --precision fp32 --compile false --ckpt full
$RUN --precision fp16 --compile false --ckpt full
$RUN --precision fp16 --compile true  --ckpt full
$RUN --precision fp16 --compile false --ckpt selective
$RUN --precision fp16 --compile true  --ckpt selective
"""

ROWS: list[tuple[str, dict]] = [
    ("baseline fp32 with custom layers",
        {"is_custom": True, "precision": "fp32", "compile": False, "grad_checkpoint_strat": "none", "attention_impl": "eager"}),
        ("baseline fp32 with custom layers",
        {"is_custom": True, "precision": "fp32", "compile": True, "grad_checkpoint_strat": "none", "attention_impl": "eager"}),
    ("baseline fp32",
        {"is_custom": False, "precision": "fp32", "compile": False, "grad_checkpoint_strat": "none"}),
    ("fp32 compiled",
        {"is_custom": False, "precision": "fp32", "compile": True,  "grad_checkpoint_strat": "none"}),
    ("baseline fp16",
        {"precision": "fp16", "compile": False, "grad_checkpoint_strat": "none"}),
    ("fp16 compiled",
        {"precision": "fp16", "compile": True,  "grad_checkpoint_strat": "none"}),
    ("baseline bf16 (not native on T4)",
        {"precision": "bf16", "compile": False, "grad_checkpoint_strat": "none"}),
    ("baseline fp32 with full checkpointing",
        {"precision": "fp32", "compile": False, "grad_checkpoint_strat": "full"}),
    ("fp16 with full checkpointing",
        {"precision": "fp16", "compile": False, "grad_checkpoint_strat": "full"}),
    ("compiled fp16 with full checkpointing",
        {"precision": "fp16", "compile": True,  "grad_checkpoint_strat": "full"}),
    ("fp16 with SELECTIVE checkpointing",
        {"precision": "fp16", "compile": False,  "grad_checkpoint_strat": "selective"}),
    ("compiled fp16 with SELECTIVE checkpointing (attn block megatron vibe)",
        {"precision": "fp16", "compile": True,  "grad_checkpoint_strat": "selective"}),
]


def _matches(rec: dict, spec: dict) -> bool:
    return all(rec.get(k) == v for k, v in spec.items())


def _fmt(v) -> str:
    if v is None:
        return "OOM / —"
    return str(v)


def _fmt_ms_with_std(rec: dict, mean_key: str, std_key: str) -> str:
    m = rec.get(mean_key)
    s = rec.get(std_key)
    if m is None:
        return "OOM / —"
    if s in (None, 0.0):
        return f"{m}"
    return f"{m} ± {s}"


def make_table(results_dir: Path) -> str:
    records = [json.loads(p.read_text()) for p in results_dir.glob("*.json")]

    lines = [
        "| config | forward ms | backward ms | peak GB | tokens/s |",
        "|---|---|---|---|---|",
    ]
    for label, spec in ROWS:
        match = next((r for r in records if _matches(r, spec)), None)
        if match is None:
            lines.append(f"| {label} | — | — | — | — |")
            continue
        lines.append(
            f"| {label} "
            f"| {_fmt_ms_with_std(match, 'fwd_ms', 'fwd_std')} "
            f"| {_fmt_ms_with_std(match, 'bwd_ms', 'bwd_std')} "
            f"| {_fmt(match.get('peak_gb'))} "
            f"| {_fmt(match.get('tok_s'))} |"
        )
    return "\n".join(lines)


def main():
    result_dir = "results/single_gpu_sweep"
    filename = "aggregate_results_table.md"
    table = make_table(Path(result_dir))
    print(table)
    Path(f"{result_dir}/{filename}").write_text(table + "\n")


if __name__ == "__main__":
    main()