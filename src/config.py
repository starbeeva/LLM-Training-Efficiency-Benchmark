from dataclasses import dataclass, field
from typing import List, Optional

import torch


@dataclass
class BenchConfig:
    """Single-GPU bench config, used to compare mp/gradient checkpointing/torch.compile()"""

    # model params
    num_layers: int = 12
    vocab_size: int = 50_304 # ближайшее % 64 == 0 к ориг числу
    d_model: int = 768
    num_heads: int = 12
    d_ff: int = 2_048
    max_seq_len: int = 1024
    theta: float = 10_000.0
    is_custom: bool = False
    attention_impl: str = "eager"

    batch_size: int = 2
    seq_len: int = 1024

    precision: str = "fp32"
    compile: bool = False
    grad_checkpoint_strat: str = "none"

    warmup_iters: int = 5
    measure_iters: int = 20
    seed: int = 0

    run_name: Optional[str] = None
    result_dir: str = "results/single_gpu_sweep"

    @property
    def autocast_dtype(self) -> torch.dtype:
        return {"fp32": torch.float32,
                "fp16": torch.float16,
                "bf16": torch.bfloat16}[self.precision]

    @property
    def use_autocast(self) -> bool:
        return self.precision != "fp32"

    @property
    def use_scaler(self) -> bool:
        return self.precision == "fp16" or self.precision == "bf16"

    def __post_init__(self):
        if self.run_name is None:
            run_name_parts = [self.precision]
            if self.compile:
                run_name_parts.append("compiled")
            if self.grad_checkpoint_strat != "none":
                run_name_parts.append(f"ckpt-{self.grad_checkpoint_strat}")
            run_name_parts.append(f"attn-{self.attention_impl}")
            if self.is_custom:
                run_name_parts.append("customlayers")
            self.run_name = "_".join(run_name_parts)


@dataclass
class AttnSweepConfig:
    """Sweep seq_len for different attention implementations at batch_size 1"""
    # model params
    num_layers: int = 12
    vocab_size: int = 50_304 # ближайшее % 64 == 0 к ориг числу
    d_model: int = 768
    num_heads: int = 12
    d_ff: int = 2_048
    max_seq_len: int = 1024
    theta: float = 10_000.0
    is_custom: bool = False
    attention_impl: str = "eager"

    seq_lens: List[int] = field(
        default_factory=lambda: [512, 2048, 4096, 8192]
    )
    seq_len: int = 1024
    compile: bool = False
    batch_size: int = 1
    precision: str = "fp16"
    warmup_iters: int = 5
    measure_iters: int = 10
    seed: int = 0
    result_dir: str = "results/attn_sweep"

    @property
    def autocast_dtype(self) -> torch.dtype:
        return {"fp32": torch.float32,
                "fp16": torch.float16,
                "bf16": torch.bfloat16}[self.precision]