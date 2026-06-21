from src.model import GPTLM
from src.config import BenchConfig, AttnSweepConfig

import torch

import json, pandas as pd, matplotlib.pyplot as plt



def quick_mean(nums):
    return sum(nums) / len(nums)


def quick_std(nums):
    mean = quick_mean(nums)
    return sum((x - mean) ** 2 for x in nums) / (len(nums) - 1) ** 0.5



def build_model(cfg: BenchConfig | AttnSweepConfig) -> torch.nn.Module:
    model = GPTLM(      
        num_layers=cfg.num_layers,  
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        max_seq_len=max(cfg.max_seq_len, cfg.seq_len),
        theta=cfg.theta,
        is_custom=cfg.is_custom,
        attention_impl=cfg.attention_impl).to("cuda")
    if cfg.compile:
        model = torch.compile(model)
    return model


def plot_attention_sweep_chart():
    df = pd.DataFrame(json.load(open("results/attn_sweep/attention_sweep.json"))["results"])
    df = df[df.get("skipped_after_oom") != True].dropna(subset=["peak_gb", "tok_s"]).sort_values("seq_len")

    color = {"eager": "C0", "sdpa": "C1"}
    style = {True: "-", False: "--"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (col, ylabel) in zip(axes, [("peak_gb", "peak memory (GB)"), ("tok_s", "throughput (tok/s)")]):
        for (impl, comp), sub in df.groupby(["attn_impl", "compile"]):
            ax.plot(sub.seq_len, sub[col], color=color[impl], linestyle=style[comp],
                    marker="o", ms=4, label=f"{impl}, {'compiled' if comp else 'not compiled'}")
        ax.set(xlabel="seq_len", ylabel=ylabel, xscale="log")
        ax.set_xticks(sorted(df.seq_len.unique()))
        ax.xaxis.set_major_formatter(plt.ScalarFormatter())
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Attention impl sweep fp16, batch=1")
    fig.tight_layout()
    fig.savefig("results/attn_sweep/attn_sweep.png", dpi=150)