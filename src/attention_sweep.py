"""Attention seq-len sweep at batch_size=1.

Compares three configurations across seq_len:
  - eager fp16 (manual softmax(QK^T) — O(S^2) memory)
  - sdpa  fp16 (F.SDPA — picks memory-efficient backend on Turing)
  - eager fp16 compiled (how well torch.compile shrinks eager attention)

Walks seq_len up; once a configuration OOMs, remaining lengths for it are
recorded as OOM without re-running (they will OOM too).

Usage:
    python -m benchmark.attention_sweep
    python -m benchmark.attention_sweep --seq_lens "[1024,2048,4096,8192]"
"""
import gc
import json
import statistics
from dataclasses import asdict
from pathlib import Path

import pyrallis
import torch
import torch.nn.functional as F

from src.config import AttnSweepConfig
from src.utils import quick_mean, quick_std, build_model


def do_sweep(cfg: AttnSweepConfig, attn_impl: str, do_compile: bool, seq_len: int) -> dict:
    """
    One (attn_impl, compile, seq_len) point. Returns metrics or OOM marker.
    """
    torch.manual_seed(cfg.seed)
    gc.collect()
    torch.cuda.empty_cache()

    base = {
        "attn_impl": attn_impl,
        "compile":   do_compile,
        "seq_len":   seq_len,
        "precision": cfg.precision,
    }

    model, opt = None, None
    try:
        model = build_model(cfg).to("cuda")
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=cfg.precision == "fp16")

        x = torch.randint(0, cfg.vocab_size, (cfg.batch_size, seq_len), device="cuda")
        y = torch.randint(0, cfg.vocab_size, (cfg.batch_size, seq_len), device="cuda")

        fwd_ms, bwd_ms, peak_bytes = [], [], []

        for i in range(cfg.warmup_iters + cfg.measure_iters):
            measured = i >= cfg.warmup_iters
            if measured:
                torch.cuda.reset_peak_memory_stats()

            opt.zero_grad(set_to_none=True)
            e_s, e_f, e_b = (torch.cuda.Event(enable_timing=True) for _ in range(3))

            e_s.record()
            with torch.autocast("cuda", dtype=cfg.autocast_dtype,
                                enabled=cfg.precision != "fp32"):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))
            e_f.record()

            if cfg.precision == "fp16":
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                opt.step()
            e_b.record()
            torch.cuda.synchronize()

            if measured:
                fwd_ms.append(e_s.elapsed_time(e_f))
                bwd_ms.append(e_f.elapsed_time(e_b))
                peak_bytes.append(torch.cuda.max_memory_allocated())

        fwd = statistics.mean(fwd_ms)
        bwd = statistics.mean(bwd_ms)
        return {
            **base,
            "fwd_ms": round(fwd, 3),
            "bwd_ms": round(bwd, 3),
            "peak_gb": round(max(peak_bytes) / 1e9, 3),
            "tok_s": round(cfg.batch_size * seq_len / ((fwd + bwd) / 1000)),
            "oom": False,
        }
    except torch.cuda.OutOfMemoryError:
        return {
            **base,
            "fwd_ms": None, "bwd_ms": None, "peak_gb": None, "tok_s": None,
            "oom": True,
        }
    finally:
        del model, opt
        gc.collect()
        torch.cuda.empty_cache()


@pyrallis.wrap()
def main(cfg: AttnSweepConfig) -> None:
    out_dir = Path(cfg.result_dir)
    out_dir.mkdir(exist_ok=True)

    configurations = [
        ("eager", False),
        ("sdpa",  False),
        ("eager", True),
        ("sdpa",  True)
    ]

    results = []
    for attn_impl, do_compile in configurations:
        oom_hit = False
        for seq_len in sorted(cfg.seq_lens):
            if oom_hit:
                results.append({
                    "attn_impl": attn_impl, 
                    "compile": do_compile,
                    "seq_len": cfg.seq_len, 
                    "precision": cfg.precision,
                    "fwd_ms": None, "bwd_ms": None, "peak_gb": None,
                    "tok_s": None, "oom": True, "skipped_after_oom": True
                })
                continue
            
            print(f"Changing compile flag {cfg.compile} to {do_compile} and maybe max_seq_len from {cfg.max_seq_len} to {seq_len} to  in model")

            cfg.compile = do_compile
            cfg.seq_len = seq_len
            cfg.max_seq_len = max(seq_len, cfg.max_seq_len)
            cfg.attention_impl = attn_impl
            
            r = do_sweep(cfg, attn_impl, do_compile, seq_len)
            tag = f"{attn_impl}{'_compiled' if do_compile else ''}@{seq_len}"
            print(f"  {tag}: peak={r['peak_gb']} GB, "
                      f"tok/s={r['tok_s']}, oom={r['oom']}")
            results.append(r)
            if r["oom"]:
                oom_hit = True

    payload = {"results": results, "config": asdict(cfg)}
    path = Path(f"{out_dir}/attention_sweep.json")
    path.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()