import pyrallis
import torch
import json
from dataclasses import asdict
from pathlib import Path

from src.config import BenchConfig
from src.utils import quick_mean, quick_std, build_model


def train_step(
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    x: torch.Tensor,
    y: torch.Tensor,
    cfg: BenchConfig,
) -> tuple[torch.cuda.Event, torch.cuda.Event, torch.cuda.Event, torch.cuda.Event]:
    """
    Performing one full step. Returns CUDA events: start, after-fwd, after-bwd, after-opt.
    """

    opt.zero_grad(set_to_none=True)
    e_start, e_fwd, e_bwd, e_opt = (torch.cuda.Event(enable_timing=True) for _ in range(4))

    e_start.record()
    # forward
    with torch.amp.autocast("cuda", dtype=cfg.autocast_dtype, enabled=cfg.use_autocast):
        logits = model(x, grad_checkpoint_strat=cfg.grad_checkpoint_strat)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))
    e_fwd.record()

    # scaling&loss
    if cfg.use_scaler:
        scaler.scale(loss).backward()
        e_bwd.record()
        scaler.step(opt)
        scaler.update()
    else:
        loss.backward()
        e_bwd.record()
        # optimizer step
        opt.step()
    e_opt.record()

    return e_start, e_fwd, e_bwd, e_opt


def bench(cfg: BenchConfig) -> dict:
    torch.manual_seed(cfg.seed)

    # dummy data
    x = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len), device="cuda")
    y = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len), device="cuda")

    # init params
    model = build_model(cfg)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_scaler)

    fwd_ms, bwd_ms, opt_ms, peak_bytes = [], [], [], []
    oom = False

    # try in case of oom
    try:
        for i in range(cfg.warmup_iters + cfg.measure_iters):
            measured = i >= cfg.warmup_iters
            if measured:
                torch.cuda.reset_peak_memory_stats()

            e_start, e_fwd, e_bwd, e_opt = train_step(model, opt, scaler, x, y, cfg)
            torch.cuda.synchronize()

            if measured:
                fwd_ms.append(e_start.elapsed_time(e_fwd))
                bwd_ms.append(e_fwd.elapsed_time(e_bwd))
                opt_ms.append(e_bwd.elapsed_time(e_opt))
                peak_bytes.append(torch.cuda.max_memory_allocated())
    # if oom - still dump config
    except torch.cuda.OutOfMemoryError:
        oom = True

    if fwd_ms:
        metrics = {
            "fwd_ms": round(quick_mean(fwd_ms), 3),
            "fwd_std": round(quick_std(fwd_ms), 3) if len(fwd_ms) > 1 else 0.0,
            "bwd_ms": round(quick_mean(bwd_ms), 3),
            "bwd_std": round(quick_std(bwd_ms), 3) if len(bwd_ms) > 1 else 0.0,
            "opt_ms": round(quick_mean(opt_ms), 3),
            "peak_gb": round(max(peak_bytes) / 1e9, 3),
            "tok_s": round(cfg.batch_size * cfg.seq_len
                        / ((quick_mean(fwd_ms)
                        + quick_mean(bwd_ms)
                        + quick_mean(opt_ms)) / 1000)),
            "oom": oom
        }
    else:
        metrics = {
            "fwd_ms": None, "fwd_std": None, "bwd_ms": None, "bwd_std": None,
            "opt_ms": None, "peak_gb": None, "tok_s": None, "oom": True
        }

    # dict with config & metrics
    return {
        "run_name": cfg.run_name,
        "precision": cfg.precision,
        "compile": cfg.compile,
        "grad_checkpoint_strat": cfg.grad_checkpoint_strat,
        "attention_impl": cfg.attention_impl,
        "is_custom": cfg.is_custom,
        "batch_size": cfg.batch_size,
        "seq_len": cfg.seq_len,
        **metrics,
        "config": asdict(cfg)
    }


def save_result(record: dict, cfg: BenchConfig) -> Path:
    out_dir = Path(cfg.result_dir)
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{cfg.run_name}.json"
    path.write_text(json.dumps(record, indent=2, default=str))
    return path


def main() -> None:
    cfg = pyrallis.parse(config_class=BenchConfig)
    # benchmark config
    record = bench(cfg)
    # dump json
    save_result(record, cfg)
    for k in ("fwd_ms", "fwd_std", "bwd_ms", "bwd_std",
              "opt_ms", "peak_gb", "tok_s", "oom"):
        print(f" {k}: {record[k]}")


if __name__ == "__main__":
    main()