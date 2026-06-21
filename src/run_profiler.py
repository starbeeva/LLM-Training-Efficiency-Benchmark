from dataclasses import asdict
from pathlib import Path

import pyrallis
import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, schedule

from src.config import BenchConfig
from src.bench_single_gpu import build_model


def step(model, opt, scaler, x, y, cfg: BenchConfig) -> None:
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=cfg.autocast_dtype, enabled=cfg.use_autocast):
        logits = model(x, grad_checkpoint_strat=cfg.grad_checkpoint_strat)
        loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))
    if cfg.use_scaler:
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
    else:
        loss.backward()
        opt.step()


@pyrallis.wrap()
def main(cfg: BenchConfig) -> None:
    torch.manual_seed(cfg.seed)

    out_dir = Path("results/profiles") / f"{cfg.run_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    x = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len), device="cuda")
    y = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len), device="cuda")

    model = build_model(cfg); 
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_scaler)

    for _ in range(5):
        step(model, opt, scaler, x, y, cfg)
    torch.cuda.synchronize()

    sched = schedule(wait=0, warmup=2, active=3, repeat=1)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=sched,
        record_shapes=True,
        profile_memory=True,
        with_stack=False
    ) as prof:
        for _ in range(5):
            step(model, opt, scaler, x, y, cfg)
            torch.cuda.synchronize()
            prof.step()

    trace_path = out_dir / "trace.json.gz"
    prof.export_chrome_trace(str(trace_path))

    cuda_table = prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=25)
    (out_dir / "top_cuda_ops.txt").write_text(cuda_table)

    mem_table = prof.key_averages().table(
        sort_by="self_cuda_memory_usage", row_limit=15)
    (out_dir / "top_memory_ops.txt").write_text(mem_table)

    try:
        prof.export_memory_timeline(str(out_dir / "memory_timeline.html"), device="cuda:0")
    except Exception as e:                                # noqa: BLE001
        (out_dir / "memory_timeline.skipped").write_text(f"{type(e).__name__}: {e}\n")

    print(cuda_table)
    (out_dir / "config.json").write_text(
        __import__("json").dumps(asdict(cfg), indent=2, default=str)
    )


if __name__ == "__main__":
    main()