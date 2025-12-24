import argparse
import time

import torch
import torch.nn as nn

from model import EgoSnakeNet


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _bench_one(
    *,
    device: torch.device,
    in_channels: int,
    batch_size: int,
    steps: int,
    warmup: int,
    compile_model: bool,
    channels_last: bool,
    autocast_bf16: bool,
) -> dict:
    width = 11
    height = 11

    model = EgoSnakeNet(width=width, height=height, in_channels=in_channels).to(device)
    model.train()

    if device.type == "cuda" and channels_last:
        model = model.to(memory_format=torch.channels_last)

    if compile_model and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except RuntimeError:
            compile_model = False

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    x = torch.randn(batch_size, in_channels, height, width, device=device, dtype=torch.float32)
    if device.type == "cuda" and channels_last:
        x = x.contiguous(memory_format=torch.channels_last)

    y_policy = torch.randint(0, 4, (batch_size,), device=device, dtype=torch.long)
    y_value = torch.empty(batch_size, device=device, dtype=torch.float32).uniform_(-1, 1)

    amp_enabled = device.type == "cuda" and autocast_bf16

    # Warmup
    for _ in range(warmup):
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp_enabled):
            p, v = model(x)
            loss = ce(p, y_policy) + mse(v.squeeze(1).float(), y_value)
        loss.backward()
        optimizer.step()

    _sync(device)

    # Timed
    t0 = time.perf_counter()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp_enabled):
            p, v = model(x)
            loss = ce(p, y_policy) + mse(v.squeeze(1).float(), y_value)
        loss.backward()
        optimizer.step()

    _sync(device)
    dt = max(1e-9, time.perf_counter() - t0)

    ex_per_s = (steps * batch_size) / dt

    peak_mem = None
    if device.type == "cuda":
        peak_mem = int(torch.cuda.max_memory_allocated())

    return {
        "device": str(device),
        "in_channels": in_channels,
        "batch_size": batch_size,
        "steps": steps,
        "warmup": warmup,
        "compile": bool(compile_model),
        "channels_last": bool(channels_last),
        "autocast_bf16": bool(autocast_bf16),
        "seconds": dt,
        "examples_per_sec": ex_per_s,
        "peak_cuda_mem_bytes": peak_mem,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--channels-last", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--autocast-bf16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--compare-in-channels",
        default="6,10",
        help="Comma-separated list of in_channels values to benchmark (e.g. '6' or '6,10').",
    )
    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except (AttributeError, RuntimeError):
            pass
        torch.cuda.reset_peak_memory_stats()

    in_channels_list = [int(x.strip()) for x in str(args.compare_in_channels).split(",") if x.strip()]

    print("=== Training micro-benchmark (forward+backward+Adam) ===")
    print(
        f"device={device} batch={args.batch_size} steps={args.steps} warmup={args.warmup} "
        f"compile={args.compile} channels_last={args.channels_last} autocast_bf16={args.autocast_bf16}"
    )

    for c in in_channels_list:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        r = _bench_one(
            device=device,
            in_channels=c,
            batch_size=args.batch_size,
            steps=args.steps,
            warmup=args.warmup,
            compile_model=bool(args.compile),
            channels_last=bool(args.channels_last),
            autocast_bf16=bool(args.autocast_bf16),
        )
        peak = r["peak_cuda_mem_bytes"]
        peak_s = f" peak_mem={peak/1e9:.2f}GB" if peak is not None else ""
        print(
            f"in_channels={c}: {r['examples_per_sec']:.0f} examples/s "
            f"({r['seconds']:.3f}s){peak_s}"
        )


if __name__ == "__main__":
    main()
