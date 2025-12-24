import argparse
import os
import time

import torch
import numpy as np

from model import EgoSnakeNet

_warned_incompatible_ckpt: set[tuple[str, int]] = set()


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def bench_torch(
    *,
    device: torch.device,
    in_channels: int,
    batch_size: int,
    seconds: float,
    warmup: int,
    compile_model: bool,
    channels_last: bool,
    autocast_bf16: bool,
    ckpt: str,
    skip_ckpt: bool,
) -> dict:
    width = 11
    height = 11

    model = EgoSnakeNet(width=width, height=height, in_channels=in_channels).to(device)
    if not skip_ckpt:
        if ckpt and os.path.exists(ckpt):
            sd = torch.load(ckpt, map_location=device)
            try:
                model.load_state_dict(sd)
            except RuntimeError as e:
                # Common when benchmarking after changing input channels.
                key = (ckpt, int(in_channels))
                if key not in _warned_incompatible_ckpt:
                    _warned_incompatible_ckpt.add(key)
                    print(
                        f"Warning: checkpoint incompatible with in_channels={in_channels}; benchmarking random weights ({e})"
                    )
    model.eval()

    if device.type == "cuda" and channels_last:
        model = model.to(memory_format=torch.channels_last)

    compiled = False
    if compile_model and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            compiled = True
        except RuntimeError:
            compiled = False

    x = torch.randn(batch_size, in_channels, height, width, device=device, dtype=torch.float32)
    if device.type == "cuda" and channels_last:
        x = x.contiguous(memory_format=torch.channels_last)

    amp_enabled = device.type == "cuda" and autocast_bf16

    # Warmup (iteration-count bounded)
    for _ in range(warmup):
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp_enabled):
            _p, _v = model(x)
    _sync(device)

    # Timed (time-bounded): estimate per-iter time, then run a fixed iteration count.
    t_est0 = time.perf_counter()
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp_enabled):
        _p, _v = model(x)
    _sync(device)
    est_dt = max(1e-6, time.perf_counter() - t_est0)
    iters = max(1, int(float(seconds) / est_dt))

    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(iters):
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp_enabled):
                _p, _v = model(x)
    _sync(device)
    dt = max(1e-9, time.perf_counter() - t0)

    inf_per_s = (iters * batch_size) / dt

    return {
        "backend": "torch",
        "device": str(device),
        "in_channels": in_channels,
        "batch_size": batch_size,
        "seconds": dt,
        "iters": iters,
        "inferences_per_sec": inf_per_s,
        "compile": compiled,
        "channels_last": bool(channels_last),
        "autocast_bf16": bool(autocast_bf16),
    }


def export_onnx_random(
    *,
    onnx_path: str,
    in_channels: int,
    opset: int = 18,
    device: torch.device | None = None,
    dtype: str = "fp32",
) -> None:
    os.makedirs(os.path.dirname(onnx_path) or ".", exist_ok=True)

    if device is None:
        device = torch.device("cpu")

    dtype = str(dtype).lower()
    if dtype not in ("fp32", "fp16", "fp16-f32io"):
        raise ValueError(f"dtype must be fp32, fp16, or fp16-f32io, got {dtype!r}")
    if dtype in ("fp16", "fp16-f32io") and device.type != "cuda":
        raise ValueError("fp16 ONNX export requires --device=cuda")

    base_model = EgoSnakeNet(width=11, height=11, in_channels=int(in_channels)).eval().to(device)
    if dtype == "fp16":
        # FP16 I/O (input + outputs are float16)
        model = base_model.half()
        x = torch.randn(1, int(in_channels), 11, 11, device=device, dtype=torch.float16)
    elif dtype == "fp16-f32io":
        # FP16 compute but keep float32 inputs/outputs so non-fp16 clients (e.g. Go)
        # can still feed/read float32 without extra glue.
        half_model = base_model.half()

        class _F32IOWrapper(torch.nn.Module):
            def __init__(self, inner: torch.nn.Module):
                super().__init__()
                self.inner = inner

            def forward(self, x_in: torch.Tensor):
                x_h = x_in.to(dtype=torch.float16)
                p, v = self.inner(x_h)
                return p.to(dtype=torch.float32), v.to(dtype=torch.float32)

        model = _F32IOWrapper(half_model).eval()
        x = torch.randn(1, int(in_channels), 11, 11, device=device, dtype=torch.float32)
    else:
        model = base_model
        x = torch.randn(1, int(in_channels), 11, 11, device=device, dtype=torch.float32)

    torch.onnx.export(
        model,
        x,
        onnx_path,
        input_names=["input"],
        output_names=["policy", "value"],
        dynamic_axes={
            "input": {0: "batch"},
            "policy": {0: "batch"},
            "value": {0: "batch"},
        },
        opset_version=int(opset),
        # Critical for correct dynamic batch behavior here.
        # The dynamo exporter previously produced a batch=1 reshape that failed for bs>1.
        dynamo=False,
        external_data=False,
        do_constant_folding=True,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--in-channels", type=int, default=10)
    p.add_argument("--batch-sizes", default="1,128,512,2048")
    p.add_argument("--seconds", type=float, default=1.5)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--channels-last", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--autocast-bf16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ckpt", default="models/latest.pt")
    p.add_argument(
        "--skip-ckpt",
        action="store_true",
        help="Do not attempt to load a checkpoint (use random weights)",
    )
    p.add_argument(
        "--save-random-ckpt",
        default="",
        help="If set, write a random-weight checkpoint (state_dict) to this path and continue",
    )
    p.add_argument("--onnx", default="models/snake_net.onnx")
    p.add_argument(
        "--export-onnx",
        action="store_true",
        help="Export a random-weight ONNX model to --onnx before benchmarking ORT",
    )
    p.add_argument("--onnx-opset", type=int, default=18)
    p.add_argument(
        "--onnx-dtype",
        default="fp32",
        choices=["fp32", "fp16", "fp16-f32io"],
        help="ONNX export/input dtype for ORT benchmark. fp16 requires CUDA.",
    )
    p.add_argument(
        "--ort-provider",
        default="auto",
        choices=["auto", "cuda", "tensorrt", "cpu"],
        help="Preferred ONNXRuntime EP (falls back automatically)",
    )
    p.add_argument(
        "--ort-iobinding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use CUDA I/O binding when CUDA EP is active (faster, avoids host copies)",
    )
    p.add_argument(
        "--ort-cuda-options",
        action="store_true",
        help="(Experimental) Pass explicit CUDAExecutionProvider options (may hurt performance)",
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

    batch_sizes = [int(x.strip()) for x in str(args.batch_sizes).split(",") if x.strip()]

    print("=== Inference benchmark ===")
    print(
        f"backend=torch device={device} in_channels={args.in_channels} "
        f"compile={args.compile} channels_last={args.channels_last} autocast_bf16={args.autocast_bf16} "
        f"seconds={args.seconds} warmup={args.warmup}"
    )

    if args.save_random_ckpt:
        out_path = str(args.save_random_ckpt)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        tmp = EgoSnakeNet(width=11, height=11, in_channels=int(args.in_channels))
        torch.save(tmp.state_dict(), out_path)
        print(f"Wrote random checkpoint: {out_path}")

    for bs in batch_sizes:
        r = bench_torch(
            device=device,
            in_channels=int(args.in_channels),
            batch_size=int(bs),
            seconds=float(args.seconds),
            warmup=int(args.warmup),
            compile_model=bool(args.compile),
            channels_last=bool(args.channels_last),
            autocast_bf16=bool(args.autocast_bf16),
            ckpt=str(args.ckpt),
            skip_ckpt=bool(args.skip_ckpt),
        )
        print(
            f"batch={bs}: {r['inferences_per_sec']:.0f} inf/s "
            f"(iters={r['iters']} time={r['seconds']:.3f}s compile={r['compile']})"
        )

    # ONNXRuntime benchmark (if available)
    try:
        import onnxruntime as ort  # type: ignore

        if args.export_onnx:
            export_onnx_random(
                onnx_path=str(args.onnx),
                in_channels=int(args.in_channels),
                opset=int(args.onnx_opset),
                device=device,
                dtype=str(args.onnx_dtype),
            )
            print(
                f"\nExported random ONNX: {args.onnx} (in_channels={args.in_channels} opset={args.onnx_opset} dtype={args.onnx_dtype})"
            )

        if not args.onnx or not os.path.exists(args.onnx):
            print(f"\nONNX model not found at {args.onnx}; skipping ONNXRuntime benchmark")
            return

        providers = ort.get_available_providers()
        print("\n=== ONNXRuntime inference benchmark ===")
        print("Available providers:", providers)

        sess_opts = ort.SessionOptions()
        try:
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        except (AttributeError, ValueError, RuntimeError):
            pass
        sess_opts.intra_op_num_threads = 1
        sess_opts.inter_op_num_threads = 1

        # Provider selection with fallback. If CUDA is chosen, pass provider options.
        preferred: list[object] = []
        want_trt = str(args.ort_provider) == "tensorrt"
        want_cuda = str(args.ort_provider) in ("auto", "cuda", "tensorrt")

        if want_trt and "TensorrtExecutionProvider" in providers:
            preferred.append("TensorrtExecutionProvider")

        if want_cuda and "CUDAExecutionProvider" in providers:
            if bool(args.ort_cuda_options):
                cuda_opts = {
                    "cudnn_conv_algo_search": "DEFAULT",
                    "cudnn_conv_use_max_workspace": 1,
                    "do_copy_in_default_stream": 1,
                }
                preferred.append(("CUDAExecutionProvider", cuda_opts))
            else:
                preferred.append("CUDAExecutionProvider")

        preferred.append("CPUExecutionProvider")

        session = ort.InferenceSession(args.onnx, sess_options=sess_opts, providers=preferred)
        in_name = session.get_inputs()[0].name
        out_names = [o.name for o in session.get_outputs()]

        providers_used = session.get_providers()
        using_cuda = "CUDAExecutionProvider" in providers_used

        print(
            f"providers_used={providers_used} onnx={args.onnx} seconds={args.seconds} warmup={args.warmup} "
            f"iobinding={bool(args.ort_iobinding) and using_cuda}"
        )

        # Warmup + timed loops (time-bounded via fixed iters like torch)
        for bs in batch_sizes:
            x_np_dtype = np.float16 if str(args.onnx_dtype) == "fp16" else np.float32
            x_cpu = np.random.randn(int(bs), int(args.in_channels), 11, 11).astype(x_np_dtype)

            if using_cuda and bool(args.ort_iobinding):
                # Upload once; reuse GPU buffers for all iterations.
                x_dev = ort.OrtValue.ortvalue_from_numpy(x_cpu, "cuda", 0)
                io = session.io_binding()
                io.bind_ortvalue_input(in_name, x_dev)
                for out in out_names:
                    io.bind_output(out, "cuda", 0)

                for _ in range(int(args.warmup)):
                    session.run_with_iobinding(io)
                    io.synchronize_outputs()

                t_est0 = time.perf_counter()
                session.run_with_iobinding(io)
                io.synchronize_outputs()
                est_dt = max(1e-6, time.perf_counter() - t_est0)
                iters = max(1, int(float(args.seconds) / est_dt))

                t0 = time.perf_counter()
                for _ in range(iters):
                    session.run_with_iobinding(io)
                io.synchronize_outputs()
                dt = max(1e-9, time.perf_counter() - t0)
            else:
                for _ in range(int(args.warmup)):
                    _ = session.run(out_names, {in_name: x_cpu})

                t_est0 = time.perf_counter()
                _ = session.run(out_names, {in_name: x_cpu})
                est_dt = max(1e-6, time.perf_counter() - t_est0)
                iters = max(1, int(float(args.seconds) / est_dt))

                t0 = time.perf_counter()
                for _ in range(iters):
                    _ = session.run(out_names, {in_name: x_cpu})
                dt = max(1e-9, time.perf_counter() - t0)

            inf_per_s = (iters * int(bs)) / dt
            print(f"batch={bs}: {inf_per_s:.0f} inf/s (iters={iters} time={dt:.3f}s)")

    except ImportError:
        print("\nonnxruntime not installed; skipping ONNXRuntime benchmark")


if __name__ == "__main__":
    main()
