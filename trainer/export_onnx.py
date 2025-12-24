import torch
import torch.onnx
import os
import argparse

# Add generated code to path
from model import EgoSnakeNet

IN_CHANNELS = 10
WIDTH = 11
HEIGHT = 11


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="models/latest.pt", help="PyTorch state_dict checkpoint")
    p.add_argument("--out", default="models/snake_net.onnx", help="Output ONNX path")
    p.add_argument("--opset", type=int, default=18)
    p.add_argument("--in-channels", type=int, default=IN_CHANNELS)
    p.add_argument(
        "--dtype",
        default="fp16-f32io",
        choices=["fp32", "fp16", "fp16-f32io"],
        help="Export dtype. fp16-f32io keeps float32 I/O but uses fp16 compute (good for Go).",
    )
    return p.parse_args()

def export():
    args = parse_args()
    # Create model
    base_model = EgoSnakeNet(width=WIDTH, height=HEIGHT, in_channels=int(args.in_channels))

    if os.path.exists(args.ckpt):
        try:
            base_model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
            print(f"Loaded checkpoint: {args.ckpt}")
        except RuntimeError as e:
            print(f"Warning: checkpoint incompatible with current model; exporting random weights ({e})")
    else:
        print(f"Warning: checkpoint not found ({args.ckpt}); exporting random weights")

    base_model.eval()

    dtype = str(args.dtype).lower()
    if dtype == "fp16":
        model = base_model.half()
        dummy_input = torch.randn(1, int(args.in_channels), WIDTH, HEIGHT, dtype=torch.float16)
    elif dtype == "fp16-f32io":
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
        dummy_input = torch.randn(1, int(args.in_channels), WIDTH, HEIGHT, dtype=torch.float32)
    else:
        model = base_model
        dummy_input = torch.randn(1, int(args.in_channels), WIDTH, HEIGHT, dtype=torch.float32)

    # Ensure models directory exists
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print(f"Exporting model to {args.out}...")
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        args.out,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
        input_names=['input'],
        output_names=['policy', 'value'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'policy': {0: 'batch_size'},
            'value': {0: 'batch_size'}
        }
    )
    print(f"Export complete! in_channels={int(args.in_channels)} dtype={dtype} out={args.out}")

if __name__ == "__main__":
    export()
