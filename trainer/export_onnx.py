import torch
import torch.onnx
import os
import argparse

# Add generated code to path
from model import EgoSnakeNet

IN_CHANNELS = 14
WIDTH = 11
HEIGHT = 11


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="models/latest.pt", help="PyTorch state_dict checkpoint")
    p.add_argument("--out", default="models/snake_net.onnx", help="Output ONNX path")
    p.add_argument("--opset", type=int, default=18)
    return p.parse_args()

def export():
    args = parse_args()
    # Create model
    model = EgoSnakeNet(width=WIDTH, height=HEIGHT)

    if os.path.exists(args.ckpt):
        model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
        print(f"Loaded checkpoint: {args.ckpt}")
    else:
        print(f"Warning: checkpoint not found ({args.ckpt}); exporting random weights")

    model.eval()

    # Create dummy input
    # Batch size is dynamic in the exported model usually, but we can specify it
    dummy_input = torch.randn(1, IN_CHANNELS, WIDTH, HEIGHT)

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
    print("Export complete!")

if __name__ == "__main__":
    export()
