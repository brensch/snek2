import torch
import torch.onnx
import sys
import os

# Add generated code to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../gen/python'))

from model import SnakeNet

IN_CHANNELS = 17
WIDTH = 11
HEIGHT = 11
MODEL_PATH = "models/snake_net.onnx"

def export():
    # Create model
    model = SnakeNet(in_channels=IN_CHANNELS, width=WIDTH, height=HEIGHT)
    model.eval()

    # Create dummy input
    # Batch size is dynamic in the exported model usually, but we can specify it
    dummy_input = torch.randn(10, IN_CHANNELS, WIDTH, HEIGHT)

    # Ensure models directory exists
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    print(f"Exporting model to {MODEL_PATH}...")
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        MODEL_PATH,
        export_params=True,
        opset_version=18,
        do_constant_folding=False,
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
