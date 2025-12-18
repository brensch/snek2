import torch
import time
import sys
import os

# Add generated code to path just in case, though not needed for model
sys.path.append(os.path.join(os.path.dirname(__file__), '../gen/python'))

from model import SnakeNet

BATCH_SIZE = 4096
IN_CHANNELS = 17
WIDTH = 11
HEIGHT = 11

def benchmark():
    if not torch.cuda.is_available():
        print("CUDA not available! This benchmark is for GPU saturation.")
        # return # Commented out to allow testing on CPU if needed, but warning is clear

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    model = SnakeNet(in_channels=IN_CHANNELS, width=WIDTH, height=HEIGHT).to(device)
    model.eval()

    # Create dummy batch
    print(f"Creating dummy batch of size {BATCH_SIZE}...")
    dummy_input = torch.randn(BATCH_SIZE, IN_CHANNELS, WIDTH, HEIGHT, device=device)

    print("Warming up...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    print("Starting benchmark loop (Press Ctrl+C to stop)...")
    start_time = time.time()
    iterations = 0
    
    try:
        while True:
            with torch.no_grad():
                # Run a burst of 100 iterations
                for _ in range(100):
                    _ = model(dummy_input)
                    iterations += 1
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            current_time = time.time()
            elapsed = current_time - start_time
            
            if elapsed > 0:
                batches_per_sec = iterations / elapsed
                inferences_per_sec = batches_per_sec * BATCH_SIZE
                print(f"\rBatches/sec: {batches_per_sec:.2f} | Inferences/sec: {inferences_per_sec:.2f}", end="")
                
                # Reset counters every 5 seconds to keep average fresh
                if elapsed > 5:
                    start_time = time.time()
                    iterations = 0
                    print() # New line

    except KeyboardInterrupt:
        print("\nBenchmark stopped.")

if __name__ == "__main__":
    benchmark()
