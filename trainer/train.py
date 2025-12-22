import os
import glob
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from model import EgoSnakeNet

import polars as pl

DATA_DIR = "data"
MODEL_PATH = "models/latest.pt"
BATCH_SIZE = 256      # Larger batch = faster training (if GPU memory allows)
EPOCHS = 10           # More passes over the data
LR = 0.01             # 10x higher learning rate for faster convergence
EXPECTED_X_BYTES = 14 * 11 * 11 * 4

class SnakeDataset(Dataset):
    def __init__(self, file_paths, max_examples: int | None = None):
        self.rows = []
        for path in file_paths:
            try:
                df = pl.read_parquet(path, columns=["x", "policy", "value"])
                x_list = df.get_column("x").to_list()
                policy_list = df.get_column("policy").to_list()
                value_list = df.get_column("value").to_list()

                for x, p, v in zip(x_list, policy_list, value_list):
                    if isinstance(x, memoryview):
                        x = x.tobytes()
                    if len(x) != EXPECTED_X_BYTES:
                        # Skip malformed rows rather than crashing mid-training.
                        continue
                    self.rows.append((x, int(p), float(v)))
                    if max_examples is not None and len(self.rows) >= max_examples:
                        return
            except (OSError, ValueError, pl.exceptions.PolarsError) as e:
                print(f"Error reading {path}: {e}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        x_bytes, policy_label, value_scalar = self.rows[idx]

        # State: 14x11x11 float32 (little-endian)
        state = np.frombuffer(x_bytes, dtype=np.float32).reshape(14, 11, 11).copy()

        # Policy: int64 class label in [0..3]
        policy = np.int64(policy_label)

        # Value: float32 scalar in [-1..1]
        value = np.float32(value_scalar)

        return (
            torch.from_numpy(state),
            torch.tensor(policy, dtype=torch.long),
            torch.tensor(value, dtype=torch.float32),
        )

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-files", type=int, default=0, help="0 = no limit")
    parser.add_argument("--max-examples", type=int, default=0, help="0 = no limit")
    return parser.parse_args()


def train():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # Load Data
    files = glob.glob(os.path.join(args.data_dir, "*.parquet"))
    if not files:
        print("No training data found.")
        return

    if args.max_files and args.max_files > 0:
        files = files[: args.max_files]

    print(f"Found {len(files)} data files.")
    max_examples = None if not args.max_examples or args.max_examples <= 0 else args.max_examples

    # Load Model
    model = EgoSnakeNet(width=11, height=11).to(device)
    if os.path.exists(args.model_path):
        print(f"Loading existing model from {args.model_path}")
        try:
            model.load_state_dict(torch.load(args.model_path, map_location=device))
        except RuntimeError as e:
            print("Warning: existing checkpoint is incompatible with current model; starting fresh.")
            print(f"Checkpoint: {args.model_path}")
            print(f"Error: {e}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Loss Functions
    # Policy: Cross Entropy over logits
    # Value: MSE
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        total_batches = 0

        epoch_files = list(files)
        random.shuffle(epoch_files)

        for file_idx, path in enumerate(epoch_files, start=1):
            dataset = SnakeDataset([path], max_examples=max_examples)
            if len(dataset) == 0:
                continue

            dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=False,
            )

            for states, policies, values in dataloader:
                states = states.to(device)
                policies = policies.to(device)
                values = values.to(device)

                optimizer.zero_grad()

                pred_policies, pred_values = model(states)

                loss_policy = ce_loss(pred_policies, policies)
                loss_value = mse_loss(pred_values.squeeze(1), values)

                loss = loss_policy + loss_value
                loss.backward()
                optimizer.step()

                total_loss += float(loss.item())
                total_batches += 1

            print(f"  epoch {epoch+1}/{args.epochs}: {file_idx}/{len(epoch_files)} files")

        if total_batches == 0:
            print(f"Epoch {epoch+1}/{args.epochs}, no batches (no data)")
            break
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {total_loss / total_batches:.4f}")

    # Save Model
    torch.save(model.state_dict(), args.model_path)
    print(f"Model saved to {args.model_path}")

    # Cleanup Data?
    # For now, keep it. In a real loop, we might archive it.

if __name__ == '__main__':
    train()
