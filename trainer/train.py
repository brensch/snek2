import os
import glob
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

class SnakeDataset(Dataset):
    def __init__(self, file_paths):
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
                    self.rows.append((x, int(p), float(v)))
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

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # Load Data
    files = glob.glob(os.path.join(DATA_DIR, "*.parquet"))
    if not files:
        print("No training data found.")
        return

    print(f"Found {len(files)} data files.")
    dataset = SnakeDataset(files)
    print(f"Loaded {len(dataset)} examples.")

    if len(dataset) == 0:
        return

    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=4,        # Parallel data loading
        pin_memory=True,      # Faster GPU transfer
        persistent_workers=True
    )

    # Load Model
    model = EgoSnakeNet(width=11, height=11).to(device)
    if os.path.exists(MODEL_PATH):
        print(f"Loading existing model from {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Loss Functions
    # Policy: Cross Entropy over logits
    # Value: MSE
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
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

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss / len(dataloader):.4f}")

    # Save Model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # Cleanup Data?
    # For now, keep it. In a real loop, we might archive it.

if __name__ == '__main__':
    train()
