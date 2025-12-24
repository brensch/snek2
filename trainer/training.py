from __future__ import annotations

import os
import time
from typing import Iterable

import torch
import torch.nn as nn
import torch.optim as optim

from model import EgoSnakeNet
from constants import HEIGHT, IN_CHANNELS, WIDTH


def setup_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except (AttributeError, RuntimeError):
            pass
    return device


def load_model(
    model_path: str,
    device: torch.device,
    *,
    width: int = WIDTH,
    height: int = HEIGHT,
    in_channels: int = IN_CHANNELS,
) -> EgoSnakeNet:
    model = EgoSnakeNet(width=int(width), height=int(height), in_channels=int(in_channels)).to(device)

    if os.path.exists(model_path):
        print(f"Loading checkpoint: {model_path}")
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
        except RuntimeError as e:
            print("Warning: checkpoint incompatible; starting fresh")
            print(f"Error: {e}")

    return model


def train_epochs(
    *,
    model: torch.nn.Module,
    loader: Iterable,
    epochs: int,
    lr: float,
    device: torch.device,
    amp: bool,
    seed: int,
    dataset_obj: object | None = None,
) -> bool:
    torch.manual_seed(int(seed))

    optimizer = optim.Adam(model.parameters(), lr=float(lr))
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and amp))

    model.train()

    for epoch in range(int(epochs)):
        if dataset_obj is not None and hasattr(dataset_obj, "set_epoch"):
            dataset_obj.set_epoch(epoch)

        total_loss = 0.0
        total_batches = 0
        t0 = time.time()
        last_log_t = t0

        for states, policies, values in loader:
            states = states.to(device, non_blocking=True)
            policies = policies.to(device, non_blocking=True, dtype=torch.long)
            values = values.to(device, non_blocking=True, dtype=torch.float32)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda" and amp)):
                pred_policies, pred_values = model(states)
                loss_policy = ce_loss(pred_policies, policies)
                loss_value = mse_loss(pred_values.squeeze(1).float(), values.float())
                loss = loss_policy + loss_value

            if not torch.isfinite(loss).all():
                print("Non-finite loss detected; aborting training")
                return False

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.item())
            total_batches += 1

            now = time.time()
            if total_batches == 1 or (now - last_log_t) >= 30.0:
                dt = max(1e-9, now - t0)
                avg = total_loss / max(1, total_batches)
                print(f"Epoch {epoch+1}/{epochs}: {total_batches} batches, avg_loss {avg:.4f} ({total_batches / dt:.2f} batches/s)")
                last_log_t = now

        if total_batches == 0:
            print(f"Epoch {epoch+1}/{epochs}: no data")
            return False

        dt = max(1e-9, time.time() - t0)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / total_batches:.4f} ({total_batches / dt:.2f} batches/s)")

    return True


def save_checkpoint(model: torch.nn.Module, model_path: str) -> None:
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
