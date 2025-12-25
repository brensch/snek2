from __future__ import annotations

import os
import time
from typing import Iterable

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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

    # Optional guard against a common failure mode: value head saturates to +1/-1 (tanh),
    # which can produce near-zero gradients and make value loss look "stuck".
    # This is now opt-in because it can be overly aggressive (and you generally
    # don't want to reset a head every time you load a checkpoint).
    if os.getenv("SNEK_RESET_SATURATED_VALUE_HEAD", "0") == "1":
        was_training = model.training
        model.eval()
        with torch.no_grad():
            x = torch.randn(32, int(in_channels), int(height), int(width), device=device, dtype=torch.float32)
            _, v = model(x)
            v = v.squeeze(1).float()
            if v.numel() > 0 and (float(v.min()) > 0.999 or float(v.max()) < -0.999):
                print("Warning: value head appears saturated; resetting value head")
                torch.nn.init.kaiming_uniform_(model.value_conv.weight, a=5**0.5)
                if model.value_conv.bias is not None:
                    torch.nn.init.zeros_(model.value_conv.bias)

                # Reset BN parameters and running stats.
                if hasattr(model.value_bn, "reset_running_stats"):
                    model.value_bn.reset_running_stats()
                if hasattr(model.value_bn, "reset_parameters"):
                    model.value_bn.reset_parameters()
                else:
                    if model.value_bn.weight is not None:
                        torch.nn.init.ones_(model.value_bn.weight)
                    if model.value_bn.bias is not None:
                        torch.nn.init.zeros_(model.value_bn.bias)

                torch.nn.init.kaiming_uniform_(model.value_fc1.weight, a=5**0.5)
                if model.value_fc1.bias is not None:
                    torch.nn.init.zeros_(model.value_fc1.bias)

                torch.nn.init.kaiming_uniform_(model.value_fc2.weight, a=5**0.5)
                if model.value_fc2.bias is not None:
                    torch.nn.init.zeros_(model.value_fc2.bias)
        model.train(was_training)

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
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_policy_acc = 0.0
        total_batches = 0
        t0 = time.time()
        last_log_t = t0

        for states, policies, values in loader:
            states = states.to(device, non_blocking=True)
            # policies can be either:
            # - LongTensor[B] class indices (legacy)
            # - FloatTensor[B,4] probability distributions (preferred)
            policies = policies.to(device, non_blocking=True)
            values = values.to(device, non_blocking=True, dtype=torch.float32)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda" and amp)):
                pred_policies, pred_values = model(states)
                if policies.dtype in (torch.int64, torch.int32, torch.int16, torch.int8):
                    loss_policy = ce_loss(pred_policies, policies.to(dtype=torch.long))
                    target_idx = policies.to(dtype=torch.long)
                else:
                    # Soft cross-entropy: -sum(p * log_softmax(logits)).
                    target = policies.to(dtype=torch.float32)
                    if target.ndim == 1:
                        # Defensive: if a caller passes floats as class ids.
                        target = target.to(dtype=torch.long)
                        loss_policy = ce_loss(pred_policies, target)
                        target_idx = target
                    else:
                        # Normalize in case of slight drift.
                        denom = target.sum(dim=1, keepdim=True).clamp_min(1e-8)
                        target = target / denom
                        logp = F.log_softmax(pred_policies.float(), dim=1)
                        loss_policy = -(target * logp).sum(dim=1).mean()
                        target_idx = target.argmax(dim=1)
                loss_value = mse_loss(pred_values.squeeze(1).float(), values.float())
                loss = loss_policy + loss_value

            if not torch.isfinite(loss).all():
                print("Non-finite loss detected; aborting training")
                return False

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.item())
            total_policy_loss += float(loss_policy.item())
            total_value_loss += float(loss_value.item())
            with torch.no_grad():
                pred_idx = pred_policies.argmax(dim=1)
                total_policy_acc += float((pred_idx == target_idx).float().mean().item())
            total_batches += 1

            now = time.time()
            if total_batches == 1 or (now - last_log_t) >= 30.0:
                dt = max(1e-9, now - t0)
                avg = total_loss / max(1, total_batches)
                avg_p = total_policy_loss / max(1, total_batches)
                avg_v = total_value_loss / max(1, total_batches)
                avg_acc = total_policy_acc / max(1, total_batches)
                print(
                    f"Epoch {epoch+1}/{epochs}: {total_batches} batches, "
                    f"avg_loss {avg:.4f} (policy {avg_p:.4f} value {avg_v:.4f} acc {avg_acc:.3f}) "
                    f"({total_batches / dt:.2f} batches/s)"
                )
                last_log_t = now

        if total_batches == 0:
            print(f"Epoch {epoch+1}/{epochs}: no data")
            return False

        dt = max(1e-9, time.time() - t0)
        avg = total_loss / total_batches
        avg_p = total_policy_loss / total_batches
        avg_v = total_value_loss / total_batches
        avg_acc = total_policy_acc / total_batches
        print(
            f"Epoch {epoch+1}/{epochs}, Loss: {avg:.4f} (policy {avg_p:.4f} value {avg_v:.4f} acc {avg_acc:.3f}) "
            f"({total_batches / dt:.2f} batches/s)"
        )

    return True


def save_checkpoint(model: torch.nn.Module, model_path: str) -> None:
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
