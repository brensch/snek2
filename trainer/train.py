import argparse
import random

import numpy as np
import torch

from constants import DATA_DIR, DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, DEFAULT_LR, MODEL_PATH
from parquet_data import LoaderConfig, build_dataloader, list_parquet_files
from training import load_model, save_checkpoint, setup_device, train_epochs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=DATA_DIR)
    p.add_argument("--model-path", default=MODEL_PATH)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-files", type=int, default=0, help="0 = no limit")
    p.add_argument("--max-examples", type=int, default=0, help="0 = no limit")
    p.add_argument(
        "--preload",
        action="store_true",
        help="Load all data into RAM (faster per-epoch, but requires memory)",
    )
    p.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use mixed precision on CUDA",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    device = setup_device()
    print(f"Training on {device}")

    files = list_parquet_files(args.data_dir, max_files=int(args.max_files))
    if not files:
        print("No training data found.")
        return 1
    print(f"Found {len(files)} parquet shards")

    max_examples = None if int(args.max_examples) <= 0 else int(args.max_examples)
    pin_memory = device.type == "cuda"

    loader_cfg = LoaderConfig(
        batch_size=int(args.batch_size),
        num_workers=max(0, int(args.num_workers)),
        pin_memory=bool(pin_memory),
        max_examples=max_examples,
    )

    stream = not bool(args.preload)
    loader, dataset_obj = build_dataloader(
        file_paths=files,
        cfg=loader_cfg,
        seed=int(args.seed),
        stream=stream,
    )

    model = load_model(str(args.model_path), device)
    ok = train_epochs(
        model=model,
        loader=loader,
        epochs=int(args.epochs),
        lr=float(args.lr),
        device=device,
        amp=bool(args.amp),
        seed=int(args.seed),
        dataset_obj=dataset_obj,
    )
    if not ok:
        print("Model not saved due to training failure")
        return 2

    save_checkpoint(model, str(args.model_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
