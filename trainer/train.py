import argparse
import datetime as _dt
import random
import subprocess
import shutil
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl
import torch

from constants import DATA_DIR, DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, DEFAULT_LR, HEIGHT, IN_CHANNELS, WIDTH, MODEL_PATH
from parquet_data import LoaderConfig, build_dataloader, list_parquet_files
from training import load_model, save_checkpoint, setup_device, train_epochs


def _looks_like_archive_parquet(path: str) -> bool:
    try:
        schema = pl.read_parquet_schema(path)
    except (OSError, ValueError, pl.exceptions.PolarsError):
        return False

    cols = set(schema.keys())
    # archive_turn_v1 should have a nested `snakes` column (and food/hazard arrays).
    return "snakes" in cols or "food_x" in cols or "hazard_x" in cols


def _infer_training_x_shape_from_file(path: str) -> tuple[int, int, int] | None:
    """Infer (c,h,w) for tensor-byte parquet formats.

    Returns None for formats that are not tensor-byte shards (e.g. raw state JSON, archive).
    """

    try:
        schema = pl.read_parquet_schema(path)
    except (OSError, ValueError, pl.exceptions.PolarsError):
        return None

    cols = set(schema.keys())
    if "x" not in cols:
        return None

    # Newer tensor-byte format includes explicit x_c/x_h/x_w.
    if {"x_c", "x_h", "x_w"}.issubset(cols):
        try:
            df = pl.read_parquet(path, columns=["x_c", "x_h", "x_w"], n_rows=1)
        except (OSError, ValueError, pl.exceptions.PolarsError):
            return None
        if df.height == 0:
            return None
        return int(df.get_column("x_c")[0]), int(df.get_column("x_h")[0]), int(df.get_column("x_w")[0])

    # Older tensor-byte format uses width/height columns.
    if {"width", "height"}.issubset(cols):
        try:
            df = pl.read_parquet(path, columns=["x", "width", "height"], n_rows=1)
        except (OSError, ValueError, pl.exceptions.PolarsError):
            return None
        if df.height == 0:
            return None

        w = int(df.get_column("width")[0])
        h = int(df.get_column("height")[0])
        if w <= 0 or h <= 0:
            return None

        xb = df.get_column("x")[0]
        if isinstance(xb, memoryview):
            xb = xb.tobytes()
        if not isinstance(xb, (bytes, bytearray)):
            return None
        if len(xb) % 4 != 0:
            return None
        cells = w * h
        floats = len(xb) // 4
        if cells <= 0 or floats % cells != 0:
            return None
        c = floats // cells
        return int(c), int(h), int(w)

    # Fallback assumes standard board size.
    try:
        df = pl.read_parquet(path, columns=["x"], n_rows=1)
    except (OSError, ValueError, pl.exceptions.PolarsError):
        return None
    if df.height == 0:
        return None
    xb = df.get_column("x")[0]
    if isinstance(xb, memoryview):
        xb = xb.tobytes()
    if not isinstance(xb, (bytes, bytearray)):
        return None
    if len(xb) % 4 != 0:
        return None
    cells = int(WIDTH) * int(HEIGHT)
    floats = len(xb) // 4
    if cells <= 0 or floats % cells != 0:
        return None
    c = floats // cells
    return int(c), int(HEIGHT), int(WIDTH)


def _infer_training_x_shape(file_paths: list[str]) -> tuple[int, int, int] | None:
    """Infer a single consistent (c,h,w) across tensor-byte shards.

    If tensor-byte shards with multiple shapes are found, raises a ValueError.
    """

    inferred: tuple[int, int, int] | None = None
    for path in file_paths:
        if _looks_like_archive_parquet(path):
            continue
        shp = _infer_training_x_shape_from_file(path)
        if shp is None:
            # Likely raw-state JSON (state/policy/value).
            continue
        if inferred is None:
            inferred = shp
        elif inferred != shp:
            raise ValueError(f"mixed tensor shapes detected: {inferred} vs {shp} (e.g. in {path})")
    return inferred


def _unique_dest_path(dest: Path) -> Path:
    if not dest.exists():
        return dest
    stem = dest.stem
    suffix = dest.suffix
    parent = dest.parent
    for i in range(1, 10_000):
        cand = parent / f"{stem}.dup{i}{suffix}"
        if not cand.exists():
            return cand
    raise RuntimeError(f"unable to find unique destination for {dest}")


def _move_to_processed(root_dir: Path, files: Iterable[str]) -> int:
    # Prefer the repo-level processed/{scraped|generated} layout so the viewer and
    # other tooling have a single canonical archive location.
    #
    # Older behavior archived to <root>/processed, which created paths like
    # data/scraped/processed. That makes it easy to accidentally treat archived
    # shards as still “active”.
    processed_dir: Path
    root_parts = tuple(root_dir.parts)
    if len(root_parts) >= 2 and root_parts[-2:] == ("data", "scraped"):
        processed_dir = Path("processed") / "scraped"
    elif len(root_parts) >= 2 and root_parts[-2:] == ("data", "generated"):
        processed_dir = Path("processed") / "generated"
    else:
        processed_dir = root_dir / "processed"

    processed_dir.mkdir(parents=True, exist_ok=True)

    moved = 0
    for f in files:
        src = Path(f)

        # Avoid moving symlinks (e.g., orchestrator builds a per-cycle training set
        # via symlinks). We only want to archive the real shards.
        try:
            if src.is_symlink():
                continue
        except OSError:
            continue

        try:
            rel = src.relative_to(root_dir)
        except ValueError:
            # Not under root_dir; skip.
            continue

        # Do not re-process already processed/materialized/tmp.
        parts = set(rel.parts)
        if "processed" in parts or "tmp" in parts or "materialized" in parts:
            continue

        # Keep the same relative layout when archiving from a canonical root.
        dest = processed_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest = _unique_dest_path(dest)

        try:
            shutil.move(str(src), str(dest))
            moved += 1
        except OSError as e:
            print(f"Warning: failed to move {src} -> {dest}: {e}")
            continue

    return moved


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


def _select_default_data_dir(cli_data_dir: str) -> str:
    """Heuristic chooser for `make train` ergonomics.

    If the user did not override `--data-dir` (i.e. it's the default), pick a
    directory that actually contains parquet shards, preferring the newer one.
    """

    def newest_mtime(paths: list[str]) -> float:
        mt = 0.0
        for p in paths:
            try:
                mt = max(mt, Path(p).stat().st_mtime)
            except OSError:
                continue
        return mt

    # Only apply the heuristic when using the default.
    if str(cli_data_dir) != str(DATA_DIR):
        return cli_data_dir

    candidates = [
        str(Path("data/scraped")),
        str(Path("data/generated")),
        str(Path(cli_data_dir)),
    ]
    seen: set[str] = set()
    uniq_candidates: list[str] = []
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        uniq_candidates.append(c)

    best_dir = cli_data_dir
    best_files: list[str] = []
    best_mtime = -1.0

    for d in uniq_candidates:
        files = list_parquet_files(d, max_files=0)
        if not files:
            continue
        mt = newest_mtime(files)
        if mt > best_mtime:
            best_mtime = mt
            best_dir = d
            best_files = files

    if best_files and best_dir != cli_data_dir:
        print(f"Auto-selected data dir: {best_dir} ({len(best_files)} shards)")
    return best_dir


def _default_data_roots(cli_data_dir: str) -> list[Path]:
    """Return one or more dataset roots to train on.

    If the user explicitly passes --data-dir, we honor it.
    Otherwise, we train on both scraped and generated if available.
    """
    if str(cli_data_dir) != str(DATA_DIR):
        return [Path(cli_data_dir)]

    roots = [Path("data/scraped"), Path("data/generated")]
    return [r for r in roots if r.exists()]


def _materialize_archive_if_needed(root: Path, files: list[str]) -> tuple[list[str], list[str]]:
    """Return (train_files, consumed_input_files)."""
    if not files:
        return [], []

    if not _looks_like_archive_parquet(files[0]):
        return files, files

    materialized_dir = root / "materialized"
    materialized_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "go",
        "run",
        "./archive2train",
        "-in-dir",
        str(root),
        "-out-dir",
        str(materialized_dir),
    ]
    print(f"Materializing archive parquet via Go converter: {root}")
    subprocess.run(cmd, check=True)

    train_files = list_parquet_files(str(materialized_dir), max_files=0)
    # consumed inputs are the archive shards.
    return train_files, files


def main() -> int:
    args = parse_args()

    # Avoid CUDA + DataLoader fork deadlocks/hangs.
    # When CUDA is initialized in the parent process, forking worker processes can
    # hang while fetching the first batch. Using 'spawn' starts clean worker
    # processes and is the recommended mode for CUDA workloads.
    if int(args.num_workers) > 0:
        try:
            import torch.multiprocessing as mp

            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            # Start method was already set.
            pass

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    device = setup_device()
    print(f"Training on {device}")

    # Determine which roots to train on.
    roots = _default_data_roots(args.data_dir)
    if not roots:
        args.data_dir = _select_default_data_dir(args.data_dir)
        roots = [Path(args.data_dir)]

    max_files = int(args.max_files)
    train_files: list[str] = []
    consumed_by_root: dict[Path, list[str]] = {}

    for root in roots:
        root_files = list_parquet_files(str(root), max_files=0 if max_files <= 0 else max_files)
        if not root_files:
            continue
        tf, consumed = _materialize_archive_if_needed(root, root_files)
        if tf:
            train_files.extend(tf)
            consumed_by_root[root] = consumed

    if not train_files:
        print("No training data found.")
        return 1

    print(f"Found {len(train_files)} parquet shards")

    # Infer tensor shape early so checkpoint + model match the data.
    try:
        inferred_shape = _infer_training_x_shape(train_files)
    except ValueError as e:
        print(f"Error: {e}")
        print("Tip: avoid mixing old 14-channel shards with current 10-channel data in the same run")
        return 2

    data_c = int(IN_CHANNELS)
    data_h = int(HEIGHT)
    data_w = int(WIDTH)
    if inferred_shape is not None:
        data_c, data_h, data_w = map(int, inferred_shape)

    if data_h != int(HEIGHT) or data_w != int(WIDTH):
        print(f"Error: dataset board size {data_h}x{data_w} does not match model {HEIGHT}x{WIDTH}")
        return 2

    if data_c != int(IN_CHANNELS):
        print(
            f"Warning: dataset has {data_c} input channels but IN_CHANNELS={IN_CHANNELS}. "
            "Trainer will build a model that matches the dataset."
        )

    model_path = Path(args.model_path)
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Checkpoint not found; initializing: {model_path}")
        subprocess.run(
            [
                sys.executable,
                "trainer/init_ckpt.py",
                "--out",
                str(model_path),
                "--in-channels",
                str(int(data_c)),
            ],
            check=True,
        )

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
        file_paths=train_files,
        cfg=loader_cfg,
        seed=int(args.seed),
        stream=stream,
    )

    model = load_model(str(args.model_path), device, width=int(WIDTH), height=int(HEIGHT), in_channels=int(data_c))
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

    # If training is writing to the mutable "models/latest.pt", also archive a
    # timestamped copy into "models/history" for easier manual workflows.
    try:
        mp = Path(args.model_path)
        if mp.name == "latest.pt" and mp.parent.name == "models":
            history_dir = mp.parent / "history"
            history_dir.mkdir(parents=True, exist_ok=True)
            ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%d_%H%M%S")
            hist_path = history_dir / f"model_{ts}.pt"
            shutil.copy2(str(mp), str(hist_path))
            print(f"Archived checkpoint to {hist_path}")
    except OSError as e:
        print(f"Warning: failed to archive checkpoint history: {e}")

    # Mark consumed inputs as processed after a successful save.
    total_moved = 0
    for root, consumed in consumed_by_root.items():
        moved = _move_to_processed(root, consumed)
        if moved:
            print(f"Moved {moved} shards to {root / 'processed'}")
        total_moved += moved
    if total_moved == 0:
        # Not an error; e.g. training from an already-processed directory.
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
