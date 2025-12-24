from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from constants import HEIGHT, IN_CHANNELS, WIDTH
from featurize import featurize_state_json

EXPECTED_X_BYTES = IN_CHANNELS * WIDTH * HEIGHT * 4


def list_parquet_files(data_dir: str, *, max_files: int = 0) -> list[str]:
    root = Path(data_dir)
    if not root.exists():
        return []

    files: list[Path] = []
    for path in root.rglob("*.parquet"):
        # Keep the filter simple and structural.
        parts = set(path.parts)
        if "tmp" in parts or "processed" in parts:
            continue
        if path.is_file():
            files.append(path)

    files = sorted(files)
    if max_files and max_files > 0:
        files = files[: int(max_files)]
    return [str(p) for p in files]


def _decode_states(
    x_col: pl.Series,
    *,
    c: int,
    h: int,
    w: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode the `x` column into a float32 array and validity mask."""
    x_list = x_col.to_list()
    n = len(x_list)

    expected_bytes = int(c) * int(h) * int(w) * 4
    states = np.empty((n, int(c), int(h), int(w)), dtype=np.float32)
    valid = np.ones(n, dtype=bool)

    for i, x in enumerate(x_list):
        if isinstance(x, memoryview):
            x = x.tobytes()
        if not isinstance(x, (bytes, bytearray)) or len(x) != expected_bytes:
            valid[i] = False
            continue
        states[i] = np.frombuffer(x, dtype=np.float32).reshape(int(c), int(h), int(w))

    if valid.all():
        return states, valid
    return states[valid], valid


def _read_parquet_examples(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Preferred format: raw state JSON.
    try:
        df = pl.read_parquet(path, columns=["state", "policy", "value"])
        state_list = df.get_column("state").to_list()
        policies = df.get_column("policy").to_numpy().astype(np.int64, copy=False)
        values = df.get_column("value").to_numpy().astype(np.float32, copy=False)

        xs: list[np.ndarray] = []
        ps: list[np.int64] = []
        vs: list[np.float32] = []
        n_rows = min(len(state_list), int(policies.shape[0]), int(values.shape[0]))
        for i in range(n_rows):
            sb = state_list[i]
            if isinstance(sb, memoryview):
                sb = sb.tobytes()
            if not isinstance(sb, (bytes, bytearray)):
                continue
            try:
                x = featurize_state_json(bytes(sb))
            except (ValueError, TypeError):
                # Bad JSON / unexpected types.
                continue
            if x is None:
                continue
            xs.append(x)
            ps.append(policies[i])
            vs.append(values[i])

        if not xs:
            return (
                np.empty((0, IN_CHANNELS, HEIGHT, WIDTH), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
                np.empty((0,), dtype=np.float32),
            )
        return (
            np.stack(xs, axis=0),
            np.asarray(ps, dtype=np.int64),
            np.asarray(vs, dtype=np.float32),
        )
    except (OSError, ValueError, pl.exceptions.PolarsError):
        # Legacy format: x tensor bytes (optionally with x_c/x_h/x_w).
        try:
            df = pl.read_parquet(path, columns=["x", "policy", "value", "x_c", "x_h", "x_w"])
            c = int(df.get_column("x_c")[0]) if df.height > 0 else IN_CHANNELS
            h = int(df.get_column("x_h")[0]) if df.height > 0 else HEIGHT
            w = int(df.get_column("x_w")[0]) if df.height > 0 else WIDTH
        except (OSError, ValueError, pl.exceptions.PolarsError):
            df = pl.read_parquet(path, columns=["x", "policy", "value"])
            c, h, w = IN_CHANNELS, HEIGHT, WIDTH

        states, valid_mask = _decode_states(df.get_column("x"), c=c, h=h, w=w)
        if states.shape[0] == 0:
            return (
                np.empty((0, int(c), int(h), int(w)), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
                np.empty((0,), dtype=np.float32),
            )

        policies = df.get_column("policy").to_numpy()
        values = df.get_column("value").to_numpy()

        policies = policies[valid_mask].astype(np.int64, copy=False)
        values = values[valid_mask].astype(np.float32, copy=False)

        n = min(states.shape[0], policies.shape[0], values.shape[0])
        return states[:n], policies[:n], values[:n]


class StreamingParquetDataset(torch.utils.data.IterableDataset):
    """Streams parquet shards.

    Shuffling is file-level (per epoch) to keep it simple and memory-safe.
    """

    def __init__(
        self,
        file_paths: list[str],
        *,
        seed: int = 0,
        max_examples: int | None = None,
        shuffle_files: bool = True,
    ):
        super().__init__()
        self.file_paths = list(file_paths)
        self.seed = int(seed)
        self.max_examples = max_examples
        self.shuffle_files = bool(shuffle_files)
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        worker_id = 0 if worker is None else worker.id
        num_workers = 1 if worker is None else worker.num_workers

        files = self.file_paths
        if self.shuffle_files:
            rng = np.random.default_rng(self.seed + self._epoch * 1_000_003)
            files = files[:]
            rng.shuffle(files)

        # Shard by worker.
        files = files[worker_id::num_workers]

        emitted = 0
        for path in files:
            if self.max_examples is not None and emitted >= self.max_examples:
                return

            try:
                states, policies, values = _read_parquet_examples(path)
            except (OSError, ValueError, pl.exceptions.PolarsError) as e:
                print(f"Error reading {path}: {e}")
                continue

            n = states.shape[0]
            for i in range(n):
                if self.max_examples is not None and emitted >= self.max_examples:
                    return
                yield (
                    torch.from_numpy(states[i]),
                    int(policies[i]),
                    float(values[i]),
                )
                emitted += 1

    def __getitem__(self, idx: int):
        raise TypeError("StreamingParquetDataset is an IterableDataset; it does not support random access")


def load_preloaded_dataset(file_paths: list[str], *, max_examples: int | None = None) -> Dataset:
    """Loads all parquet shards into RAM.

    Prefer streaming unless you know the dataset fits comfortably.
    """
    states_chunks: list[np.ndarray] = []
    policy_chunks: list[np.ndarray] = []
    value_chunks: list[np.ndarray] = []

    total = 0
    for path in file_paths:
        try:
            states, policies, values = _read_parquet_examples(path)
        except (OSError, ValueError, pl.exceptions.PolarsError) as e:
            print(f"Error reading {path}: {e}")
            continue

        n = states.shape[0]
        if n == 0:
            continue

        if max_examples is not None and total + n > max_examples:
            take = max_examples - total
            if take <= 0:
                break
            states = states[:take]
            policies = policies[:take]
            values = values[:take]
            n = take

        states_chunks.append(states)
        policy_chunks.append(policies)
        value_chunks.append(values)
        total += n

        if max_examples is not None and total >= max_examples:
            break

    if total == 0:
        return TensorDataset(
            torch.empty((0, IN_CHANNELS, HEIGHT, WIDTH), dtype=torch.float32),
            torch.empty((0,), dtype=torch.long),
            torch.empty((0,), dtype=torch.float32),
        )

    states_all = np.concatenate(states_chunks, axis=0)
    policies_all = np.concatenate(policy_chunks, axis=0)
    values_all = np.concatenate(value_chunks, axis=0)

    return TensorDataset(
        torch.from_numpy(states_all),
        torch.from_numpy(policies_all).to(torch.long),
        torch.from_numpy(values_all).to(torch.float32),
    )


@dataclass(frozen=True)
class LoaderConfig:
    batch_size: int
    num_workers: int
    pin_memory: bool
    max_examples: int | None


def build_dataloader(
    *,
    file_paths: list[str],
    cfg: LoaderConfig,
    seed: int,
    stream: bool,
) -> tuple[Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]], object]:
    """Returns (loader, dataset_like)."""

    if stream:
        dataset = StreamingParquetDataset(
            file_paths,
            seed=seed,
            max_examples=cfg.max_examples,
            shuffle_files=True,
        )
        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            persistent_workers=cfg.num_workers > 0,
            prefetch_factor=4 if cfg.num_workers > 0 else None,
        )
        return loader, dataset

    dataset = load_preloaded_dataset(file_paths, max_examples=cfg.max_examples)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=cfg.pin_memory,
    )
    return loader, dataset
