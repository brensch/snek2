import os
import glob
import argparse
import random
import time
from dataclasses import dataclass
import gc

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


def _decode_x_column(x_col) -> np.ndarray:
    """Decode a Polars Binary column of fixed-size float32 planes into (N,14,11,11)."""
    # Polars returns Python objects for Binary (bytes/memoryview).
    x_list = x_col.to_list()
    n = len(x_list)
    states = np.empty((n, 14, 11, 11), dtype=np.float32)
    valid_mask = np.ones(n, dtype=bool)
    for i, x in enumerate(x_list):
        if isinstance(x, memoryview):
            x = x.tobytes()
        if not isinstance(x, (bytes, bytearray)) or len(x) != EXPECTED_X_BYTES:
            valid_mask[i] = False
            continue
        states[i] = np.frombuffer(x, dtype=np.float32).reshape(14, 11, 11)
    if not valid_mask.all():
        states = states[valid_mask]
    return states


def _decode_x_column_with_mask(x_col) -> tuple[np.ndarray, np.ndarray]:
    """Decode x column and return (states, valid_mask).

    valid_mask aligns with the original row order/length of the parquet columns.
    """
    x_list = x_col.to_list()
    n = len(x_list)
    states = np.empty((n, 14, 11, 11), dtype=np.float32)
    valid_mask = np.ones(n, dtype=bool)
    for i, x in enumerate(x_list):
        if isinstance(x, memoryview):
            x = x.tobytes()
        if not isinstance(x, (bytes, bytearray)) or len(x) != EXPECTED_X_BYTES:
            valid_mask[i] = False
            continue
        states[i] = np.frombuffer(x, dtype=np.float32).reshape(14, 11, 11)
    if not valid_mask.all():
        states = states[valid_mask]
    return states, valid_mask


def _parquet_row_count(path: str) -> int:
    try:
        df = pl.scan_parquet(path).select(pl.len().alias("n")).collect()
        return int(df["n"][0])
    except (OSError, ValueError, pl.exceptions.PolarsError):
        # Fallback: if scan/count fails, we will infer counts while reading.
        return 0


def _bytes_per_example(*, states_dtype: torch.dtype, values_dtype: torch.dtype) -> int:
    states_bytes = 14 * 11 * 11 * torch.tensor([], dtype=states_dtype).element_size()
    policy_bytes = torch.tensor([], dtype=torch.long).element_size()
    value_bytes = torch.tensor([], dtype=values_dtype).element_size()
    return int(states_bytes + policy_bytes + value_bytes)


def _select_files_for_example_budget(file_paths: list[str], example_budget: int) -> tuple[list[str], int]:
    """Pick a prefix of files whose *estimated* rows fit within example_budget.

    We avoid partially reading a file (no row offsets), so we stop before a file
    that would exceed the budget. If a file has unknown row count, we include
    just that single file and stop (conservative).

    Returns (selected_files, estimated_examples).
    """
    if example_budget <= 0:
        return ([], 0)
    selected: list[str] = []
    total = 0
    for path in file_paths:
        n = _parquet_row_count(path)
        if n <= 0:
            # Unknown count: include exactly one file and stop to avoid overshoot.
            if not selected:
                selected.append(path)
            break
        if total + n > example_budget:
            break
        selected.append(path)
        total += n
        if total >= example_budget:
            break
    return (selected, total)


def _load_dataset_direct_to_vram(
    file_paths: list[str],
    *,
    max_examples: int | None,
    device: torch.device,
    states_dtype: torch.dtype,
    values_dtype: torch.dtype,
    verbose: bool = True,
) -> "PreloadedSnakeDataset":
    """Load parquet shards into CUDA tensors without building a full CPU-resident dataset."""
    # First pass: estimate rows for preallocation (fast metadata scan).
    estimated = 0
    for path in file_paths:
        if max_examples is not None and estimated >= max_examples:
            break
        n = _parquet_row_count(path)
        if n <= 0:
            continue
        estimated += n
        if max_examples is not None and estimated >= max_examples:
            estimated = max_examples
            break

    if estimated <= 0:
        # If estimation failed, do a conservative small allocation and grow by chunks.
        # Given your dataset sizes, scan_parquet should usually work.
        estimated = max_examples or 0

    if max_examples is not None:
        estimated = min(estimated, max_examples)

    if estimated <= 0:
        return PreloadedSnakeDataset(
            [],
            states=torch.empty((0, 14, 11, 11), device=device, dtype=states_dtype),
            policies=torch.empty((0,), device=device, dtype=torch.long),
            values=torch.empty((0,), device=device, dtype=values_dtype),
        )

    states_gpu = torch.empty((estimated, 14, 11, 11), device=device, dtype=states_dtype)
    policies_gpu = torch.empty((estimated,), device=device, dtype=torch.long)
    values_gpu = torch.empty((estimated,), device=device, dtype=values_dtype)

    write = 0
    for path in file_paths:
        if max_examples is not None and write >= max_examples:
            break
        try:
            df = pl.read_parquet(path, columns=["x", "policy", "value"])
        except (OSError, ValueError, pl.exceptions.PolarsError) as e:
            print(f"Error reading {path}: {e}")
            continue

        states_np, valid_mask = _decode_x_column_with_mask(df.get_column("x"))
        if states_np.shape[0] == 0:
            continue

        policies_np = df.get_column("policy").to_numpy()
        values_np = df.get_column("value").to_numpy()

        # Filter policies/values with the same valid_mask (align rows correctly).
        # If shapes don't line up for any reason, fall back to min-length truncation.
        if getattr(valid_mask, "shape", None) is not None and valid_mask.shape[0] == policies_np.shape[0] == values_np.shape[0]:
            policies_np = policies_np[valid_mask]
            values_np = values_np[valid_mask]
        else:
            n = min(states_np.shape[0], policies_np.shape[0], values_np.shape[0])
            states_np = states_np[:n]
            policies_np = policies_np[:n]
            values_np = values_np[:n]

        n = min(states_np.shape[0], policies_np.shape[0], values_np.shape[0])
        if n <= 0:
            continue

        if max_examples is not None:
            n = min(n, max_examples - write)
            if n <= 0:
                break

        # Host->device copy for this shard.
        states_cpu = torch.from_numpy(states_np[:n])
        if states_dtype == torch.float16:
            states_cpu = states_cpu.to(torch.float16)
        policies_cpu = torch.from_numpy(policies_np[:n].astype(np.int64, copy=False)).to(torch.long)
        values_cpu = torch.from_numpy(values_np[:n].astype(np.float32, copy=False))

        states_gpu[write : write + n].copy_(states_cpu, non_blocking=False)
        policies_gpu[write : write + n].copy_(policies_cpu, non_blocking=False)
        values_gpu[write : write + n].copy_(values_cpu, non_blocking=False)
        write += n

        if verbose and write and (write % (256 * 1024) == 0):
            print(f"Loaded {write} examples into VRAM...")

    if write == 0:
        return PreloadedSnakeDataset(
            [],
            states=torch.empty((0, 14, 11, 11), device=device, dtype=states_dtype),
            policies=torch.empty((0,), device=device, dtype=torch.long),
            values=torch.empty((0,), device=device, dtype=values_dtype),
        )

    if write < estimated:
        states_gpu = states_gpu[:write]
        policies_gpu = policies_gpu[:write]
        values_gpu = values_gpu[:write]

    return PreloadedSnakeDataset([], states=states_gpu, policies=policies_gpu, values=values_gpu)


class PreloadedSnakeDataset(Dataset):
    """Preload all examples into contiguous CPU tensors.

    This avoids Python list-of-tuples overhead and per-sample bytes->numpy conversion.
    """

    def __init__(
        self,
        file_paths,
        max_examples: int | None = None,
        *,
        states: torch.Tensor | None = None,
        policies: torch.Tensor | None = None,
        values: torch.Tensor | None = None,
    ):
        if states is not None or policies is not None or values is not None:
            if states is None or policies is None or values is None:
                raise ValueError("states/policies/values must all be provided together")
            self.states = states
            self.policies = policies
            self.values = values
            return

        states_chunks: list[np.ndarray] = []
        policies_chunks: list[np.ndarray] = []
        values_chunks: list[np.ndarray] = []
        total = 0
        for path in file_paths:
            try:
                df = pl.read_parquet(path, columns=["x", "policy", "value"])
            except (OSError, ValueError, pl.exceptions.PolarsError) as e:
                print(f"Error reading {path}: {e}")
                continue

            states, valid_mask = _decode_x_column_with_mask(df.get_column("x"))
            if states.shape[0] == 0:
                continue

            # Filter policy/value with the same validity mask to keep rows aligned.
            policies = df.get_column("policy").to_numpy()[valid_mask]
            values = df.get_column("value").to_numpy()[valid_mask]
            n = min(states.shape[0], policies.shape[0], values.shape[0])
            if n <= 0:
                continue

            states = states[:n]
            policies = policies[:n].astype(np.int64, copy=False)
            values = values[:n].astype(np.float32, copy=False)

            if max_examples is not None and total + n > max_examples:
                take = max_examples - total
                if take <= 0:
                    break
                states = states[:take]
                policies = policies[:take]
                values = values[:take]
                n = take

            states_chunks.append(states)
            policies_chunks.append(policies)
            values_chunks.append(values)
            total += n
            if max_examples is not None and total >= max_examples:
                break

        if total == 0:
            self.states = torch.empty((0, 14, 11, 11), dtype=torch.float32)
            self.policies = torch.empty((0,), dtype=torch.long)
            self.values = torch.empty((0,), dtype=torch.float32)
            return

        states_all = np.concatenate(states_chunks, axis=0)
        policies_all = np.concatenate(policies_chunks, axis=0)
        values_all = np.concatenate(values_chunks, axis=0)

        # Store as torch tensors once; indexing becomes very cheap.
        self.states = torch.from_numpy(states_all)
        self.policies = torch.from_numpy(policies_all).to(torch.long)
        self.values = torch.from_numpy(values_all).to(torch.float32)

    def __len__(self):
        return int(self.states.shape[0])

    def __getitem__(self, idx):
        return self.states[idx], self.policies[idx], self.values[idx]


class StreamingParquetDataset(torch.utils.data.IterableDataset):
    """Stream examples by reading parquet files inside DataLoader workers.

    This avoids loading everything into RAM and parallelizes parquet decoding.
    Shuffling is approximate: files are shuffled each epoch; rows are streamed.
    """

    def __init__(
        self,
        file_paths: list[str],
        *,
        epoch_seed: int = 0,
        max_examples: int | None = None,
        shuffle_files: bool = True,
    ):
        super().__init__()
        self.file_paths = list(file_paths)
        self.epoch_seed = int(epoch_seed)
        self.epoch = 0
        self.max_examples = max_examples
        self.shuffle_files = shuffle_files

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        if worker is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker.id
            num_workers = worker.num_workers

        # Deterministic per-epoch shuffle of files, then shard across workers.
        files = self.file_paths
        if self.shuffle_files:
            rng = random.Random(self.epoch_seed + self.epoch * 1_000_003 + 1337)
            files = files[:]
            rng.shuffle(files)

        files = files[worker_id::num_workers]
        emitted = 0

        for path in files:
            if self.max_examples is not None and emitted >= self.max_examples:
                return
            try:
                df = pl.read_parquet(path, columns=["x", "policy", "value"])
            except (OSError, ValueError, pl.exceptions.PolarsError) as e:
                print(f"Error reading {path}: {e}")
                continue

            states, valid_mask = _decode_x_column_with_mask(df.get_column("x"))
            if states.shape[0] == 0:
                continue

            policies = df.get_column("policy").to_numpy()[valid_mask].astype(np.int64, copy=False)
            values = df.get_column("value").to_numpy()[valid_mask].astype(np.float32, copy=False)

            n = min(states.shape[0], policies.shape[0], values.shape[0])
            if n <= 0:
                continue
            states = states[:n]
            policies = policies[:n]
            values = values[:n]

            for i in range(n):
                if self.max_examples is not None and emitted >= self.max_examples:
                    return
                # Yield numpy-backed tensors; DataLoader will collate.
                yield (
                    torch.from_numpy(states[i]),
                    int(policies[i]),
                    float(values[i]),
                )
                emitted += 1

    def __getitem__(self, idx):
        raise TypeError("StreamingParquetDataset is an IterableDataset; it does not support random access")

@dataclass(frozen=True)
class DataConfig:
    stream: bool
    num_workers: int
    prefetch_factor: int
    persistent_workers: bool
    pin_memory: bool
    max_examples: int | None
    epoch_seed: int

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--stream", action="store_true", help="Stream parquet files instead of preloading into RAM")
    parser.add_argument(
        "--dataset-in-vram",
        action="store_true",
        help="Preload the full dataset onto CUDA VRAM (disables --stream; ignores --num-workers)",
    )
    parser.add_argument(
        "--direct-vram-load",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When using --dataset-in-vram, load parquet shards directly into VRAM (minimizes CPU RAM use)",
    )
    parser.add_argument(
        "--vram-fp16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store states in VRAM as float16 (recommended with AMP)",
    )
    parser.add_argument(
        "--vram-util",
        type=float,
        default=0.85,
        help="Target fraction of currently-free VRAM to use for dataset tensors (0..1)",
    )
    parser.add_argument(
        "--vram-chunked",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When using --dataset-in-vram, iterate all files by loading VRAM-sized chunks sequentially",
    )
    parser.add_argument(
        "--no-vram-shuffle",
        action="store_true",
        help="Disable epoch-level in-VRAM shuffling (uses sequential batches)",
    )
    parser.add_argument(
        "--cache-path",
        default="",
        help="Optional path to a tensor cache (.pt) to avoid re-parsing many parquet shards",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Rebuild --cache-path even if it exists",
    )
    parser.add_argument(
        "--cache-fp16",
        action="store_true",
        help="Store cached states/values as float16 (smaller/faster IO; cast back during training)",
    )
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--no-persistent-workers", action="store_true")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision on CUDA")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-files", type=int, default=0, help="0 = no limit")
    parser.add_argument("--max-examples", type=int, default=0, help="0 = no limit")
    return parser.parse_args()


def _maybe_load_cached_dataset(cache_path: str):
    if not cache_path:
        return None
    if not os.path.exists(cache_path):
        return None
    obj = torch.load(cache_path, map_location="cpu")
    if not isinstance(obj, dict):
        return None
    if not all(k in obj for k in ("states", "policies", "values")):
        return None
    return PreloadedSnakeDataset(
        [],
        states=obj["states"],
        policies=obj["policies"],
        values=obj["values"],
    )


def _maybe_save_cached_dataset(cache_path: str, dataset: PreloadedSnakeDataset, *, fp16: bool) -> None:
    if not cache_path:
        return
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    states = dataset.states
    if fp16:
        states = states.to(torch.float16)
    # Keep values as fp32 to avoid fp16 MSE instability.
    torch.save({"states": states, "policies": dataset.policies, "values": dataset.values.to(torch.float32)}, cache_path)


def train():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except (AttributeError, RuntimeError):
            pass

    # Load Data
    # Allow nested layout (e.g. data/generated/*.parquet and data/scraped/*.parquet)
    # and avoid partially-written tmp shards.
    files = glob.glob(os.path.join(args.data_dir, "**", "*.parquet"), recursive=True)
    files = [
        f
        for f in files
        if os.path.isfile(f)
        and ("/tmp/" not in f.replace("\\", "/"))
        and ("/generated/tmp/" not in f.replace("\\", "/"))
        and ("/scraped/tmp/" not in f.replace("\\", "/"))
        and ("/processed/" not in f.replace("\\", "/"))
    ]
    files.sort()
    if not files:
        print("No training data found.")
        return

    if args.max_files and args.max_files > 0:
        files = files[: args.max_files]

    print(f"Found {len(files)} data files.")
    max_examples = None if not args.max_examples or args.max_examples <= 0 else args.max_examples

    if args.dataset_in_vram and not torch.cuda.is_available():
        print("--dataset-in-vram requires CUDA; falling back to CPU preload")
        args.dataset_in_vram = False

    if args.dataset_in_vram and args.stream:
        print("--dataset-in-vram disables --stream")
        args.stream = False

    pin_memory = torch.cuda.is_available() and (not args.dataset_in_vram)
    amp_enabled = torch.cuda.is_available() and (args.amp or not args.no_amp)

    if args.no_persistent_workers:
        persistent_workers = False
    elif args.persistent_workers:
        persistent_workers = True
    else:
        # Default: persistent workers help a lot in streaming mode.
        persistent_workers = bool(args.stream and args.num_workers > 0)

    data_cfg = DataConfig(
        stream=bool(args.stream),
        num_workers=int(args.num_workers),
        prefetch_factor=max(1, int(args.prefetch_factor)),
        persistent_workers=bool(persistent_workers),
        pin_memory=bool(pin_memory),
        max_examples=max_examples,
        epoch_seed=int(args.seed),
    )

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

    # If the checkpoint (or current weights) contain NaN/Inf, don't keep training/saving garbage.
    with torch.no_grad():
        nonfinite = False
        for p in model.parameters():
            if not torch.isfinite(p).all():
                nonfinite = True
                break
        if nonfinite:
            print("Warning: model parameters contain NaN/Inf; reinitializing model weights")
            model = EgoSnakeNet(width=11, height=11).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Loss Functions
    # Policy: Cross Entropy over logits
    # Value: MSE
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    model.train()
    dataset_on_vram = False
    vram_shuffle = bool(args.dataset_in_vram and (not args.no_vram_shuffle))
    dataloader = None

    def _compute_vram_example_cap() -> tuple[int, int, torch.dtype, torch.dtype]:
        states_dtype = torch.float16 if (amp_enabled and args.vram_fp16) else torch.float32
        # Keep value targets in fp32 to avoid fp16 under/overflow in MSE.
        values_dtype = torch.float32

        free_bytes, total_bytes = torch.cuda.mem_get_info()
        util = float(args.vram_util)
        if not (0.0 < util <= 1.0):
            util = 0.85

        baseline_alloc = int(torch.cuda.memory_allocated())
        baseline_resv = int(torch.cuda.memory_reserved())
        safety_bytes = int(0.10 * total_bytes) + int(0.50 * 1024**3)  # 10% + 0.5GiB
        budget_bytes = max(0, int(free_bytes * util) - safety_bytes)
        bpe = _bytes_per_example(states_dtype=states_dtype, values_dtype=values_dtype)
        cap = budget_bytes // max(1, bpe)

        # Respect user --max-examples if provided.
        if max_examples is not None:
            cap = min(int(cap), int(max_examples))

        if cap <= 0:
            print(
                "Not enough free VRAM for dataset tensors; falling back to streaming mode. "
                f"free={free_bytes/1e9:.2f}GB reserved={baseline_resv/1e9:.2f}GB alloc={baseline_alloc/1e9:.2f}GB"
            )
        return int(cap), int(budget_bytes), states_dtype, values_dtype

    if args.dataset_in_vram:
        # VRAM mode: by default we avoid a full CPU-resident dataset by loading shards directly into VRAM.
        dataset = None
        if args.cache_path and not args.rebuild_cache:
            dataset = _maybe_load_cached_dataset(args.cache_path)
            if dataset is not None:
                print(f"Loaded cached dataset: {args.cache_path}")

        # If a .pt cache is used, it necessarily loads into CPU RAM first (torch.load).
        # For minimum RAM usage, skip --cache-path and rely on direct VRAM load.

        if dataset is None and args.direct_vram_load:
            cap_examples, budget_bytes, states_dtype, values_dtype = _compute_vram_example_cap()
            if cap_examples <= 0:
                # fall back to streaming below
                pass
            elif args.vram_chunked:
                # Chunked VRAM mode loads data per-epoch/per-chunk. Avoid preloading anything here.
                print(
                    "VRAM-chunked mode enabled: "
                    f"cap ~{cap_examples} examples per chunk (budget ~{budget_bytes/1e9:.2f}GB)"
                )
                dataset = PreloadedSnakeDataset(
                    [],
                    states=torch.empty((0, 14, 11, 11), device=device, dtype=states_dtype),
                    policies=torch.empty((0,), device=device, dtype=torch.long),
                    values=torch.empty((0,), device=device, dtype=values_dtype),
                )
                dataset_on_vram = False
                dataloader = None
            else:
                selected_files, est_examples = _select_files_for_example_budget(files, int(cap_examples))
                print(
                    "Direct-loading dataset into VRAM (you may see small transient CPU RAM per shard)...\n"
                    f"VRAM budget: ~{budget_bytes/1e9:.2f}GB -> cap ~{cap_examples} examples\n"
                    f"Selecting {len(selected_files)}/{len(files)} files (est {est_examples} rows)"
                )
                dataset = _load_dataset_direct_to_vram(
                    selected_files,
                    max_examples=int(cap_examples),
                    device=device,
                    states_dtype=states_dtype,
                    values_dtype=values_dtype,
                    verbose=False,
                )
                dataset_on_vram = True
                dataloader = None
                torch.cuda.synchronize()
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                print(f"CUDA after load: allocated={allocated:.2f}GB reserved={reserved:.2f}GB")

        if dataset is None:
            # Fallback: CPU preload (optionally cached), then move.
            dataset = PreloadedSnakeDataset(files, max_examples=max_examples)
            if len(dataset) != 0 and args.cache_path:
                _maybe_save_cached_dataset(args.cache_path, dataset, fp16=bool(args.cache_fp16))
                print(f"Wrote cached dataset: {args.cache_path}")
            if len(dataset) == 0:
                print("No training data found.")
                return

        if (not dataset_on_vram) and (not (args.vram_chunked and args.direct_vram_load)):
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            dataset_bytes = (
                dataset.states.numel() * dataset.states.element_size()
                + dataset.policies.numel() * dataset.policies.element_size()
                + dataset.values.numel() * dataset.values.element_size()
            )

            # Keep some headroom for activations/optimizer buffers.
            headroom = int(0.20 * total_bytes)
            if dataset_bytes + headroom > free_bytes:
                print(
                    "Dataset may not fit in free VRAM; "
                    f"need ~{dataset_bytes/1e9:.2f}GB (+headroom), free ~{free_bytes/1e9:.2f}GB. "
                    "Falling back to streaming mode."
                )
                data_cfg = DataConfig(
                    stream=True,
                    num_workers=int(args.num_workers),
                    prefetch_factor=max(1, int(args.prefetch_factor)),
                    persistent_workers=bool(persistent_workers),
                    pin_memory=True,
                    max_examples=max_examples,
                    epoch_seed=int(args.seed),
                )
                dataset = StreamingParquetDataset(
                    files,
                    epoch_seed=data_cfg.epoch_seed,
                    max_examples=data_cfg.max_examples,
                    shuffle_files=True,
                )
                dataloader = DataLoader(
                    dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=data_cfg.num_workers,
                    pin_memory=data_cfg.pin_memory,
                    persistent_workers=data_cfg.persistent_workers,
                    prefetch_factor=data_cfg.prefetch_factor if data_cfg.num_workers > 0 else None,
                )
            else:
                print(f"Moving dataset to VRAM: ~{dataset_bytes/1e9:.2f}GB (free ~{free_bytes/1e9:.2f}GB)")
                dataset = PreloadedSnakeDataset(
                    [],
                    states=dataset.states.to(device),
                    policies=dataset.policies.to(device),
                    values=dataset.values.to(device),
                )
                dataset_on_vram = True
                dataloader = None
                # Encourage CPU RAM to drop after the move.
                gc.collect()
                torch.cuda.synchronize()
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                print(f"CUDA after move: allocated={allocated:.2f}GB reserved={reserved:.2f}GB")
    elif data_cfg.stream:
        dataset = StreamingParquetDataset(
            files,
            epoch_seed=data_cfg.epoch_seed,
            max_examples=data_cfg.max_examples,
            shuffle_files=True,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=data_cfg.num_workers,
            pin_memory=data_cfg.pin_memory,
            persistent_workers=data_cfg.persistent_workers,
            prefetch_factor=data_cfg.prefetch_factor if data_cfg.num_workers > 0 else None,
        )
    else:
        if data_cfg.num_workers != 0:
            print("Note: preload mode ignores --num-workers; using 0 to avoid duplicating RAM in workers")
        dataset = None
        if args.cache_path and not args.rebuild_cache:
            dataset = _maybe_load_cached_dataset(args.cache_path)
            if dataset is not None:
                print(f"Loaded cached dataset: {args.cache_path}")
        if dataset is None:
            dataset = PreloadedSnakeDataset(files, max_examples=max_examples)
            if len(dataset) != 0 and args.cache_path:
                _maybe_save_cached_dataset(args.cache_path, dataset, fp16=bool(args.cache_fp16))
                print(f"Wrote cached dataset: {args.cache_path}")
        if len(dataset) == 0:
            print("No training data found.")
            return
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=data_cfg.pin_memory,
            persistent_workers=False,
        )

    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    had_nonfinite = False

    for epoch in range(args.epochs):
        if (not dataset_on_vram) and data_cfg.stream and hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)

        total_loss = 0.0
        total_batches = 0

        t0 = time.time()

        if args.dataset_in_vram and args.direct_vram_load and args.vram_chunked:
            # Chunked VRAM epoch: iterate over all files by repeatedly loading a chunk into VRAM.
            cap_examples, budget_bytes, states_dtype, values_dtype = _compute_vram_example_cap()
            if cap_examples <= 0:
                raise RuntimeError("VRAM chunked mode has no capacity; reduce model/batch or use --stream")

            rng = random.Random(int(args.seed) + epoch * 1_000_003)
            epoch_files = files[:]
            rng.shuffle(epoch_files)

            file_idx = 0
            while file_idx < len(epoch_files):
                chunk_files, est_examples = _select_files_for_example_budget(epoch_files[file_idx:], int(cap_examples))
                if not chunk_files:
                    # Shouldn't happen, but avoid infinite loop.
                    chunk_files = [epoch_files[file_idx]]

                print(
                    f"Epoch {epoch+1}/{args.epochs} VRAM-chunk: loading {len(chunk_files)} files "
                    f"(est {est_examples} rows) [{file_idx+1}/{len(epoch_files)}...]"
                )

                chunk_ds = _load_dataset_direct_to_vram(
                    chunk_files,
                    max_examples=None,
                    device=device,
                    states_dtype=states_dtype,
                    values_dtype=values_dtype,
                    verbose=False,
                )

                # Train over this chunk once.
                n = len(chunk_ds)
                if n > 0:
                    g = torch.Generator(device=device)
                    g.manual_seed(int(args.seed) + epoch + file_idx)
                    perm = torch.randperm(n, generator=g, device=device) if vram_shuffle else None

                    for start in range(0, n, args.batch_size):
                        if vram_shuffle:
                            idx = perm[start : start + args.batch_size]
                            states = chunk_ds.states.index_select(0, idx)
                            policies = chunk_ds.policies.index_select(0, idx)
                            values = chunk_ds.values.index_select(0, idx)
                        else:
                            states = chunk_ds.states[start : start + args.batch_size]
                            policies = chunk_ds.policies[start : start + args.batch_size]
                            values = chunk_ds.values[start : start + args.batch_size]

                        optimizer.zero_grad(set_to_none=True)
                        with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
                            pred_policies, pred_values = model(states)
                            loss_policy = ce_loss(pred_policies, policies)
                            loss_value = mse_loss(pred_values.squeeze(1).float(), values.float())
                            loss = loss_policy + loss_value

                        if not torch.isfinite(loss).all():
                            had_nonfinite = True
                            print("Non-finite loss detected; aborting epoch to avoid corrupting checkpoint")
                            break

                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                        total_loss += float(loss.item())
                        total_batches += 1

                    if had_nonfinite:
                        break

                # Free VRAM for next chunk.
                del chunk_ds
                torch.cuda.empty_cache()

                if had_nonfinite:
                    break

                file_idx += len(chunk_files)

        elif dataset_on_vram:
            assert isinstance(dataset, PreloadedSnakeDataset)
            n = len(dataset)
            states_all = dataset.states
            policies_all = dataset.policies
            values_all = dataset.values
            if vram_shuffle:
                g = torch.Generator(device=device)
                g.manual_seed(int(args.seed) + epoch)
                perm = torch.randperm(n, generator=g, device=device)

            for start in range(0, n, args.batch_size):
                if vram_shuffle:
                    idx = perm[start : start + args.batch_size]
                    states = states_all.index_select(0, idx)
                    policies = policies_all.index_select(0, idx)
                    values = values_all.index_select(0, idx)
                else:
                    states = states_all[start : start + args.batch_size]
                    policies = policies_all[start : start + args.batch_size]
                    values = values_all[start : start + args.batch_size]

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
                    pred_policies, pred_values = model(states)

                    loss_policy = ce_loss(pred_policies, policies)
                    loss_value = mse_loss(pred_values.squeeze(1).float(), values.float())
                    loss = loss_policy + loss_value

                if not torch.isfinite(loss).all():
                    had_nonfinite = True
                    print("Non-finite loss detected; aborting epoch to avoid corrupting checkpoint")
                    break

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += float(loss.item())
                total_batches += 1

            if had_nonfinite:
                break
        else:
            if dataloader is None:
                raise RuntimeError("Internal error: dataloader is None while dataset_on_vram is False")
            for states, policies, values in dataloader:
                # Fast host->device copy when pin_memory=True
                states = states.to(device, non_blocking=True)
                policies = policies.to(device, non_blocking=True, dtype=torch.long)
                values = values.to(device, non_blocking=True, dtype=torch.float32)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
                    pred_policies, pred_values = model(states)

                    loss_policy = ce_loss(pred_policies, policies)
                    loss_value = mse_loss(pred_values.squeeze(1).float(), values.float())
                    loss = loss_policy + loss_value

                if not torch.isfinite(loss).all():
                    had_nonfinite = True
                    print("Non-finite loss detected; aborting epoch to avoid corrupting checkpoint")
                    break
 
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += float(loss.item())
                total_batches += 1

            if had_nonfinite:
                break

        if had_nonfinite:
            break
        if total_batches == 0:
            print(f"Epoch {epoch+1}/{args.epochs}, no batches (no data)")
            break
        dt = max(1e-9, time.time() - t0)
        print(
            f"Epoch {epoch+1}/{args.epochs}, Loss: {total_loss / total_batches:.4f} "
            f"({total_batches / dt:.2f} batches/s)"
        )

    # Save Model
    if not had_nonfinite:
        os.makedirs(os.path.dirname(args.model_path) or ".", exist_ok=True)
        torch.save(model.state_dict(), args.model_path)
        print(f"Model saved to {args.model_path}")
    else:
        print("Model not saved due to non-finite loss")

    # Cleanup Data?
    # For now, keep it. In a real loop, we might archive it.

if __name__ == '__main__':
    train()
