#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
import numpy as np
import torch

# Allow importing sibling scripts without packaging.
sys.path.append(os.path.dirname(__file__))

from bench_next_state import (  # noqa: E402
    MOVE_DOWN,
    MOVE_LEFT,
    MOVE_RIGHT,
    MOVE_UP,
    GameState,
    SnakeState,
    encode_10_planes_into,
    get_legal_moves,
    next_state,
    _make_random_state,
)

from model import EgoSnakeNet  # noqa: E402


@dataclass(slots=True)
class PredictRequest:
    state: GameState
    done: threading.Event
    policy_logits: Optional[np.ndarray] = None  # shape (4,), float32
    value: Optional[float] = None
    err: Optional[BaseException] = None


class TorchBatchPredictor:
    """Batches predict() calls onto one GPU worker.

    This matches the Go OnnxClient pattern: synchronous per-caller API, internally
    queued + batched.

    encode_mode:
      - cpu10: encode to float32 on CPU, stage through pinned batch, then H2D
      - gpu_sparse: build inputs directly on GPU from sparse indices (avoids H2D planes)
    """

    def __init__(
        self,
        *,
        device: str,
        batch_size: int,
        batch_timeout_ms: float,
        encode_mode: str,
        compile_model: bool,
    ) -> None:
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.batch_timeout_s = float(batch_timeout_ms) / 1000.0
        self.encode_mode = str(encode_mode)

        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        self.model = EgoSnakeNet(width=11, height=11, in_channels=10).to(self.device).eval()
        if compile_model:
            self.model = torch.compile(self.model, mode="reduce-overhead")

        # Pinned staging for cpu10 path
        self._pinned = torch.empty(
            (self.batch_size, 10, 11, 11), dtype=torch.float32, pin_memory=True
        )

        self._q: "queue.Queue[PredictRequest]" = queue.Queue(maxsize=self.batch_size * 8)
        self._stop = threading.Event()

        # Stats
        self._lock = threading.Lock()
        self.total_items = 0
        self.total_batches = 0
        self.total_run_s = 0.0
        self.last_batch = 0

        self._thr = threading.Thread(target=self._loop, name="TorchBatchPredictor", daemon=True)
        self._thr.start()

    def close(self) -> None:
        self._stop.set()
        self._thr.join(timeout=2.0)

    def stats(self) -> dict:
        with self._lock:
            batches = int(self.total_batches)
            items = int(self.total_items)
            run_s = float(self.total_run_s)
            last = int(self.last_batch)
        avg_batch = (items / batches) if batches else 0.0
        avg_ms = (run_s * 1000.0 / batches) if batches else 0.0
        return {
            "total_batches": batches,
            "total_items": items,
            "avg_batch": avg_batch,
            "avg_run_ms": avg_ms,
            "last_batch": last,
            "queue": self._q.qsize(),
        }

    def predict(self, state: GameState) -> Tuple[np.ndarray, float]:
        req = PredictRequest(state=state, done=threading.Event())
        self._q.put(req)
        req.done.wait()
        if req.err is not None:
            raise req.err
        assert req.policy_logits is not None
        assert req.value is not None
        return req.policy_logits, float(req.value)

    def predict_many(self, states: Sequence[GameState]) -> Tuple[np.ndarray, np.ndarray]:
        """Predict a batch by enqueueing all requests first, then waiting.

        Returns:
          policy_logits: (B,4) float32
          values: (B,) float32
        """

        reqs = [PredictRequest(state=s, done=threading.Event()) for s in states]
        for r in reqs:
            self._q.put(r)
        for r in reqs:
            r.done.wait()
            if r.err is not None:
                raise r.err

        policy = np.stack([r.policy_logits for r in reqs], axis=0).astype(np.float32, copy=False)  # type: ignore[arg-type]
        values = np.asarray([r.value for r in reqs], dtype=np.float32)
        return policy, values

    @torch.inference_mode()
    def _loop(self) -> None:
        torch.set_num_threads(1)
        while not self._stop.is_set():
            try:
                first = self._q.get(timeout=0.05)
            except queue.Empty:
                continue

            batch: List[PredictRequest] = [first]
            t0 = time.perf_counter()
            t_deadline = t0 + self.batch_timeout_s

            while len(batch) < self.batch_size:
                remaining = t_deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    batch.append(self._q.get(timeout=remaining))
                except queue.Empty:
                    break

            k = len(batch)

            try:
                if self.encode_mode == "cpu10":
                    # Encode into pinned batch tensor via numpy view (no extra Python lists).
                    # Note: torch CPU tensor -> numpy is a view; writes go into pinned storage.
                    pinned_np = self._pinned[:k].numpy()
                    for i, req in enumerate(batch):
                        encode_10_planes_into(req.state, pinned_np[i])

                    x = self._pinned[:k].to(self.device, non_blocking=True)

                elif self.encode_mode == "gpu_sparse":
                    x = self._encode10_gpu_sparse([r.state for r in batch])

                else:
                    raise ValueError(f"unknown encode_mode={self.encode_mode}")

                start_run = time.perf_counter()
                policy, value = self.model(x)
                torch.cuda.synchronize(self.device) if self.device.type == "cuda" else None
                run_s = time.perf_counter() - start_run

                # Move back to CPU once per batch.
                policy_cpu = policy.float().detach().cpu().numpy()
                value_cpu = value.float().detach().cpu().numpy().reshape(-1)

                for i, req in enumerate(batch):
                    req.policy_logits = policy_cpu[i].astype(np.float32, copy=False)
                    req.value = float(value_cpu[i])
                    req.done.set()

                with self._lock:
                    self.total_batches += 1
                    self.total_items += k
                    self.total_run_s += run_s
                    self.last_batch = k

            except BaseException as e:
                for req in batch:
                    req.err = e
                    req.done.set()

    def _encode10_gpu_sparse(self, states: Sequence[GameState]) -> torch.Tensor:
        # Build (B,10,11,11) directly on GPU from sparse indices.
        # This avoids transferring full 10x11x11 planes from host.
        device = self.device
        bsz = len(states)
        x = torch.zeros((bsz, 10, 11, 11), device=device, dtype=torch.float32)

        for b, st in enumerate(states):
            if st.food:
                idx = torch.as_tensor(st.food, device=device, dtype=torch.int64)
                x[b, 0].view(-1).index_fill_(0, idx, 1.0)
            if st.hazards:
                idx = torch.as_tensor(st.hazards, device=device, dtype=torch.int64)
                x[b, 1].view(-1).index_fill_(0, idx, 1.0)

            # Order: ego first then enemies by id (same as bench_next_state encoder)
            snakes: List[SnakeState] = []
            ego = st.snakes[int(st.you_index)]
            if ego.health > 0 and ego.length > 0:
                snakes.append(ego)
            enemies = [
                s
                for i, s in enumerate(st.snakes)
                if i != int(st.you_index) and s.health > 0 and s.length > 0
            ]
            enemies.sort(key=lambda s: s.id_bytes)
            snakes.extend(enemies)
            if len(snakes) > 4:
                snakes = snakes[:4]

            for i, s in enumerate(snakes):
                ttl_c = 2 + 2 * i
                h_c = ttl_c + 1

                health = float(s.health) / 100.0
                x[b, h_c].fill_(health)

                body = s.body
                L = int(body.shape[0])
                if L <= 0:
                    continue

                idx = torch.as_tensor(body, device=device, dtype=torch.int64)
                ttls = torch.arange(L, 0, -1, device=device, dtype=torch.float32)
                plane = x[b, ttl_c].view(-1)
                plane.scatter_reduce_(0, idx, ttls, reduce="amax", include_self=True)

        return x


class AtomicCounter:
    __slots__ = ("_lock", "_value")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value = 0

    def add(self, n: int) -> None:
        if n == 0:
            return
        with self._lock:
            self._value += int(n)

    def value(self) -> int:
        with self._lock:
            return int(self._value)


class Node:
    __slots__ = ("state", "prior", "children", "visit", "value_sum", "expanded", "inflight")

    def __init__(self, state: GameState, prior: float) -> None:
        self.state = state
        self.prior = float(prior)
        self.children: List[Optional[Node]] = [None, None, None, None]
        self.visit = 0
        self.value_sum = 0.0
        self.expanded = False
        self.inflight = 0

    def q(self) -> float:
        return (self.value_sum / self.visit) if self.visit else 0.0


def softmax4(logits: np.ndarray) -> np.ndarray:
    x = logits.astype(np.float32, copy=False)
    m = float(x.max())
    e = np.exp(x - m, dtype=np.float32)
    s = float(e.sum())
    return (e / s) if s > 0 else np.zeros((4,), dtype=np.float32)


def mcts_search(
    *,
    root_state: GameState,
    predictor: TorchBatchPredictor,
    simulations: int,
    cpuct: float,
    minimum_food: int,
    food_spawn_chance: int,
    eval_batch: int,
) -> int:
    root = Node(root_state, prior=1.0)

    sims_left = int(simulations)
    eb = int(eval_batch)
    if eb <= 0:
        eb = 1

    while sims_left > 0:
        # If the root is not expanded yet, we can only evaluate it once.
        if not root.expanded:
            legal = get_legal_moves(root.state)
            if not legal:
                root.visit += 1
                root.value_sum += -1.0
                sims_left -= 1
                continue

            pol, vals = predictor.predict_many([root.state])
            priors = softmax4(pol[0])
            value = float(vals[0])
            for mv in legal:
                child_state = next_state(
                    root.state,
                    int(mv),
                    minimum_food=int(minimum_food),
                    food_spawn_chance=int(food_spawn_chance),
                )
                root.children[int(mv)] = Node(child_state, prior=float(priors[int(mv)]))
            root.expanded = True
            root.visit += 1
            root.value_sum += value
            sims_left -= 1
            if sims_left <= 0:
                break

        batch_nodes: List[Node] = []
        batch_paths: List[List[Node]] = []
        batch_legals: List[List[int]] = []

        # We allow repeated selection but try to avoid exact duplicates within one batch.
        selected = set()

        # Collect leaves to evaluate.
        while sims_left > 0 and len(batch_nodes) < eb:
            node = root
            path: List[Node] = [node]

            # Selection
            while node.expanded:
                parent_n = node.visit + node.inflight
                sqrt_sum = float(np.sqrt(parent_n)) if parent_n > 0 else 1.0
                best_move = -1
                best_score = -1e9

                for mv in (MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT):
                    child = node.children[mv]
                    if child is None:
                        continue
                    q = child.q()
                    child_n = child.visit + child.inflight
                    u = q + float(cpuct) * child.prior * sqrt_sum / (1.0 + child_n)
                    if u > best_score:
                        best_score = u
                        best_move = mv

                if best_move < 0:
                    break

                node = node.children[best_move]  # type: ignore[index]
                path.append(node)

            # Avoid selecting the exact same leaf many times in a single batch.
            nid = id(node)
            if nid in selected:
                break
            selected.add(nid)

            legal = get_legal_moves(node.state)
            if not legal:
                # Terminal leaf: backprop immediately.
                value = -1.0
                for n in path:
                    n.visit += 1
                    n.value_sum += value
                sims_left -= 1
                continue

            batch_nodes.append(node)
            batch_paths.append(path)
            batch_legals.append(legal)
            for n in path:
                n.inflight += 1
            sims_left -= 1

        if batch_nodes:
            pol, vals = predictor.predict_many([n.state for n in batch_nodes])
            for i, node in enumerate(batch_nodes):
                priors = softmax4(pol[i])
                value = float(vals[i])

                legal = batch_legals[i]
                for mv in legal:
                    child_state = next_state(
                        node.state,
                        int(mv),
                        minimum_food=int(minimum_food),
                        food_spawn_chance=int(food_spawn_chance),
                    )
                    node.children[int(mv)] = Node(child_state, prior=float(priors[int(mv)]))
                node.expanded = True

                for n in batch_paths[i]:
                    n.visit += 1
                    n.value_sum += value
                    n.inflight -= 1

    # Pick move by max visits at root.
    best_mv = MOVE_UP
    best_n = -1
    for mv in (MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT):
        child = root.children[mv]
        if child is None:
            continue
        if child.visit > best_n:
            best_n = child.visit
            best_mv = mv

    return int(best_mv)


def worker_loop(
    *,
    tid: int,
    games: List[GameState],
    turns: int,
    predictor: TorchBatchPredictor,
    simulations: int,
    eval_batch: int,
    cpuct: float,
    minimum_food: int,
    food_spawn_chance: int,
    counter: "list[int]",
    moves_counter: AtomicCounter,
) -> None:
    local = 0
    for _ in range(int(turns)):
        for gi, st in enumerate(games):
            mv = mcts_search(
                root_state=st,
                predictor=predictor,
                simulations=simulations,
                eval_batch=eval_batch,
                cpuct=cpuct,
                minimum_food=minimum_food,
                food_spawn_chance=food_spawn_chance,
            )
            games[gi] = next_state(
                st,
                mv,
                minimum_food=minimum_food,
                food_spawn_chance=food_spawn_chance,
            )
            local += 1
            moves_counter.add(1)
    counter[tid] = local


def reporter_loop(
    *,
    start_t: float,
    stop: threading.Event,
    predictor: TorchBatchPredictor,
    moves_counter: AtomicCounter,
    every_s: float,
) -> None:
    if every_s <= 0:
        return

    last_t = time.perf_counter()
    last_moves = moves_counter.value()
    st0 = predictor.stats()
    last_items = int(st0["total_items"])
    last_batches = int(st0["total_batches"])

    while not stop.wait(timeout=every_s):
        now = time.perf_counter()
        dt = now - last_t
        if dt <= 0:
            dt = 1e-9

        moves = moves_counter.value()
        st = predictor.stats()
        items = int(st["total_items"])
        batches = int(st["total_batches"])

        win_moves_s = (moves - last_moves) / dt
        win_inf_s = (items - last_items) / dt
        win_batches_s = (batches - last_batches) / dt
        elapsed = now - start_t

        print(
            f"t={elapsed:6.2f}s moves={moves} win_moves/s={win_moves_s:,.1f} "
            f"inf_items={items} win_inf/s={win_inf_s:,.1f} win_batches/s={win_batches_s:,.1f} "
            f"avg_batch={st['avg_batch']:.1f} avg_run_ms={st['avg_run_ms']:.3f} q={st['queue']} last_batch={st['last_batch']}",
            flush=True,
        )

        last_t = now
        last_moves = moves
        last_items = items
        last_batches = batches


def main() -> None:
    ap = argparse.ArgumentParser(description="Python MCTS benchmark with batched Torch GPU inference.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--games", type=int, default=64)
    ap.add_argument("--turns", type=int, default=8)

    ap.add_argument("--simulations", type=int, default=128)
    ap.add_argument("--eval-batch", type=int, default=64, help="Leaf evaluations to queue per step within a search.")
    ap.add_argument("--cpuct", type=float, default=1.0)

    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--timeout-ms", type=float, default=1.0)
    ap.add_argument("--encode", choices=["cpu10", "gpu_sparse"], default="cpu10")
    ap.add_argument("--compile", action="store_true")

    ap.add_argument(
        "--report-every",
        type=float,
        default=1.0,
        help="Print runtime stats every N seconds (0 to disable).",
    )

    ap.add_argument("--food-min", type=int, default=0)
    ap.add_argument("--food-chance", type=int, default=0)

    args = ap.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    predictor = TorchBatchPredictor(
        device=str(args.device),
        batch_size=int(args.batch),
        batch_timeout_ms=float(args.timeout_ms),
        encode_mode=str(args.encode),
        compile_model=bool(args.compile),
    )

    # Create initial games. Each game is sequential, but we run many at once.
    games: List[GameState] = [
        _make_random_state(
            width=11,
            height=11,
            num_snakes=4,
            snake_len=10,
            num_food=1,
            num_hazards=0,
            seed=1 + i,
        )
        for i in range(int(args.games))
    ]

    # Split games per thread.
    t = int(args.threads)
    if t <= 0:
        t = 1
    chunks: List[List[GameState]] = [games[i::t] for i in range(t)]
    counters = [0 for _ in range(t)]

    moves_counter = AtomicCounter()
    report_stop = threading.Event()

    start = time.perf_counter()

    report_thr = threading.Thread(
        target=reporter_loop,
        kwargs=dict(
            start_t=start,
            stop=report_stop,
            predictor=predictor,
            moves_counter=moves_counter,
            every_s=float(args.report_every),
        ),
        name="Reporter",
        daemon=True,
    )
    report_thr.start()

    threads: List[threading.Thread] = []
    for tid in range(t):
        thr = threading.Thread(
            target=worker_loop,
            kwargs=dict(
                tid=tid,
                games=chunks[tid],
                turns=int(args.turns),
                predictor=predictor,
                simulations=int(args.simulations),
                eval_batch=int(args.eval_batch),
                cpuct=float(args.cpuct),
                minimum_food=int(args.food_min),
                food_spawn_chance=int(args.food_chance),
                counter=counters,
                moves_counter=moves_counter,
            ),
            daemon=True,
        )
        thr.start()
        threads.append(thr)

    for thr in threads:
        thr.join()

    report_stop.set()
    report_thr.join(timeout=1.0)

    dt = time.perf_counter() - start
    moves = sum(counters)
    moves_per_s = moves / dt if dt > 0 else 0.0

    st = predictor.stats()
    predictor.close()

    print(
        f"threads={t} games={args.games} turns={args.turns} sims={args.simulations} "
        f"encode={args.encode} batch={args.batch} timeout_ms={args.timeout_ms} compile={bool(args.compile)}"
    )
    print(f"total_moves={moves} time={dt:.3f}s => {moves_per_s:,.1f} moves/s")
    print(
        f"predict_batches={st['total_batches']} predict_items={st['total_items']} "
        f"avg_batch={st['avg_batch']:.1f} avg_run_ms={st['avg_run_ms']:.3f} queue={st['queue']}"
    )


if __name__ == "__main__":
    main()
