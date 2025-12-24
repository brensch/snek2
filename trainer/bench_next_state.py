#!/usr/bin/env python3

from __future__ import annotations

import argparse
import dataclasses
import time
from typing import List, Optional, Sequence, Tuple

import numpy as np

# Keep move IDs consistent with Go `rules` package.
MOVE_UP = 0
MOVE_DOWN = 1
MOVE_LEFT = 2
MOVE_RIGHT = 3


@dataclasses.dataclass(slots=True)
class SnakeState:
    # Body is packed as `pos = x + y*W`.
    id_bytes: bytes
    health: int
    body: np.ndarray  # int16/ int32

    @property
    def length(self) -> int:
        return int(self.body.shape[0])


@dataclasses.dataclass(slots=True)
class GameState:
    width: int
    height: int
    snakes: List[SnakeState]
    food: List[int]
    hazards: List[int]
    you_index: int
    turn: int

    def clone(self) -> "GameState":
        # Match Go clone semantics: deep copy snakes + food.
        return GameState(
            width=self.width,
            height=self.height,
            snakes=[
                SnakeState(id_bytes=s.id_bytes, health=int(s.health), body=s.body.copy())
                for s in self.snakes
            ],
            food=list(self.food),
            hazards=list(self.hazards),
            you_index=int(self.you_index),
            turn=int(self.turn),
        )


def _ordered_snakes_for_encoding(state: GameState, *, max_snakes: int = 4) -> List[SnakeState]:
    """Match executor ordering: ego first, then enemies by stable id."""

    out: List[SnakeState] = []
    if 0 <= int(state.you_index) < len(state.snakes):
        ego = state.snakes[int(state.you_index)]
        if ego.health > 0 and ego.length > 0:
            out.append(ego)

    enemies: List[SnakeState] = []
    for i, s in enumerate(state.snakes):
        if i == int(state.you_index):
            continue
        if s.health <= 0 or s.length == 0:
            continue
        enemies.append(s)

    enemies.sort(key=lambda s: s.id_bytes)
    out.extend(enemies)

    if len(out) > max_snakes:
        out = out[:max_snakes]
    return out


def encode_10_planes_into(state: GameState, out: np.ndarray, *, max_snakes: int = 4) -> None:
    """Encode into 10 planes: food, hazards, then [ttl,health] per snake (up to 4).

    Planes:
      0: food mask (1 at food cells)
      1: hazards mask (1 at hazard cells)
      For snake i in ordered snakes:
        2 + 2*i: TTL map (body segments write ttl=len-i at their cells; max on stack)
        2 + 2*i + 1: health plane (filled with health/100 everywhere)
    """

    h, w = int(state.height), int(state.width)
    if out.shape != (10, h, w) or out.dtype != np.float32:
        raise ValueError(f"out must be float32 with shape (10,{h},{w}), got {out.dtype} {out.shape}")

    out.fill(0.0)

    # Food / hazards (vectorized in flat index space).
    if state.food:
        out0 = out[0].reshape(-1)
        idx = np.asarray(state.food, dtype=np.int64)
        out0[idx] = 1.0
    if state.hazards:
        out1 = out[1].reshape(-1)
        idx = np.asarray(state.hazards, dtype=np.int64)
        out1[idx] = 1.0

    snakes = _ordered_snakes_for_encoding(state, max_snakes=max_snakes)
    for i, s in enumerate(snakes):
        ttl_c = 2 + 2 * i
        h_c = ttl_c + 1
        health = float(s.health) / 100.0
        out[h_c, :, :].fill(np.float32(health))

        body = s.body
        L = int(body.shape[0])
        if L <= 0:
            continue

        # TTL: head has ttl=L, tail ttl=1.
        # Stack handling: take max ttl for that cell.
        ttl_flat = out[ttl_c].reshape(-1)

        # Precompute TTL values vector without Python loops.
        # Use float32 directly for the scatter.
        ttls = np.arange(L, 0, -1, dtype=np.float32)

        # Ensure indices are int64 for np.maximum.at.
        idx = body.astype(np.int64, copy=False)
        np.maximum.at(ttl_flat, idx, ttls)


def _unpack_xy(pos: int, width: int) -> Tuple[int, int]:
    # `pos = x + y*width`
    x = int(pos % width)
    y = int(pos // width)
    return x, y


def _pack_xy(x: int, y: int, width: int) -> int:
    return int(x + y * width)


def deterministic_u64_fast(state: GameState, salt: int) -> int:
    """Cheap deterministic mix (Go uses FNV64a over a small summary).

    This is intentionally fast and stable; it is not cryptographic.
    """

    # 64-bit mix constants
    x = (state.width & 0xFFFFFFFF) | ((state.height & 0xFFFFFFFF) << 32)
    x ^= (state.turn & 0xFFFFFFFF) * 0x9E3779B97F4A7C15
    x ^= salt & 0xFFFFFFFFFFFFFFFF
    x ^= (len(state.food) & 0xFFFFFFFF) * 0xBF58476D1CE4E5B9

    for s in state.snakes:
        if s.health <= 0 or s.length == 0:
            continue
        # Fold id bytes cheaply.
        # Using built-in hash() would be salted per-process; keep deterministic.
        # We just xor a small rolling hash.
        h = 0
        for b in s.id_bytes:
            h = ((h * 131) ^ b) & 0xFFFFFFFFFFFFFFFF
        head = int(s.body[0])
        x ^= h
        x ^= (head & 0xFFFFFFFFFFFFFFFF) * 0x94D049BB133111EB
        # xorshift-like mixing
        x ^= (x >> 30) & 0xFFFFFFFFFFFFFFFF
        x = (x * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        x ^= (x >> 27) & 0xFFFFFFFFFFFFFFFF
        x = (x * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        x ^= (x >> 31) & 0xFFFFFFFFFFFFFFFF

    return x & 0xFFFFFFFFFFFFFFFF


def get_legal_moves(state: GameState) -> List[int]:
    you = state.snakes[state.you_index]
    if you.health <= 0 or you.length == 0:
        return []

    head = int(you.body[0])
    hx, hy = _unpack_xy(head, state.width)

    candidates = (
        (MOVE_UP, hx, hy + 1),
        (MOVE_DOWN, hx, hy - 1),
        (MOVE_LEFT, hx - 1, hy),
        (MOVE_RIGHT, hx + 1, hy),
    )

    # Neck check (explicit like Go).
    neck_pos = int(you.body[1]) if you.length > 1 else None

    moves: List[int] = []
    for mv, nx, ny in candidates:
        if nx < 0 or nx >= state.width or ny < 0 or ny >= state.height:
            continue
        npos = _pack_xy(nx, ny, state.width)

        # Conservative collision check: cannot move into any body cell (including tails).
        safe = True
        for s in state.snakes:
            if s.health <= 0:
                continue
            # Numpy vectorized equality would allocate; scan is fine at these sizes.
            if int(np.any(s.body == npos)):
                safe = False
                break

        if not safe:
            continue

        if neck_pos is not None and npos == neck_pos:
            continue

        moves.append(mv)

    return moves


def apply_food_rules(
    state: GameState,
    *,
    minimum_food: int,
    food_spawn_chance: int,
    salt: int,
) -> None:
    if minimum_food < 0:
        minimum_food = 0
    if food_spawn_chance < 0:
        food_spawn_chance = 0
    if food_spawn_chance > 100:
        food_spawn_chance = 100

    deficit = minimum_food - len(state.food)
    if deficit < 0:
        deficit = 0

    spawn_extra = False
    if food_spawn_chance > 0:
        spawn_extra = int(deterministic_u64_fast(state, salt) % 100) < food_spawn_chance

    to_spawn = deficit + (1 if spawn_extra else 0)
    if to_spawn <= 0:
        return

    # Build occupancy bitset for the small board.
    w, h = state.width, state.height
    n = w * h
    occ = np.zeros((n,), dtype=np.bool_)

    for s in state.snakes:
        if s.health <= 0:
            continue
        occ[s.body.astype(np.int64)] = True

    if state.food:
        occ[np.asarray(state.food, dtype=np.int64)] = True

    avail = np.flatnonzero(~occ)
    if avail.size == 0:
        return

    seed = int(deterministic_u64_fast(state, salt) & 0xFFFFFFFFFFFFFFFF)
    if seed == 0:
        seed = 1
    rng = np.random.default_rng(seed)

    k = int(min(to_spawn, int(avail.size)))
    if k <= 0:
        return

    # Choose without replacement.
    chosen = rng.choice(avail, size=k, replace=False)
    # Extend Python list with ints.
    state.food.extend(int(x) for x in chosen.tolist())


def next_state(
    state: GameState,
    move: int,
    *,
    minimum_food: int,
    food_spawn_chance: int,
) -> GameState:
    new_state = state.clone()
    new_state.turn += 1

    you = new_state.snakes[new_state.you_index]
    if you.health <= 0 or you.length == 0:
        return new_state

    head = int(you.body[0])
    hx, hy = _unpack_xy(head, new_state.width)

    nx, ny = hx, hy
    if move == MOVE_UP:
        ny += 1
    elif move == MOVE_DOWN:
        ny -= 1
    elif move == MOVE_LEFT:
        nx -= 1
    elif move == MOVE_RIGHT:
        nx += 1

    new_head = _pack_xy(nx, ny, new_state.width)

    ate_food = False
    if new_state.food:
        # Small list: linear scan is cheap.
        for i, f in enumerate(new_state.food):
            if f == new_head:
                ate_food = True
                new_state.food.pop(i)
                break

    old_body = you.body
    L = int(old_body.shape[0])

    if ate_food:
        new_body = np.empty((L + 1,), dtype=old_body.dtype)
        new_body[0] = new_head
        if L > 1:
            new_body[1:L] = old_body[0 : L - 1]
        elif L == 1:
            # Only head.
            pass
        # Duplicate new tail.
        new_body[L] = new_body[L - 1]
        you.health = 100
    else:
        new_body = np.empty((L,), dtype=old_body.dtype)
        new_body[0] = new_head
        if L > 1:
            new_body[1:L] = old_body[0 : L - 1]
        you.health -= 1

    you.body = new_body

    # Apply Battlesnake-style food spawning (default matches Go DefaultFoodSettings).
    apply_food_rules(
        new_state,
        minimum_food=minimum_food,
        food_spawn_chance=food_spawn_chance,
        salt=int(move) & 0xFFFFFFFFFFFFFFFF,
    )

    return new_state


def _make_random_state(
    *,
    width: int,
    height: int,
    num_snakes: int,
    snake_len: int,
    num_food: int,
    num_hazards: int,
    seed: int,
) -> GameState:
    rng = np.random.default_rng(seed)
    n = width * height

    # Sample distinct cells for snake bodies + food.
    # Bodies can overlap in Battlesnake spawn sometimes, but for throughput benchmarking
    # we keep them distinct to avoid trivial collisions.
    needed = num_snakes * snake_len + num_food + num_hazards
    if needed > n:
        raise ValueError("too many cells requested")

    cells = rng.choice(np.arange(n, dtype=np.int16), size=needed, replace=False)
    offset = 0

    snakes: List[SnakeState] = []
    for i in range(num_snakes):
        body = cells[offset : offset + snake_len].copy()
        offset += snake_len
        snakes.append(
            SnakeState(
                id_bytes=f"s{i}".encode("ascii"),
                health=100,
                body=body,
            )
        )

    food = [int(x) for x in cells[offset : offset + num_food].tolist()]
    offset += num_food
    hazards = [int(x) for x in cells[offset : offset + num_hazards].tolist()]

    return GameState(
        width=width,
        height=height,
        snakes=snakes,
        food=food,
        hazards=hazards,
        you_index=0,
        turn=0,
    )


def _bench_loop(
    roots: Sequence[GameState],
    *,
    seconds: float,
    include_legal_moves: bool,
    branch_all_moves: bool,
    minimum_food: int,
    food_spawn_chance: int,
    encode10: bool,
    encode_batch: int,
) -> Tuple[int, float]:
    t0 = time.perf_counter()
    t_end = t0 + seconds

    ops = 0
    i = 0
    # Keep a rolling state so we don't benchmark purely cache-hot roots.
    s = roots[0]

    enc_buf = None
    if encode10:
        b = int(encode_batch)
        if b <= 0:
            b = 1
        enc_buf = np.empty((b, 10, int(s.height), int(s.width)), dtype=np.float32)
        enc_i = 0

    while True:
        now = time.perf_counter()
        if now >= t_end:
            break

        if include_legal_moves:
            moves = get_legal_moves(s)
            if not moves:
                # Reset to a root.
                i = (i + 1) % len(roots)
                s = roots[i]
                continue
        else:
            moves = [MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT]

        if branch_all_moves:
            # Mimic MCTS expansion cost: create a child per legal move.
            for mv in moves:
                child = next_state(
                    s,
                    mv,
                    minimum_food=minimum_food,
                    food_spawn_chance=food_spawn_chance,
                )
                if enc_buf is not None:
                    encode_10_planes_into(child, enc_buf[enc_i])
                    enc_i += 1
                    if enc_i >= enc_buf.shape[0]:
                        enc_i = 0
                ops += 1
            # Move on to another root.
            i = (i + 1) % len(roots)
            s = roots[i]
        else:
            mv = moves[ops & (len(moves) - 1)] if (len(moves) & (len(moves) - 1)) == 0 else moves[ops % len(moves)]
            s = next_state(
                s,
                mv,
                minimum_food=minimum_food,
                food_spawn_chance=food_spawn_chance,
            )
            if enc_buf is not None:
                encode_10_planes_into(s, enc_buf[enc_i])
                enc_i += 1
                if enc_i >= enc_buf.shape[0]:
                    enc_i = 0
            ops += 1

    return ops, time.perf_counter() - t0


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark Python next-state creation (Go rules.NextState equivalent).")
    p.add_argument("--seconds", type=float, default=2.0)
    p.add_argument("--warmup", type=float, default=0.5)

    p.add_argument("--width", type=int, default=11)
    p.add_argument("--height", type=int, default=11)
    p.add_argument("--snakes", type=int, default=4)
    p.add_argument("--snake-len", type=int, default=10)
    p.add_argument("--food", type=int, default=1)
    p.add_argument("--hazards", type=int, default=0)
    p.add_argument("--roots", type=int, default=128)
    p.add_argument("--seed", type=int, default=1)

    p.add_argument("--food-min", type=int, default=1)
    p.add_argument("--food-chance", type=int, default=15)

    p.add_argument(
        "--no-legal-moves",
        action="store_true",
        help="Skip legal-move filtering (upper bound on pure NextState cost).",
    )
    p.add_argument(
        "--branch-all-moves",
        action="store_true",
        help="Create a child for each legal move (closer to MCTS expansion behavior).",
    )

    p.add_argument(
        "--encode10",
        action="store_true",
        help="Also encode a 10-plane tensor per produced state (food, hazards, per-snake ttl+health).",
    )
    p.add_argument(
        "--encode-batch",
        type=int,
        default=1024,
        help="Preallocated batch size for encode10 (to amortize allocations).",
    )

    args = p.parse_args()

    roots = [
        _make_random_state(
            width=args.width,
            height=args.height,
            num_snakes=args.snakes,
            snake_len=args.snake_len,
            num_food=args.food,
            num_hazards=args.hazards,
            seed=args.seed + i,
        )
        for i in range(args.roots)
    ]

    # Warmup
    _bench_loop(
        roots,
        seconds=float(args.warmup),
        include_legal_moves=not args.no_legal_moves,
        branch_all_moves=bool(args.branch_all_moves),
        minimum_food=int(args.food_min),
        food_spawn_chance=int(args.food_chance),
        encode10=bool(args.encode10),
        encode_batch=int(args.encode_batch),
    )

    ops, dt = _bench_loop(
        roots,
        seconds=float(args.seconds),
        include_legal_moves=not args.no_legal_moves,
        branch_all_moves=bool(args.branch_all_moves),
        minimum_food=int(args.food_min),
        food_spawn_chance=int(args.food_chance),
        encode10=bool(args.encode10),
        encode_batch=int(args.encode_batch),
    )

    rate = ops / dt if dt > 0 else 0.0
    target = 67000.0

    print(
        f"ops={ops} time={dt:.3f}s rate={rate:,.0f} next_states/s "
        f"(target {target:,.0f}/s => {'OK' if rate >= target else 'SLOW'})"
    )


if __name__ == "__main__":
    main()
