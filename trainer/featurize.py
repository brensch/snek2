from __future__ import annotations

import json
from typing import Any

import numpy as np

from constants import HEIGHT, IN_CHANNELS, WIDTH


def _in_bounds(x: int, y: int, w: int, h: int) -> bool:
    return 0 <= x < w and 0 <= y < h


def featurize_state_json(state_bytes: bytes) -> np.ndarray | None:
    """Convert a raw state JSON blob into the current (C,H,W) float32 tensor.

    This featurizer matches the current Go selfplay encoding:
    - C=10, H=W=11
    - 0: food
    - 1: hazards
    - 2..5: body TTL planes (ego + up to 3 enemies)
    - 6..9: health planes (ego + up to 3 enemies)

    Returns None if the state shape doesn't match the current model constants.
    """

    obj: dict[str, Any] = json.loads(state_bytes)

    w = int(obj.get("width", 0))
    h = int(obj.get("height", 0))
    if w != WIDTH or h != HEIGHT:
        return None

    you_id = str(obj.get("you_id", ""))

    x = np.zeros((IN_CHANNELS, HEIGHT, WIDTH), dtype=np.float32)

    for p in obj.get("food", []) or []:
        px = int(p.get("x", -1))
        py = int(p.get("y", -1))
        if _in_bounds(px, py, WIDTH, HEIGHT):
            x[0, py, px] = 1.0

    for p in obj.get("hazards", []) or []:
        px = int(p.get("x", -1))
        py = int(p.get("y", -1))
        if _in_bounds(px, py, WIDTH, HEIGHT):
            x[1, py, px] = 1.0

    snakes = obj.get("snakes", []) or []

    def is_alive(s: dict[str, Any]) -> bool:
        return int(s.get("health", 0)) > 0 and len(s.get("body", []) or []) > 0

    ego = None
    enemies: list[dict[str, Any]] = []
    for s in snakes:
        sid = str(s.get("id", ""))
        if sid == you_id:
            ego = s
        else:
            enemies.append(s)

    enemies.sort(key=lambda s: str(s.get("id", "")))
    enemies = enemies[:3]

    def encode_snake(ttl_c: int, health_c: int, s: dict[str, Any] | None) -> None:
        if not s or not is_alive(s):
            return

        health = float(s.get("health", 0.0)) / 100.0
        x[health_c, :, :] = health

        body = s.get("body", []) or []
        l = len(body)
        if l <= 0:
            return
        denom = float(l)
        for i, bp in enumerate(body):
            px = int(bp.get("x", -1))
            py = int(bp.get("y", -1))
            if not _in_bounds(px, py, WIDTH, HEIGHT):
                continue
            ttl = float(l - i) / denom
            x[ttl_c, py, px] = ttl

    encode_snake(2, 6, ego)
    for i in range(3):
        encode_snake(3 + i, 7 + i, enemies[i] if i < len(enemies) else None)

    return x
