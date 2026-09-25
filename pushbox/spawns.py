"""Spawn sets.

Training set: NUM_TRAIN_SPAWNS positions stratified in angle around the robot
start (alternating inner/outer radius). Stratification guarantees the GA
always sees boxes in front of, beside, and behind the robot -- v1's
3 uniform-random spawns happened to contain one trivial case and two cases the
controller could not solve, so fitness measured luck of the draw.

Validation set: the 3 legacy spawns (for a direct before/after comparison)
plus NUM_VALIDATION_SPAWNS seeded random spawns never used for selection. The
GA never sees validation scores, so they measure generalisation, not
overfitting to 6 positions.
"""
import math
import random
from typing import List, Tuple

from . import config as C


def _valid(x: float, y: float) -> bool:
    d_goal = math.hypot(C.GOAL_XY[0] - x, C.GOAL_XY[1] - y)
    return d_goal - C.GOAL_RADIUS >= C.SPAWN_MIN_GOAL_CLEARANCE


def train_spawns() -> List[Tuple[float, float]]:
    n = C.NUM_TRAIN_SPAWNS
    r_in = C.SPAWN_MIN_DIST + 0.25 * (C.SPAWN_MAX_DIST - C.SPAWN_MIN_DIST)
    r_out = C.SPAWN_MIN_DIST + 0.75 * (C.SPAWN_MAX_DIST - C.SPAWN_MIN_DIST)
    out = []
    for k in range(n):
        a = math.pi / n + 2 * math.pi * k / n       # offset so no spawn sits on the goal ray
        r = r_in if k % 2 == 0 else r_out
        x, y = r * math.cos(a), r * math.sin(a)
        if not _valid(x, y):
            r = r_in
            x, y = r * math.cos(a), r * math.sin(a)
        out.append((round(x, 4), round(y, 4)))
    return out


def validation_spawns() -> List[Tuple[float, float]]:
    rng = random.Random(C.SPAWN_SEED)
    out = [tuple(s) for s in C.LEGACY_SPAWNS]
    while len(out) < len(C.LEGACY_SPAWNS) + C.NUM_VALIDATION_SPAWNS:
        r = rng.uniform(C.SPAWN_MIN_DIST, C.SPAWN_MAX_DIST)
        a = rng.uniform(0, 2 * math.pi)
        x, y = r * math.cos(a), r * math.sin(a)
        if _valid(x, y):
            out.append((round(x, 4), round(y, 4)))
    return out
