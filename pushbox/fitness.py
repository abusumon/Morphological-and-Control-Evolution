"""Fitness.

Per spawn (max 150):
  reached      : 100 + 50 * (1 - t / T_max)
  not reached  : 80 * clip(box progress, -0.25, 1) + 10 * approach progress   (< 100 always)
  flipped      : additional -10
where
  box progress      = (d0 - d_final) / (d0 - goal_radius)    [1.0 == box at goal edge]
  approach progress = 1 - best_dist_to_staging / initial_dist_to_staging

Why this replaces the v1 fitness:
  * v1 gave exactly 0 to every non-reaching episode that did not
    move the box goalward, and its "initial_dist" was corrupted by the
    watchdog -- so the GA had no gradient on the hard spawns at all.
  * Normalising by (d0 - goal_radius) makes a far spawn and a near spawn
    worth the same, instead of letting the trivial spawn dominate.
  * The approach term rewards getting to the right side of the box before any
    pushing happens, so early generations are not a flat zero landscape.
  * Path-efficiency term dropped: its "shortest path" (origin -> box) was not
    the shortest path for the task and the speed term already rewards it.

Aggregate over spawns = (1 - w) * mean + w * min, w = FIT_WORST_CASE_WEIGHT,
so a genome cannot win by solving the easy spawns and ignoring the hard one.
"""
from typing import Any, Mapping, Sequence

from . import config as C


def spawn_fitness(m: Mapping[str, Any]) -> float:
    if m["reached"]:
        return C.FIT_REACH_BASE + C.FIT_REACH_SPEED * max(0.0, 1.0 - m["time"] / m["max_time"])
    need = max(1e-6, m["d0"] - C.GOAL_RADIUS)
    progress = min(1.0, max(-0.25, (m["d0"] - m["d_final"]) / need))
    approach = min(1.0, max(0.0, 1.0 - m["best_stage"] / m["stage0"]))
    f = C.FIT_PROGRESS * progress + C.FIT_APPROACH * approach
    if m["flipped"]:
        f -= C.FIT_FLIP_PENALTY
    return f


def aggregate(spawn_scores: Sequence[float]) -> float:
    mean = sum(spawn_scores) / len(spawn_scores)
    w = C.FIT_WORST_CASE_WEIGHT
    return (1.0 - w) * mean + w * min(spawn_scores)
