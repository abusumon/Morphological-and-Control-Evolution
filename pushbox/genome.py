"""Genome = vector in [0,1]^N; decode() maps each gene to its physical range.

Normalised space means one mutation sigma is meaningful for every gene,
crossover never leaves the valid range, and circular genes (phases) wrap
instead of piling up at a boundary.

Layout (24 genes):
  leg blocks   : for each leg, hip amp/phase + knee amp/phase  (4 x 4)
                 -- the v1 CPG representation, crossed over as
                 whole legs so coordinated amp/phase sets travel together
  global block : frequency, duty, leg_length
  control block: steering / navigation gains used by controller.py
                 (NOT evolvable in v1)
"""
import json
import math
import os
import random
from typing import Dict, List, Sequence, Tuple

from . import body

TWO_PI = 2 * math.pi

GENE_SPEC: List[Tuple[str, float, float, bool, str]] = []  # name, low, high, circular, block
for _leg in body.LEG_NAMES:
    for _joint in ("hip", "knee"):
        GENE_SPEC.append((f"amp_{_joint}_{_leg}", 0.05, 0.55, False, f"leg_{_leg}"))
        GENE_SPEC.append((f"phase_{_joint}_{_leg}", 0.0, TWO_PI, True, f"leg_{_leg}"))
GENE_SPEC += [
    ("frequency",      0.5,  2.5,  False, "global"),
    ("duty",           0.55, 0.85, False, "global"),
    ("leg_length",     0.25, 0.35, False, "global"),
    ("steer_gain",     0.3,  3.0,  False, "control"),
    ("turn_threshold", math.radians(15), math.radians(70), False, "control"),
    ("turn_drive",     0.4,  1.0,  False, "control"),
    ("stage_offset",   0.80, 1.30, False, "control"),
    ("push_lookahead", 0.0,  1.5,  False, "control"),
]
GENE_NAMES = [g[0] for g in GENE_SPEC]
N_GENES = len(GENE_SPEC)
BLOCKS: Dict[str, List[int]] = {}
for _i, _g in enumerate(GENE_SPEC):
    BLOCKS.setdefault(_g[4], []).append(_i)


def decode(vec: Sequence[float]) -> Dict[str, float]:
    out = {}
    for x, (name, lo, hi, _c, _b) in zip(vec, GENE_SPEC):
        out[name] = lo + x * (hi - lo)
    return out


def encode(params: Dict[str, float]) -> List[float]:
    vec = []
    for name, lo, hi, circ, _b in GENE_SPEC:
        x = (params[name] - lo) / (hi - lo)
        vec.append(x % 1.0 if circ else min(1.0, max(0.0, x)))
    return vec


def random_genome(rng: random.Random) -> List[float]:
    return [rng.random() for _ in range(N_GENES)]


DEFAULT_CONTROL = {"steer_gain": 1.2, "turn_threshold": math.radians(35), "turn_drive": 0.8,
                   "stage_offset": 1.0, "push_lookahead": 0.5}

# Seed gait = best genome of the v1 run (generation 346): proven to walk
# forward upright and to spin.
V1_BEST_GAIT = {
    "hip_FL": (0.42585394442989927, 0.04948501179559982), "knee_FL": (0.4203730714739379, 1.6842264471544786),
    "hip_FR": (0.3183542830622498, 3.4034084848648716), "knee_FR": (0.47007984179075385, 3.9773489369983133),
    "hip_BL": (0.3131863570104073, 3.072013544802619), "knee_BL": (0.29062888660964226, 5.883115776577492),
    "hip_BR": (0.17155308579313566, 0.04184343137033946), "knee_BR": (0.27409641012661073, 0.9713486927987286),
}
V1_FREQUENCY = 1.8627707618042917
V1_LEG_LENGTH = 0.31244822266225536


def seed_params() -> Dict[str, float]:
    prm = dict(DEFAULT_CONTROL)
    for jname, (amp, ph) in V1_BEST_GAIT.items():
        joint, leg = jname.split("_")
        prm[f"amp_{joint}_{leg}"] = amp
        prm[f"phase_{joint}_{leg}"] = ph
    prm.update(frequency=V1_FREQUENCY, duty=0.7, leg_length=V1_LEG_LENGTH)
    return prm


def seed_genome() -> List[float]:
    return encode(seed_params())


def clip_or_wrap(vec: Sequence[float]) -> List[float]:
    return [x % 1.0 if spec[3] else min(1.0, max(0.0, x)) for x, spec in zip(vec, GENE_SPEC)]


def mutate(vec: Sequence[float], sigma: float, gene_prob: float, rng: random.Random) -> List[float]:
    child = list(vec)
    mutated = False
    for i in range(N_GENES):
        if rng.random() < gene_prob:
            child[i] += rng.gauss(0.0, sigma)
            mutated = True
    if not mutated:  # always change something: never waste an evaluation on a clone
        i = rng.randrange(N_GENES)
        child[i] += rng.gauss(0.0, sigma)
    return clip_or_wrap(child)


def block_crossover(a: Sequence[float], b: Sequence[float], rng: random.Random) -> List[float]:
    """Whole-block uniform crossover for leg and global blocks; per-gene uniform
    for the (largely independent) control gains."""
    child = [0.0] * N_GENES
    for block, idxs in BLOCKS.items():
        if block == "control":
            for i in idxs:
                child[i] = a[i] if rng.random() < 0.5 else b[i]
        else:
            src = a if rng.random() < 0.5 else b
            for i in idxs:
                child[i] = src[i]
    return child


def genome_key(vec: Sequence[float]) -> Tuple[float, ...]:
    return tuple(round(x, 9) for x in vec)


def load_vec(path: str) -> List[float]:
    with open(path) as f:
        return json.load(f)["vec"]
