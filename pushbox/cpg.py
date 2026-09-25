"""Joint-space CPG -- the locomotion model carried over from v1, kept on purpose.

Evidence for keeping it (see README, "What was tested"):
  * the v1 evolved genome walks forward ~0.29 m/s upright and spins
    ~17 deg/s nearly in place with this model;
  * a redesigned per-side, time-reversal CPG was built and tested -- random
    search found few gaits that walk forward AND spin both ways, and the proven
    gait did not transfer to it (it fell over in every configuration).

Model: every joint j follows an asymmetric triangle wave
    target_j = a_j * wave(t * f + phase_j / 2pi, duty)
Walking/arcs : a_j = amp_j * stride_side            (stride in [0, 1] per side)
Turn in place: a_j = +/- turn_amp  (uniform, sign by side) -- the v1
               ALIGN primitive, which was measured to rotate the robot.
The controller issues (dir_l, dir_r, stride_l, stride_r); dir_l != dir_r means
turn in place, and then stride is the turn drive in [0, 1].
"""
import math
from typing import Dict, List

from . import body
from .config import TURN_AMP_MAX

TWO_PI = 2 * math.pi


def asymmetric_wave(cycle_pos: float, amplitude: float, duty: float) -> float:
    c = cycle_pos % 1.0
    if c < duty:
        return -amplitude + 2 * amplitude * (c / duty)
    return amplitude - 2 * amplitude * ((c - duty) / (1 - duty))


class CPG:
    """Joint-space central pattern generator (see module docstring)."""

    def __init__(self, params: Dict[str, float]) -> None:
        self.p = params
        self.t = 0.0
        self.amps = [params[f"amp_{j}"] for j in body.JOINT_NAMES]        # j like "hip_FL"
        self.phases = [params[f"phase_{j}"] / TWO_PI for j in body.JOINT_NAMES]

    def step(self, dt: float, dir_l: int, dir_r: int, stride_l: float, stride_r: float) -> List[float]:
        self.t += dt
        return self.targets(dir_l, dir_r, stride_l, stride_r)

    def targets(self, dir_l: int = 1, dir_r: int = 1, stride_l: float = 0.0, stride_r: float = 0.0) -> List[float]:
        f, duty = self.p["frequency"], self.p["duty"]
        spin = dir_l != dir_r
        out = [0.0] * 8
        for j, name in enumerate(body.JOINT_NAMES):
            left = name.endswith("L")
            if spin:
                a = (dir_l if left else dir_r) * TURN_AMP_MAX * (stride_l if left else stride_r)
            else:
                a = self.amps[j] * (stride_l if left else stride_r)
            out[j] = asymmetric_wave(self.t * f + self.phases[j], a, duty)
        return out
