"""Task controller: APPROACH -> ALIGN -> PUSH, with fallbacks to APPROACH.

Differences from the v1 controller (friday.py) finite-state machine, and why:
  * APPROACH goes to a staging point S = box - u*stage_offset (u = unit vector
    box->goal) and routes AROUND the box via waypoints on a clearance circle.
    The v1 ORBIT walked straight at a point 0.5 m behind the box -- a line
    that often crosses the box and shoves it the wrong way, and a point that
    is inside the robot+box collision envelope.
  * PUSH steers toward T = box + u*push_lookahead (i.e. through the box toward
    the goal). The v1 PUSH steered toward the behind point -- the robot's own
    position -- so it never drove the box goalward except by accident.
  * Locomotion primitive: turn in place when |bearing| > turn_threshold, else
    walk with differential stride. This removes the constant-speed
    pure-pursuit orbit (limit cycle) described in the v1 README.
  * Turning in place uses time-reversed oscillation on one side (cpg.py),
    not negative amplitude.
"""
import math
from typing import Dict, Tuple

from .config import (
    ALIGN_HOLD,
    ALIGN_MAX_DRIFT,
    ALIGN_SETTLE,
    ALIGN_TIMEOUT,
    ALIGN_TOLERANCE,
    BOX_CLEARANCE,
    FINAL_APPROACH_CONE,
    MAX_WAYPOINT_STEP,
    MIN_APPROACH_SPEED,
    MIN_STRIDE,
    PRESTAGE_BACKOFF,
    PUSH_LOST_MARGIN,
    PUSH_MAX_ANGLE,
    PUSH_STALL_PROGRESS,
    PUSH_STALL_TIME,
    SLOWDOWN_RADIUS,
    STAGE_ARRIVAL,
    WAYPOINT_RADIUS,
)

TWO_PI = 2 * math.pi


def wrap(a: float) -> float:
    return (a + math.pi) % TWO_PI - math.pi


def unit(dx: float, dy: float) -> Tuple[float, float]:
    n = math.hypot(dx, dy)
    return (dx / n, dy / n) if n > 1e-9 else (1.0, 0.0)


def seg_point_dist(ax: float, ay: float, bx: float, by: float, px: float, py: float) -> float:
    vx, vy = bx - ax, by - ay
    L2 = vx * vx + vy * vy
    t = 0.0 if L2 < 1e-12 else max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / L2))
    return math.hypot(ax + t * vx - px, ay + t * vy - py)


class PushController:
    """APPROACH -> ALIGN -> PUSH task controller with fallback to APPROACH."""

    def __init__(self, params: Dict[str, float], goal_xy: Tuple[float, float]) -> None:
        self.p = params
        self.gx, self.gy = goal_xy
        self.state = "APPROACH"
        self.t_state = 0.0
        self.align_ok_time = 0.0
        self.push_ref_dist = None
        self.push_ref_time = 0.0
        self.transitions = 0
        self.last_target = None

    # ------------------------------------------------------------ geometry --
    def staging_point(self, ox: float, oy: float) -> Tuple[float, float, float, float]:
        ux, uy = unit(self.gx - ox, self.gy - oy)
        s = self.p["stage_offset"]
        return ox - ux * s, oy - uy * s, ux, uy

    def approach_target(self, rx: float, ry: float, ox: float, oy: float) -> Tuple[float, float, bool]:
        """Returns (tx, ty, is_final). Final leg = straight into S from behind,
        along the push direction u."""
        sx, sy, ux, uy = self.staging_point(ox, oy)
        to_s = math.atan2(sy - ry, sx - rx)
        final = (abs(wrap(to_s - math.atan2(uy, ux))) < FINAL_APPROACH_CONE
                 or math.hypot(sx - rx, sy - ry) < STAGE_ARRIVAL * 1.5)
        if final:
            tx, ty = sx, sy
        else:
            tx, ty = sx - ux * PRESTAGE_BACKOFF, sy - uy * PRESTAGE_BACKOFF
        if seg_point_dist(rx, ry, tx, ty, ox, oy) >= BOX_CLEARANCE:
            return tx, ty, final
        a_r = math.atan2(ry - oy, rx - ox)
        a_s = math.atan2(ty - oy, tx - ox)
        delta = wrap(a_s - a_r)
        step = math.copysign(min(abs(delta), MAX_WAYPOINT_STEP), delta)
        a = a_r + step
        return ox + WAYPOINT_RADIUS * math.cos(a), oy + WAYPOINT_RADIUS * math.sin(a), False

    # ---------------------------------------------------------- locomotion --
    def locomote(self, bearing: float, allow_turn_in_place: bool = True, speed: float = 1.0) -> Tuple[int, int, float, float]:
        """Return (dir_l, dir_r, stride_l, stride_r)."""
        p = self.p
        if allow_turn_in_place and abs(bearing) > p["turn_threshold"]:
            d = p["turn_drive"]
            return (-1, 1, d, d) if bearing > 0 else (1, -1, d, d)
        g = p["steer_gain"]
        if bearing > 0:   # target on the left -> shorten left strides
            return 1, 1, speed * max(MIN_STRIDE, 1.0 - g * bearing), speed
        return 1, 1, speed, speed * max(MIN_STRIDE, 1.0 + g * bearing)

    def set_state(self, s: str) -> None:
        if s != self.state:
            self.state = s
            self.t_state = 0.0
            self.align_ok_time = 0.0
            self.push_ref_dist = None
            self.transitions += 1

    # ---------------------------------------------------------------- step --
    def step(self, dt: float, rx: float, ry: float, yaw: float, ox: float, oy: float) -> Tuple[int, int, float, float]:
        self.t_state += dt
        box_goal = math.hypot(self.gx - ox, self.gy - oy)

        if self.state == "APPROACH":
            sx, sy, _, _ = self.staging_point(ox, oy)
            if math.hypot(sx - rx, sy - ry) < STAGE_ARRIVAL:
                self.set_state("ALIGN")
            else:
                tx, ty, final = self.approach_target(rx, ry, ox, oy)
                self.last_target = (tx, ty)
                bearing = wrap(math.atan2(ty - ry, tx - rx) - yaw)
                speed = 1.0
                if final:
                    speed = min(1.0, max(MIN_APPROACH_SPEED, math.hypot(tx - rx, ty - ry) / SLOWDOWN_RADIUS))
                return self.locomote(bearing, speed=speed)

        if self.state == "ALIGN":
            sx, sy, ux, uy = self.staging_point(ox, oy)
            err = wrap(math.atan2(uy, ux) - yaw)
            if math.hypot(sx - rx, sy - ry) > ALIGN_MAX_DRIFT or self.t_state > ALIGN_TIMEOUT:
                self.set_state("APPROACH")
                return 1, 1, 0.0, 0.0
            if self.t_state < ALIGN_SETTLE:
                return 1, 1, 0.0, 0.0
            if abs(err) < ALIGN_TOLERANCE:
                self.align_ok_time += dt
                if self.align_ok_time >= ALIGN_HOLD:
                    self.set_state("PUSH")
                    self.push_ref_dist, self.push_ref_time = box_goal, 0.0
                return 1, 1, 0.0, 0.0          # hold still while settling
            self.align_ok_time = 0.0
            d = max(0.35, self.p["turn_drive"] * min(1.0, abs(err) / 0.5))
            return (-1, 1, d, d) if err > 0 else (1, -1, d, d)

        # PUSH
        _, _, ux, uy = self.staging_point(ox, oy)
        L = self.p["push_lookahead"]
        tx, ty = ox + ux * L, oy + uy * L
        self.last_target = (tx, ty)
        to_box = math.atan2(oy - ry, ox - rx)
        behind_angle = abs(wrap(to_box - math.atan2(uy, ux)))
        dist_box = math.hypot(ox - rx, oy - ry)
        if self.push_ref_dist is None:
            self.push_ref_dist, self.push_ref_time = box_goal, 0.0
        if box_goal < self.push_ref_dist - PUSH_STALL_PROGRESS:
            self.push_ref_dist, self.push_ref_time = box_goal, 0.0
        else:
            self.push_ref_time += dt
        if (behind_angle > PUSH_MAX_ANGLE
                or dist_box > self.p["stage_offset"] + PUSH_LOST_MARGIN
                or self.push_ref_time > PUSH_STALL_TIME):
            self.set_state("APPROACH")
            return 1, 1, 0.0, 0.0
        bearing = wrap(math.atan2(ty - ry, tx - rx) - yaw)
        return self.locomote(bearing, allow_turn_in_place=False)
