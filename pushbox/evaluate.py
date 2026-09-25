"""Run one episode for (genome, spawn) and return raw metrics.

Watchdogs here only STOP an episode early; they never change the metrics the
fitness is computed from (fitness uses the true initial distance and
best-so-far quantities), so an early stop cannot corrupt the score. This fixes
the v1 bug where the progress watchdog overwrote `initial_dist`, and the
20-second cut-off that killed every far spawn before the robot could reach
the box.
"""
import math
from typing import Any, Dict, Optional, Sequence, Tuple

import pybullet as p

from . import body
from . import config as C
from . import genome as G
from .cpg import CPG
from .controller import PushController

_CLIENT = None


def get_client(gui: bool = False) -> int:
    """One physics client per process, reused across episodes."""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = p.connect(p.GUI if gui else p.DIRECT)
    return _CLIENT


def run_episode(vec: Sequence[float], spawn_xy: Sequence[float], gui: bool = False,
                record_every: int = 0, max_time: Optional[float] = None) -> Dict[str, Any]:
    params = G.decode(vec)
    client = get_client(gui)
    body.build_world(client)
    robot = body.create_robot(client, params["leg_length"])
    box = body.create_box(client, spawn_xy)
    if gui:
        body.create_goal_marker(client)

    cpg = CPG(params)
    ctrl = PushController(params, C.GOAL_XY)
    gx, gy = C.GOAL_XY
    max_time = C.MAX_EPISODE_TIME if max_time is None else max_time

    # settle: let the robot land on its feet with the neutral posture
    neutral = cpg.targets(1, 1, 0.0, 0.0)
    p.setJointMotorControlArray(robot, list(range(8)), p.POSITION_CONTROL, targetPositions=neutral,
                                forces=[C.MOTOR_FORCE] * 8, physicsClientId=client)
    for _ in range(int(C.SETTLE_TIME / C.SIM_DT)):
        p.stepSimulation(physicsClientId=client)

    ox, oy, _, _ = body.pose2d(client, box)
    rx, ry, yaw, up = body.pose2d(client, robot)
    d0 = math.hypot(gx - ox, gy - oy)
    sx, sy, _, _ = ctrl.staging_point(ox, oy)
    stage0 = max(1e-6, math.hypot(sx - rx, sy - ry))

    ctrl_dt = C.SIM_DT * C.CONTROL_EVERY
    n_ctrl = int(max_time / ctrl_dt)
    best_stage = stage0
    best_potential = -1e9
    t_best_potential = 0.0
    stall_ref = (rx, ry, yaw, 0.0)
    flipped = reached = False
    stop_reason = "timeout"
    path_len = 0.0
    traj = []
    state_time = {"APPROACH": 0.0, "ALIGN": 0.0, "PUSH": 0.0}
    t = 0.0
    dist_goal = d0

    for k in range(n_ctrl):
        t = k * ctrl_dt
        dirs = ctrl.step(ctrl_dt, rx, ry, yaw, ox, oy)
        state_time[ctrl.state] += ctrl_dt
        targets = cpg.step(ctrl_dt, *dirs)
        p.setJointMotorControlArray(robot, list(range(8)), p.POSITION_CONTROL, targetPositions=targets,
                                    forces=[C.MOTOR_FORCE] * 8, physicsClientId=client)
        for _ in range(C.CONTROL_EVERY):
            p.stepSimulation(physicsClientId=client)

        nrx, nry, yaw, up = body.pose2d(client, robot)
        path_len += math.hypot(nrx - rx, nry - ry)
        rx, ry = nrx, nry
        ox, oy, _, _ = body.pose2d(client, box)
        dist_goal = math.hypot(gx - ox, gy - oy)
        sx, sy, _, _ = ctrl.staging_point(ox, oy)
        best_stage = min(best_stage, math.hypot(sx - rx, sy - ry))

        if record_every and k % record_every == 0:
            traj.append((t, rx, ry, yaw, ox, oy, ctrl.state))

        if dist_goal <= C.GOAL_RADIUS:
            reached, stop_reason = True, "reached"
            break
        if up < C.FLIP_UP_Z:
            flipped, stop_reason = True, "flipped"
            break

        # progress potential: box progress dominates, approach gives early signal
        potential = (d0 - dist_goal) * C.POTENTIAL_BOX_WEIGHT + (stage0 - best_stage)
        if potential > best_potential + C.POTENTIAL_IMPROVE_EPS:
            best_potential, t_best_potential = potential, t
        elif t - t_best_potential > C.NO_PROGRESS_TIMEOUT:
            stop_reason = "no_progress"
            break

        if t - stall_ref[3] >= C.STALL_WINDOW:
            moved = math.hypot(rx - stall_ref[0], ry - stall_ref[1])
            turned = abs((yaw - stall_ref[2] + math.pi) % (2 * math.pi) - math.pi)
            if moved < C.STALL_MIN_TRANSLATION and turned < C.STALL_MIN_ROTATION:
                stop_reason = "stalled"
                break
            stall_ref = (rx, ry, yaw, t)

    if record_every:
        traj.append((t, rx, ry, yaw, ox, oy, ctrl.state))

    return {
        "spawn": list(spawn_xy), "reached": reached, "flipped": flipped, "time": t,
        "d0": d0, "d_final": dist_goal, "stage0": stage0, "best_stage": best_stage,
        "path_len": path_len, "stop_reason": stop_reason, "transitions": ctrl.transitions,
        "state_time": state_time, "traj": traj, "max_time": max_time,
    }


def evaluate_task(args: Tuple[Sequence[float], Sequence[float], float]) -> Dict[str, Any]:
    """Pool entry point: (genome_vec, spawn_xy, max_time) -> metrics without trajectory."""
    vec, spawn, max_time = args
    return run_episode(vec, spawn, max_time=max_time)


# ------------------------------------------------------------------ stage 1 --
# Scripted command sequence that exercises every locomotion primitive the
# controller uses, INCLUDING the transitions between them (isolated tests
# passed gaits that then fell over when the controller switched commands).
LOCO_SCRIPT = [  # (name, seconds, dir_l, dir_r, stride_l, stride_r)
    ("walk", 2.5, 1, 1, 1.0, 1.0),
    ("spinL", 2.5, -1, 1, 0.8, 0.8),
    ("walk", 2.5, 1, 1, 1.0, 1.0),
    ("spinR", 2.5, 1, -1, 0.8, 0.8),
    ("arcL", 2.5, 1, 1, 0.3, 1.0),
    ("arcR", 2.5, 1, 1, 1.0, 0.3),
    ("hold", 1.0, 1, 1, 0.0, 0.0),
    ("walk", 2.5, 1, 1, 1.0, 1.0),
]


def run_locomotion_test(vec: Sequence[float]) -> Dict[str, Any]:
    params = G.decode(vec)
    client = get_client()
    body.build_world(client)
    robot = body.create_robot(client, params["leg_length"])
    cpg = CPG(params)
    dt = C.SIM_DT * C.CONTROL_EVERY
    neutral = cpg.targets(1, 1, 0.0, 0.0)
    p.setJointMotorControlArray(robot, list(range(8)), p.POSITION_CONTROL, targetPositions=neutral,
                                forces=[C.MOTOR_FORCE] * 8, physicsClientId=client)
    for _ in range(int(C.SETTLE_TIME / C.SIM_DT)):
        p.stepSimulation(physicsClientId=client)

    total = sum(seg[1] for seg in LOCO_SCRIPT)
    elapsed, flipped = 0.0, False
    seg_stats = []
    for name, secs, dl, dr, sl, sr in LOCO_SCRIPT:
        x0, y0, yaw0, _ = body.pose2d(client, robot)
        prev, yaw_acc, path = yaw0, 0.0, 0.0
        px, py = x0, y0
        for _ in range(int(secs / dt)):
            tg = cpg.step(dt, dl, dr, sl, sr)
            p.setJointMotorControlArray(robot, list(range(8)), p.POSITION_CONTROL, targetPositions=tg,
                                        forces=[C.MOTOR_FORCE] * 8, physicsClientId=client)
            for _ in range(C.CONTROL_EVERY):
                p.stepSimulation(physicsClientId=client)
            elapsed += dt
            x, y, yaw, up = body.pose2d(client, robot)
            yaw_acc += (yaw - prev + math.pi) % (2 * math.pi) - math.pi
            path += math.hypot(x - px, y - py)
            px, py, prev = x, y, yaw
            if up < C.FLIP_UP_Z:
                flipped = True
                break
        dx, dy = x - x0, y - y0
        fwd = (dx * math.cos(yaw0) + dy * math.sin(yaw0)) / secs
        # (name, forward speed along initial heading, yaw rate deg/s, path speed m/s)
        # path speed matters: a "spin" that is really a tight circle has ~0 net
        # forward displacement but large path speed.
        seg_stats.append((name, fwd, math.degrees(yaw_acc) / secs, path / secs))
        if flipped:
            break
    return {"flipped": flipped, "upright_frac": min(1.0, elapsed / total), "segments": seg_stats}


def locomotion_fitness(m: Dict[str, Any]) -> float:
    """Stage-1 score: forward speed + tight spins in both directions + correct
    arc signs, all multiplied by the fraction of the script completed upright."""
    def seg(name, idx):
        vals = [s[idx] for s in m["segments"] if s[0] == name]
        return sum(vals) / len(vals) if vals else 0.0
    v = seg("walk", 1)
    drift = abs(seg("walk", 2))
    arc = min(seg("arcL", 2), -seg("arcR", 2))
    hold_translation = seg("hold", 3)

    def spin_quality(name):
        # credit = yaw rate scaled by tightness; turning radius R = path speed / yaw rate.
        # A genuine spin (R ~ 0) keeps full credit; a wide circle (R >= 0.8 m) gets none.
        rate = abs(seg(name, 2))
        sign_ok = seg(name, 2) > 0 if name == "spinL" else seg(name, 2) < 0
        if not sign_ok or rate < 1e-6:
            return 0.0
        radius = seg(name, 3) / math.radians(rate)
        return min(rate, 90.0) * max(0.0, 1.0 - radius / 0.8)

    spin = min(spin_quality("spinL"), spin_quality("spinR"))
    score = (2.0 * min(max(v, 0.0), 0.5)
             + 0.015 * spin
             + 0.005 * min(max(arc, 0.0), 40.0)
             - 0.002 * min(drift, 60.0)
             - 0.5 * hold_translation)
    score = max(0.0, score) * m["upright_frac"]
    if m["flipped"]:
        score *= 0.5
    return score
