"""World construction: quadruped robot, box, and goal marker.

Robot = the 8-DOF sagittal quadruped carried over from v1 of this project
(torso + 4 legs x {hip, knee}, revolute about body y), geometry and contact
model unchanged. A redesign (sphere feet, wider stance) was built and tested
but the v1 evolved gait did not transfer to it, so it was reverted -- see
README.
"""
import math
import pybullet as p
import pybullet_data

from typing import Sequence, Tuple

from . import config as C

JOINT_NAMES = ["hip_FL", "knee_FL", "hip_FR", "knee_FR",
               "hip_BL", "knee_BL", "hip_BR", "knee_BR"]
LEG_NAMES = ["FL", "FR", "BL", "BR"]
HIP_JOINT = {"FL": 0, "FR": 2, "BL": 4, "BR": 6}
KNEE_JOINT = {"FL": 1, "FR": 3, "BL": 5, "BR": 7}

TORSO_HALF_EXTENTS = (0.24, 0.12, 0.05)
TORSO_MASS = 1.0
LINK_MASS = 0.1
LIMB_RADIUS = 0.02
# v1 geometry, unchanged: the v1 locomotion genome was evolved on it.
LEG_OFFSETS = {"FL": (0.2, 0.1, -0.05), "FR": (0.2, -0.1, -0.05),
               "BL": (-0.2, 0.1, -0.05), "BR": (-0.2, -0.1, -0.05)}
SPAWN_HEIGHT = 1.0

TORSO_COLOR = (0.16, 0.16, 0.18, 1)
THIGH_COLOR = (0.14, 0.14, 0.16, 1)
SHIN_COLOR = (0.95, 0.6, 0.05, 1)


def build_world(client: int) -> int:
    p.resetSimulation(physicsClientId=client)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
    p.setGravity(0, 0, -9.8, physicsClientId=client)
    p.setTimeStep(C.SIM_DT, physicsClientId=client)
    return p.loadURDF("plane.urdf", physicsClientId=client)


def create_robot(client: int, leg_length: float) -> int:
    thigh_len = shin_len = leg_length / 2.0
    thigh_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=LIMB_RADIUS, height=thigh_len, physicsClientId=client)
    shin_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=LIMB_RADIUS, height=shin_len, physicsClientId=client)
    thigh_vis = p.createVisualShape(p.GEOM_CAPSULE, radius=LIMB_RADIUS * 1.35, length=thigh_len * 0.75,
                                    rgbaColor=THIGH_COLOR, physicsClientId=client)
    shin_vis = p.createVisualShape(p.GEOM_CAPSULE, radius=LIMB_RADIUS * 1.1, length=shin_len * 0.75,
                                   rgbaColor=SHIN_COLOR, physicsClientId=client)
    torso_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=TORSO_HALF_EXTENTS, physicsClientId=client)
    torso_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=TORSO_HALF_EXTENTS, rgbaColor=TORSO_COLOR,
                                    physicsClientId=client)
    masses, cols, viss, pos, inert, parents = [], [], [], [], [], []
    for i, leg in enumerate(LEG_NAMES):
        masses += [LINK_MASS, LINK_MASS]
        cols += [thigh_col, shin_col]
        viss += [thigh_vis, shin_vis]
        pos += [list(LEG_OFFSETS[leg]), [0, 0, -thigh_len]]
        inert += [[0, 0, -thigh_len / 2], [0, 0, -shin_len / 2]]
        parents += [0, 2 * i + 1]
    return p.createMultiBody(
        baseMass=TORSO_MASS, baseCollisionShapeIndex=torso_col, baseVisualShapeIndex=torso_vis,
        basePosition=[C.ROBOT_START_XY[0], C.ROBOT_START_XY[1], SPAWN_HEIGHT],
        baseOrientation=p.getQuaternionFromEuler([0, 0, C.ROBOT_START_YAW]),
        linkMasses=masses, linkCollisionShapeIndices=cols, linkVisualShapeIndices=viss,
        linkPositions=pos, linkOrientations=[[0, 0, 0, 1]] * 8,
        linkInertialFramePositions=inert, linkInertialFrameOrientations=[[0, 0, 0, 1]] * 8,
        linkParentIndices=parents, linkJointTypes=[p.JOINT_REVOLUTE] * 8,
        linkJointAxis=[[0, 1, 0]] * 8, physicsClientId=client)


def create_box(client: int, xy: Sequence[float], yaw: float = 0.0) -> int:
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=C.BOX_HALF_EXTENTS, physicsClientId=client)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=C.BOX_HALF_EXTENTS, rgbaColor=[0.2, 0.45, 0.9, 1],
                              physicsClientId=client)
    box = p.createMultiBody(C.BOX_MASS, col, vis,
                            basePosition=[xy[0], xy[1], C.BOX_HALF_EXTENTS[2]],
                            baseOrientation=p.getQuaternionFromEuler([0, 0, yaw]),
                            physicsClientId=client)
    p.changeDynamics(box, -1, lateralFriction=C.BOX_FRICTION, physicsClientId=client)
    return box


def create_goal_marker(client: int) -> int:
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=C.GOAL_RADIUS, length=0.01,
                              rgbaColor=[0, 1, 0, 0.4], physicsClientId=client)
    return p.createMultiBody(0, -1, vis, basePosition=[C.GOAL_XY[0], C.GOAL_XY[1], 0.005],
                             physicsClientId=client)


def pose2d(client: int, body_id: int) -> Tuple[float, float, float, float]:
    """Return ``(x, y, yaw, up_z)`` of a body; ``up_z`` is the z-component of its local up axis."""
    pos, orn = p.getBasePositionAndOrientation(body_id, physicsClientId=client)
    m = p.getMatrixFromQuaternion(orn)
    yaw = math.atan2(m[3], m[0])
    return pos[0], pos[1], yaw, m[8]
