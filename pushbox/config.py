"""Central configuration. Every tunable constant lives here so experiments are
reproducible and diffs are reviewable. Nothing else in the project hard-codes
task, physics, or GA numbers."""
import math

# ----------------------------------------------------------------- task ----
GOAL_XY = (3.0, 0.0)
GOAL_RADIUS = 0.5
ROBOT_START_XY = (0.0, 0.0)
ROBOT_START_YAW = 0.0

BOX_HALF_EXTENTS = (0.3, 0.1, 0.2)   # same box as in v1
BOX_MASS = 0.25
SPAWN_MIN_DIST = 1.0                 # box spawn annulus around the robot start
SPAWN_MAX_DIST = 3.0
SPAWN_MIN_GOAL_CLEARANCE = 0.4       # reject spawns already (almost) inside the goal

# ------------------------------------------------------------- physics -----
SIM_DT = 1.0 / 240.0
CONTROL_EVERY = 4                    # controller runs at 60 Hz, physics at 240 Hz
MAX_EPISODE_TIME = 90.0             # s of simulated time; 60 s cut off 3 near-solved pushes (README sec. 8)
MOTOR_FORCE = 60.0                   # N*m per joint (v1 value)
BOX_FRICTION = 0.5                   # PyBullet default, as in the original
SETTLE_TIME = 1.0                    # robot is dropped from 1 m and settles before control starts

# ---------------------------------------------------------- watchdogs ------
FLIP_UP_Z = 0.3                      # torso "up" vector z-component below this = flipped
NO_PROGRESS_TIMEOUT = 15.0           # stop if the progress potential hasn't improved for this long
STALL_WINDOW = 4.0                   # robot must translate or rotate within this window...
STALL_MIN_TRANSLATION = 0.04         # ...by at least this much (m)
STALL_MIN_ROTATION = math.radians(10)  # ...or this much yaw (rad)

# ------------------------------------------------------------- fitness -----
FIT_REACH_BASE = 100.0
FIT_REACH_SPEED = 50.0
FIT_PROGRESS = 80.0
FIT_APPROACH = 10.0
FIT_FLIP_PENALTY = 10.0
FIT_WORST_CASE_WEIGHT = 0.25         # aggregate = (1-w)*mean + w*min over spawns

# ------------------------------------------------------------------ GA -----
POPULATION_SIZE = 24
NUM_GENERATIONS = 2000
ELITE_COUNT = 2
TOURNAMENT_SIZE = 3
CROSSOVER_RATE = 0.7
GENE_MUTATION_PROB = 0.15            # per-gene probability
SIGMA_INIT = 0.10                    # mutation std-dev in normalised [0,1] gene space
SIGMA_MIN = 0.02
SIGMA_MAX = 0.30
STAGNATION_WINDOW = 15               # gens without improvement before sigma grows
IMMIGRANT_WINDOW = 40                # gens without improvement before random immigrants
IMMIGRANT_COUNT = 3

# memetic local search: (1+1) hill climb on the current best
LOCAL_SEARCH_EVERY = 5               # generations
LOCAL_SEARCH_STEPS = 6               # candidate evaluations per local-search phase
LOCAL_SEARCH_SIGMA = 0.03

# spawns
NUM_TRAIN_SPAWNS = 6
NUM_VALIDATION_SPAWNS = 9            # random held-out spawns (plus the 3 legacy spawns)
VALIDATE_EVERY = 25
SPAWN_SEED = 12345
GA_SEED = 42

LEGACY_SPAWNS = [                    # the three spawns v1 was scored on
    (2.250773031828618, 0.3566433447702901),
    (0.25967853073281194, 1.5281520991063862),
    (-1.0991003925136404, -2.2152703178170534),
]

# -------------------------------------------------------------- files ------
RUN_DIR = "runs/main"

# ---------------------------------------------------------- controller ---
# Push-controller thresholds and geometry margins. These lived in
# controller.py in v1; they are kept here so config.py stays the single
# source of tunable constants (values unchanged).
BOX_CLEARANCE = 0.70          # box bounding radius (0.32) + robot half-length w/ legs + margin
WAYPOINT_RADIUS = 0.90
MAX_WAYPOINT_STEP = math.radians(55)
STAGE_ARRIVAL = 0.40          # ALIGN fixes the residual; tighter values deadlock (see README)
ALIGN_TOLERANCE = math.radians(15)   # PUSH steering corrects residual heading error
ALIGN_HOLD = 0.15
ALIGN_TIMEOUT = 15.0          # measured spin ~22 deg/s -> 180 deg takes ~8 s
ALIGN_MAX_DRIFT = 0.70
PUSH_MAX_ANGLE = math.radians(50)
PUSH_LOST_MARGIN = 0.45
PUSH_STALL_TIME = 5.0
PUSH_STALL_PROGRESS = 0.03
MIN_STRIDE = 0.15
PRESTAGE_BACKOFF = 0.5        # extra waypoint behind S so the robot arrives facing the push direction
FINAL_APPROACH_CONE = math.radians(45)
SLOWDOWN_RADIUS = 0.6         # stride scales down inside this distance of S (kills arrival momentum)
MIN_APPROACH_SPEED = 0.6      # short strides barely move this gait
ALIGN_SETTLE = 0.3            # hold still this long on entering ALIGN before spinning

# ----------------------------------------------------------------- cpg ---
TURN_AMP_MAX = 0.5            # rad; v1 used a fixed 0.4 (== drive 0.8)

# ------------------------------------------------------------- evaluate ---
POTENTIAL_BOX_WEIGHT = 4.0    # box-progress term dominates the progress potential
POTENTIAL_IMPROVE_EPS = 0.02  # potential must improve by more than this to reset the no-progress timer
