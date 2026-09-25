"""Stage 2: evolve the full push task (gait + posture + controller gains).

    python main.py                         # full run (runs Stage 1 first if no seeds exist)
    python main.py --generations 300       # stop after 300 total generations, resumable
    python main.py --minutes 90            # stop after ~90 minutes, resumable
    python main.py --workers 8             # parallel episodes (default: cpu_count - 1)
    python main.py --smoke                 # 3-minute end-to-end check in runs/smoke

Outputs in --run-dir (default runs/main):
    ga_log.csv            per-generation statistics
    validation.csv        best genome scored on held-out spawns every VALIDATE_EVERY gens
    best_genome.json      best genome: vector, decoded params, per-spawn results
    checkpoint.json       full resumable state (atomic write every generation)
    spawns.json           training + validation spawn positions
    config_snapshot.json  every constant in config.py at run start
"""
from pushbox.stage2 import main

if __name__ == "__main__":
    main()
