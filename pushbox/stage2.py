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
import argparse
import os
import random
import time

from typing import Any, Dict, List, Sequence, Tuple

from . import config as C
from . import evaluate as E
from . import fitness as F
from . import genome as G
from . import pretrain_locomotion
from . import spawns as S
from .ga_core import CachedScorer, GeneticAlgorithm, ParallelMap
from .runio import CsvLog, RunLock, read_json, write_json

POOL = None
TRAIN = None
MAX_TIME = None


def score_spawns(vecs: Sequence[Sequence[float]], spawn_list: Sequence[Sequence[float]],
                 max_time: float) -> List[Tuple[float, Dict[str, Any]]]:
    tasks = [(v, sp, max_time) for v in vecs for sp in spawn_list]
    metrics = POOL(E.evaluate_task, tasks)
    n = len(spawn_list)
    out = []
    for i in range(len(vecs)):
        ms = metrics[i * n:(i + 1) * n]
        scores = [F.spawn_fitness(m) for m in ms]
        info = {
            "spawn_scores": [round(s, 2) for s in scores],
            "reached": sum(m["reached"] for m in ms),
            "times": [round(m["time"], 2) for m in ms],
            "stop_reasons": [m["stop_reason"] for m in ms],
            "final_box_dist": [round(m["d_final"], 3) for m in ms],
        }
        out.append((F.aggregate(scores), info))
    return out


def score_batch(vecs: Sequence[Sequence[float]]) -> List[Tuple[float, Dict[str, Any]]]:
    return score_spawns(vecs, TRAIN, MAX_TIME)


def ga_config(pop: int) -> Dict[str, Any]:
    return dict(population_size=pop, elite_count=C.ELITE_COUNT, tournament_size=C.TOURNAMENT_SIZE,
                crossover_rate=C.CROSSOVER_RATE, gene_prob=C.GENE_MUTATION_PROB, sigma_init=C.SIGMA_INIT,
                sigma_min=C.SIGMA_MIN, sigma_max=C.SIGMA_MAX, stagnation_window=C.STAGNATION_WINDOW,
                immigrant_window=C.IMMIGRANT_WINDOW, immigrant_count=C.IMMIGRANT_COUNT,
                ls_every=C.LOCAL_SEARCH_EVERY, ls_steps=C.LOCAL_SEARCH_STEPS, ls_sigma=C.LOCAL_SEARCH_SIGMA)


def initial_population(pop: int, seeds_path: str, rng: random.Random) -> List[List[float]]:
    """Stage-1 seeds, small-sigma variants of them (varying the control genes
    too), then a few uniform randoms for diversity."""
    seeds = [s["vec"] for s in read_json(seeds_path)] if seeds_path and os.path.exists(seeds_path) else []
    popn = [list(v) for v in seeds[: max(1, pop // 3)]]
    control = set(G.BLOCKS["control"])
    while len(popn) < pop - max(1, pop // 8) and seeds:
        base = list(rng.choice(seeds))
        child = G.mutate(base, 0.05, 0.2, rng)
        for i in control:  # explore controller gains widely from the start
            child[i] = rng.random()
        popn.append(child)
    while len(popn) < pop:
        popn.append(G.random_genome(rng))
    return popn


def snapshot_config() -> Dict[str, Any]:
    return {k: getattr(C, k) for k in dir(C) if k.isupper()}


def main() -> None:
    global POOL, TRAIN, MAX_TIME
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generations", type=int, default=C.NUM_GENERATIONS, help="total generation target")
    ap.add_argument("--minutes", type=float, default=None, help="wall-clock budget for this session")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--pop", type=int, default=C.POPULATION_SIZE)
    ap.add_argument("--run-dir", default=C.RUN_DIR)
    ap.add_argument("--stage1-generations", type=int, default=60)
    ap.add_argument("--fresh", action="store_true", help="start over even if a checkpoint exists")
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end run in runs/smoke")
    a = ap.parse_args()

    MAX_TIME = C.MAX_EPISODE_TIME
    val_every = C.VALIDATE_EVERY
    if a.smoke:
        a.run_dir, a.generations, a.pop, a.stage1_generations, a.fresh = "runs/smoke", 6, 10, 4, True
        MAX_TIME, val_every = 30.0, 3

    os.makedirs(a.run_dir, exist_ok=True)
    with RunLock(a.run_dir):
        seeds_path = os.path.join(a.run_dir, "locomotion_seeds.json")
        if a.fresh or not os.path.exists(seeds_path):
            print(f"No Stage-1 seeds in {a.run_dir} -> running Stage 1 for {a.stage1_generations} generations")
            pretrain_locomotion.run(a.stage1_generations, a.workers, a.run_dir, a.pop, fresh=True)

        TRAIN = S.train_spawns()
        validation = S.validation_spawns()
        ckpt_path = os.path.join(a.run_dir, "checkpoint.json")
        resume = (not a.fresh) and os.path.exists(ckpt_path)

        POOL = ParallelMap(a.workers)
        try:
            rng = random.Random(C.GA_SEED)
            ga = GeneticAlgorithm(CachedScorer(score_batch), ga_config(a.pop), C.GA_SEED,
                                  initial_population(a.pop, seeds_path, rng))
            if resume:
                ga.load_state_dict(read_json(ckpt_path))
                print(f"Resuming Stage 2 at generation {ga.generation}")
            else:
                write_json(os.path.join(a.run_dir, "spawns.json"), {"train": TRAIN, "validation": validation})
                write_json(os.path.join(a.run_dir, "config_snapshot.json"),
                           {**snapshot_config(), "MAX_EPISODE_TIME_USED": MAX_TIME, "POPULATION_USED": a.pop})
            print(f"Training spawns: {TRAIN}")
            print(f"workers={a.workers} pop={a.pop} target_generations={a.generations} max_episode={MAX_TIME}s")

            fields = ["generation", "best_so_far", "best_reached", "gen_best", "gen_mean", "gen_median",
                      "sigma", "stagnation", "local_search_improved", "immigrants", "evals", "cache_hits",
                      "gen_seconds", "wall_s"]
            log = CsvLog(os.path.join(a.run_dir, "ga_log.csv"), fields, resume)
            vlog = CsvLog(os.path.join(a.run_dir, "validation.csv"),
                          ["generation", "train_fitness", "val_fitness", "val_reached", "val_total",
                           "legacy_reached", "val_spawn_scores"], resume)
            t0 = time.time()
            last_val_key = None
            while ga.generation < a.generations:
                g0 = time.time()
                row = ga.step()
                row["best_reached"] = ga.best[2]["reached"]
                row["gen_seconds"] = round(time.time() - g0, 2)
                row["wall_s"] = round(time.time() - t0, 1)
                log.write(row)

                bf, bv, binfo = ga.best
                write_json(os.path.join(a.run_dir, "best_genome.json"),
                           {"generation": row["generation"], "fitness": bf, "train_info": binfo,
                            "train_spawns": TRAIN, "vec": bv, "params": G.decode(bv)})
                write_json(ckpt_path, ga.state_dict())
                print(f"gen {row['generation']:4d} | best {bf:7.2f} reached {binfo['reached']}/{len(TRAIN)} | "
                      f"gen best {row['gen_best']:7.2f} mean {row['gen_mean']:7.2f} | sigma {row['sigma']:.3f} "
                      f"stag {row['stagnation']:3d}{' LS+' if row['local_search_improved'] else ''} | "
                      f"{row['gen_seconds']:.1f}s", flush=True)

                is_last = ga.generation >= a.generations
                if (row["generation"] + 1) % val_every == 0 or is_last:
                    key = G.genome_key(bv)
                    if key != last_val_key:
                        (vf, vinfo), = score_spawns([bv], validation, MAX_TIME)
                        legacy = sum(1 for i in range(len(C.LEGACY_SPAWNS)) if vinfo["spawn_scores"][i] >= C.FIT_REACH_BASE)
                        vlog.write({"generation": row["generation"], "train_fitness": round(bf, 3),
                                    "val_fitness": round(vf, 3), "val_reached": vinfo["reached"],
                                    "val_total": len(validation), "legacy_reached": legacy,
                                    "val_spawn_scores": " ".join(map(str, vinfo["spawn_scores"]))})
                        print(f"   validation: fitness {vf:.2f}, reached {vinfo['reached']}/{len(validation)} "
                              f"(legacy spawns {legacy}/3)", flush=True)
                        last_val_key = key

                if a.minutes is not None and time.time() - t0 >= a.minutes * 60 and not is_last:
                    print(f"Time budget reached. Re-run the same command to resume at generation {ga.generation}.")
                    break
            log.close()
            vlog.close()
            print(f"Done. Best training fitness {ga.best[0]:.2f}; see {a.run_dir}/best_genome.json")
        except KeyboardInterrupt:
            print("\nInterrupted. The last completed generation is checkpointed; re-run to resume.")
        finally:
            POOL.close()

