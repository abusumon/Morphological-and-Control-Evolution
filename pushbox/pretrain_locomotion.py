"""Stage 1: evolve locomotion primitives (walk, spin both ways, arc, and the
transitions between them) before the push task.

Why a separate stage: with random genomes almost every robot falls over or
cannot turn, so the push-task fitness is flat near zero and selection has
nothing to work with. A short, cheap locomotion stage gives Stage 2 a
population that can already move where the controller tells it to.

Output: <run_dir>/locomotion_seeds.json  (top distinct genomes, control genes
reset to the seed defaults because Stage 1 cannot evaluate them).

    python pretrain_locomotion.py --generations 60 --workers 8
"""
import argparse
import os
import time

from typing import Any, Dict, List, Sequence, Tuple

from . import config as C
from . import evaluate as E
from . import genome as G
from .ga_core import CachedScorer, GeneticAlgorithm, ParallelMap
from .runio import CsvLog, RunLock, read_json, write_json

POOL = None


def _loco_task(vec: Sequence[float]) -> Tuple[float, Dict[str, Any]]:
    m = E.run_locomotion_test(vec)
    return E.locomotion_fitness(m), {"flipped": m["flipped"], "upright_frac": round(m["upright_frac"], 3),
                                     "segments": [(s[0], round(s[1], 3), round(s[2], 1), round(s[3], 3)) for s in m["segments"]]}


def score_batch(vecs: Sequence[Sequence[float]]) -> List[Tuple[float, Dict[str, Any]]]:
    return POOL(_loco_task, vecs)


def ga_config(pop: int) -> Dict[str, Any]:
    return dict(population_size=pop, elite_count=C.ELITE_COUNT, tournament_size=C.TOURNAMENT_SIZE,
                crossover_rate=C.CROSSOVER_RATE, gene_prob=C.GENE_MUTATION_PROB, sigma_init=0.15,
                sigma_min=C.SIGMA_MIN, sigma_max=C.SIGMA_MAX, stagnation_window=10, immigrant_window=20,
                immigrant_count=C.IMMIGRANT_COUNT, ls_every=C.LOCAL_SEARCH_EVERY, ls_steps=C.LOCAL_SEARCH_STEPS,
                ls_sigma=C.LOCAL_SEARCH_SIGMA)


def run(generations: int, workers: int, run_dir: str, pop: int, n_seeds: int = 8,
        fresh: bool = False, verbose: bool = True) -> str:
    global POOL
    os.makedirs(run_dir, exist_ok=True)
    ckpt_path = os.path.join(run_dir, "stage1_checkpoint.json")
    POOL = ParallelMap(workers)
    try:
        import random
        rng = random.Random(C.GA_SEED + 1)
        # v1 proven gait + small/medium mutants + a few randoms for diversity
        seed = G.seed_genome()
        n_random = max(1, pop // 8)
        init = [seed]
        while len(init) < pop - n_random:
            init.append(G.mutate(seed, rng.choice([0.05, 0.1, 0.2]), 0.3, rng))
        init += [G.random_genome(rng) for _ in range(n_random)]
        ga = GeneticAlgorithm(CachedScorer(score_batch), ga_config(pop), C.GA_SEED + 1, init)
        resume = (not fresh) and os.path.exists(ckpt_path)
        if resume:
            ga.load_state_dict(read_json(ckpt_path))
            print(f"[stage1] resuming at generation {ga.generation}")
        log = CsvLog(os.path.join(run_dir, "stage1_log.csv"),
                     ["generation", "best_so_far", "gen_best", "gen_mean", "gen_median", "sigma", "stagnation",
                      "local_search_improved", "immigrants", "evals", "cache_hits", "wall_s"], resume)
        t0 = time.time()
        while ga.generation < generations:
            row = ga.step()
            row["wall_s"] = round(time.time() - t0, 1)
            log.write(row)
            write_json(ckpt_path, ga.state_dict())
            if verbose:
                print(f"[stage1] gen {row['generation']:3d} best={row['best_so_far']:.3f} "
                      f"mean={row['gen_mean']:.3f} sigma={row['sigma']:.3f} evals={row['evals']} "
                      f"t={row['wall_s']}s", flush=True)
        log.close()

        # distinct top genomes over the final population + best-so-far
        pool = [(ga.best[0], ga.best[1], ga.best[2])] + list(ga.last_results or [])
        seeds, keys = [], set()
        control = G.BLOCKS["control"]
        seed_vec = G.seed_genome()
        for f, v, info in sorted(pool, key=lambda r: r[0], reverse=True):
            v = list(v)
            for i in control:
                v[i] = seed_vec[i]
            k = G.genome_key(v)
            if k in keys:
                continue
            keys.add(k)
            seeds.append({"loco_fitness": f, "vec": v, "params": G.decode(v), "info": info})
            if len(seeds) >= n_seeds:
                break
        out = os.path.join(run_dir, "locomotion_seeds.json")
        write_json(out, seeds)
        print(f"[stage1] best locomotion fitness {ga.best[0]:.3f}; wrote {len(seeds)} seeds to {out}")
        return out
    finally:
        POOL.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generations", type=int, default=60)
    ap.add_argument("--pop", type=int, default=C.POPULATION_SIZE)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--run-dir", default=C.RUN_DIR)
    ap.add_argument("--fresh", action="store_true", help="ignore an existing stage-1 checkpoint")
    a = ap.parse_args()
    os.makedirs(a.run_dir, exist_ok=True)
    with RunLock(a.run_dir):
        run(a.generations, a.workers, a.run_dir, a.pop, fresh=a.fresh)


if __name__ == "__main__":
    main()
