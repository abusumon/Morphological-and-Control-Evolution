"""Generic memetic GA engine used by both stages.

Loop per generation
  1. evaluate population (deterministic simulation -> results are cached by
     genome, so elites and duplicates cost nothing)
  2. update best-so-far and stagnation counter
  3. adapt mutation sigma: shrink on improvement, grow on stagnation
  4. every LOCAL_SEARCH_EVERY gens: (1+lambda) hill climb around the best,
     lambda = LOCAL_SEARCH_STEPS small-sigma neighbours evaluated in parallel;
     an improvement replaces the best and is injected as an elite
  5. build next population: elites + children (tournament selection,
     block crossover, gaussian mutation); on long stagnation replace some
     children with heavy-mutation immigrants to restore diversity

Everything needed to resume exactly (population, best, sigma, counters, RNG
state) is exposed through state_dict()/load_state_dict().
"""
import multiprocessing as mp
import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import genome as G


class ParallelMap:
    def __init__(self, workers: int) -> None:
        self.workers = max(1, int(workers))
        self.pool = mp.Pool(self.workers) if self.workers > 1 else None

    def __call__(self, fn: Callable, tasks: Sequence) -> list:
        if not tasks:
            return []
        if self.pool is None:
            return [fn(t) for t in tasks]
        chunk = max(1, len(tasks) // (self.workers * 4))
        return self.pool.map(fn, tasks, chunksize=chunk)

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()


class CachedScorer:
    """Wraps a batch scoring function `score_batch(vecs) -> [(fitness, info)]`
    with a genome-keyed cache. Cache hits are counted for the log."""

    def __init__(self, score_batch: Callable) -> None:
        self.score_batch = score_batch
        self.cache = {}
        self.evals = 0
        self.hits = 0

    def __call__(self, vecs: Sequence) -> list:
        keys = [G.genome_key(v) for v in vecs]
        todo, seen = [], set()
        for k, v in zip(keys, vecs):
            if k in self.cache or k in seen:
                self.hits += 1
            else:
                todo.append(v)
                seen.add(k)
        if todo:
            for v, res in zip(todo, self.score_batch(todo)):
                self.cache[G.genome_key(v)] = res
            self.evals += len(todo)
        return [self.cache[k] for k in keys]


class GeneticAlgorithm:
    def __init__(self, scorer: CachedScorer, cfg: Dict[str, Any], rng_seed: int,
                 initial_population: Sequence[Sequence[float]]) -> None:
        self.scorer = scorer
        self.cfg = cfg
        self.rng = random.Random(rng_seed)
        self.population = [G.clip_or_wrap(v) for v in initial_population]
        self.generation = 0
        self.best = None                 # (fitness, vec, info)
        self.sigma = cfg["sigma_init"]
        self.stagnation = 0
        self.last_results = None

    # ------------------------------------------------------------ operators --
    def _tournament(self, ranked: Sequence[Tuple[float, list, Any]]) -> list:
        contenders = self.rng.sample(ranked, self.cfg["tournament_size"])
        return max(contenders, key=lambda r: r[0])[1]

    def _local_search(self) -> bool:
        if self.best is None or self.cfg["ls_steps"] <= 0:
            return False
        base_f, base_v, _ = self.best
        cands = [G.mutate(base_v, self.cfg["ls_sigma"], 0.3, self.rng) for _ in range(self.cfg["ls_steps"])]
        results = self.scorer(cands)
        f, info = max(results, key=lambda r: r[0])
        v = cands[results.index((f, info))]
        if f > base_f:
            self.best = (f, v, info)
            return True
        return False

    # ----------------------------------------------------------------- step --
    def step(self) -> Dict[str, Any]:
        cfg = self.cfg
        results = self.scorer(self.population)
        ranked = sorted(((r[0], v, r[1]) for r, v in zip(results, self.population)),
                        key=lambda r: r[0], reverse=True)
        self.last_results = ranked

        improved = False
        if self.best is None or ranked[0][0] > self.best[0] + 1e-9:
            self.best = ranked[0]
            improved = True

        ls_improved = False
        if cfg["ls_every"] > 0 and (self.generation + 1) % cfg["ls_every"] == 0:
            ls_improved = self._local_search()
            improved = improved or ls_improved

        if improved:
            self.stagnation = 0
            self.sigma = max(cfg["sigma_min"], self.sigma * 0.95)
        else:
            self.stagnation += 1
            if self.stagnation % cfg["stagnation_window"] == 0:
                self.sigma = min(cfg["sigma_max"], self.sigma * 1.3)

        # next population
        elites = [self.best[1]] + [r[1] for r in ranked if G.genome_key(r[1]) != G.genome_key(self.best[1])]
        new_pop = [list(v) for v in elites[:cfg["elite_count"]]]
        immigrants = 0
        if self.stagnation >= cfg["immigrant_window"] and self.stagnation % cfg["immigrant_window"] == 0:
            immigrants = cfg["immigrant_count"]
        while len(new_pop) < cfg["population_size"] - immigrants:
            if self.rng.random() < cfg["crossover_rate"]:
                child = G.block_crossover(self._tournament(ranked), self._tournament(ranked), self.rng)
            else:
                child = list(self._tournament(ranked))
            new_pop.append(G.mutate(child, self.sigma, cfg["gene_prob"], self.rng))
        for _ in range(immigrants):
            parent = self.rng.choice(ranked[: max(1, len(ranked) // 2)])[1]
            new_pop.append(G.mutate(parent, cfg["sigma_max"] * 1.5, 0.5, self.rng))
        self.population = new_pop
        self.generation += 1

        scores = [r[0] for r in ranked]
        return {
            "generation": self.generation - 1,
            "best_so_far": self.best[0],
            "gen_best": scores[0],
            "gen_mean": sum(scores) / len(scores),
            "gen_median": sorted(scores)[len(scores) // 2],
            "sigma": self.sigma,
            "stagnation": self.stagnation,
            "local_search_improved": ls_improved,
            "immigrants": immigrants,
            "evals": self.scorer.evals,
            "cache_hits": self.scorer.hits,
        }

    # ----------------------------------------------------------- checkpoint --
    def state_dict(self) -> Dict[str, Any]:
        st = self.rng.getstate()
        return {
            "generation": self.generation,
            "population": self.population,
            "best": None if self.best is None else {"fitness": self.best[0], "vec": self.best[1], "info": self.best[2]},
            "sigma": self.sigma,
            "stagnation": self.stagnation,
            "rng_state": [st[0], list(st[1]), st[2]],
        }

    def load_state_dict(self, d: Dict[str, Any]) -> None:
        self.generation = d["generation"]
        self.population = d["population"]
        b = d["best"]
        self.best = None if b is None else (b["fitness"], b["vec"], b["info"])
        self.sigma = d["sigma"]
        self.stagnation = d["stagnation"]
        s = d["rng_state"]
        self.rng.setstate((s[0], tuple(s[1]), s[2]))
