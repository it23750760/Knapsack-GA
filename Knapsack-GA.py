from __future__ import annotations

import argparse
import random
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import mean

from deap import base, creator, tools


@dataclass(frozen=True)
class GAConfig:
    selection: str = "tournament"
    tournament_k: int = 3
    crossover: str = "two_point"
    crossover_prob: float = 0.9
    mutation_prob: float = 0.02
    mutation_indpb: float = 0.02
    population_size: int = 150
    generations: int = 200
    elitism: int = 2
    seed: int = 42
    penalty_multiplier: float = 10.0


@dataclass(frozen=True)
class GARunResult:
    config: GAConfig
    best_individual: list[int]
    best_fitness: float
    best_profit: float
    best_weight: float
    overflow: float
    history_best: list[float]
    history_avg: list[float]


def load_instance(
    path: str | Path = "knap1.txt",
    index: int = 0,
) -> tuple[list[float], list[float], float, float]:
    tokens = Path(path).read_text().split()
    if not tokens:
        raise ValueError("Empty dataset file.")

    cursor = 0
    n_instances = int(tokens[cursor])
    cursor += 1

    instances: list[tuple[list[float], list[float], float, float]] = []
    for _ in range(n_instances):
        n = int(tokens[cursor])
        m = int(tokens[cursor + 1])
        optimal_value = float(tokens[cursor + 2])
        cursor += 3

        if m != 1:
            raise ValueError(f"Expected a single-constraint dataset, but found m={m}.")

        v = [float(tokens[cursor + i]) for i in range(n)]
        cursor += n

        w = [float(tokens[cursor + i]) for i in range(n)]
        cursor += n

        C = float(tokens[cursor])
        cursor += 1

        instances.append((v, w, C, optimal_value))

    if index < 0 or index >= len(instances):
        raise IndexError(f"Instance index {index} out of range. Found {len(instances)} instances.")
    return instances[index]


def _ensure_creator_types() -> None:
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)


def make_fitness(v: list[float], w: list[float], C: float, hard: bool = True, M: float = 1000.0):
    def fitness(ind: list[int]) -> tuple[float]:
        val = sum(g * vi for g, vi in zip(ind, v))
        wt = sum(g * wi for g, wi in zip(ind, w))
        of = max(0.0, wt - C)
        if hard:
            return (val - M * of,)
        return (val / (1.0 + of),)

    return fitness


def analyze_solution(individual: list[int], v: list[float], w: list[float], C: float) -> tuple[float, float, float]:
    best_profit = sum(g * vi for g, vi in zip(individual, v))
    best_weight = sum(g * wi for g, wi in zip(individual, w))
    overflow = max(0.0, best_weight - C)
    return best_profit, best_weight, overflow


def rank_select(population: list[creator.Individual], k: int) -> list[creator.Individual]:
    ranked = sorted(population, key=lambda individual: individual.fitness.values[0])
    weights = list(range(1, len(ranked) + 1))
    return random.choices(ranked, weights=weights, k=k)


def build_toolbox(
    n: int,
    fitness_fn,
    pc: float = 0.9,
    pm: float = 0.02,
    tourn_k: int = 3,
    selection: str = "tournament",
    crossover: str = "two_point",
) -> base.Toolbox:
    _ensure_creator_types()

    tb = base.Toolbox()
    tb.register("attr_bool", random.randint, 0, 1)
    tb.register("individual", tools.initRepeat, creator.Individual, tb.attr_bool, n=n)
    tb.register("population", tools.initRepeat, list, tb.individual)
    tb.register("evaluate", fitness_fn)

    if selection == "tournament":
        tb.register("select", tools.selTournament, tournsize=tourn_k)
    elif selection == "roulette":
        tb.register("select", tools.selRoulette)
    elif selection == "rank":
        tb.register("select", rank_select)
    else:
        raise ValueError(f"Unsupported selection method: {selection}")

    if crossover == "one_point":
        tb.register("mate", tools.cxOnePoint)
    elif crossover == "two_point":
        tb.register("mate", tools.cxTwoPoint)
    elif crossover == "uniform":
        tb.register("mate", tools.cxUniform, indpb=0.5)
    else:
        raise ValueError(f"Unsupported crossover method: {crossover}")

    tb.register("mutate", tools.mutFlipBit, indpb=pm)
    tb.register("clone", deepcopy)
    tb.pc = pc
    tb.pm = pm
    return tb


def _record_stats(population: list[creator.Individual]) -> tuple[float, float]:
    fitnesses = [individual.fitness.values[0] for individual in population]
    return max(fitnesses), mean(fitnesses)


def run(
    v: list[float],
    w: list[float],
    C: float,
    gens: int = 200,
    pop_size: int = 150,
    optimal_value: float | None = None,
    seed: int = 42,
    elitism: int = 2,
    penalty_multiplier: float = 10.0,
    **ops,
) -> GARunResult:
    config = GAConfig(
        selection=ops.get("selection", "tournament"),
        tournament_k=ops.get("tourn_k", 3),
        crossover=ops.get("crossover", "two_point"),
        crossover_prob=ops.get("pc", 0.9),
        mutation_prob=ops.get("pm", 0.02),
        mutation_indpb=ops.get("pm", 0.02),
        population_size=pop_size,
        generations=gens,
        elitism=elitism,
        seed=seed,
        penalty_multiplier=penalty_multiplier,
    )

    random.seed(config.seed)
    tb = build_toolbox(
        len(v),
        make_fitness(v, w, C, hard=True, M=penalty_multiplier * max(v)),
        pc=config.crossover_prob,
        pm=config.mutation_indpb,
        tourn_k=config.tournament_k,
        selection=config.selection,
        crossover=config.crossover,
    )
    pop = tb.population(n=config.population_size)

    invalid = [individual for individual in pop if not individual.fitness.valid]
    for individual, fitness in zip(invalid, map(tb.evaluate, invalid)):
        individual.fitness.values = fitness

    history_best: list[float] = []
    history_avg: list[float] = []

    for _ in range(config.generations):
        best_fit, avg_fit = _record_stats(pop)
        history_best.append(best_fit)
        history_avg.append(avg_fit)

        elites = [tb.clone(ind) for ind in tools.selBest(pop, config.elitism)]
        offspring = tb.select(pop, config.population_size - config.elitism)
        offspring = [tb.clone(individual) for individual in offspring]

        for left, right in zip(offspring[::2], offspring[1::2]):
            if random.random() < config.crossover_prob:
                tb.mate(left, right)
                del left.fitness.values
                del right.fitness.values

        for mutant in offspring:
            if random.random() < config.mutation_prob:
                tb.mutate(mutant)
                del mutant.fitness.values

        invalid = [individual for individual in offspring if not individual.fitness.valid]
        for individual, fitness in zip(invalid, map(tb.evaluate, invalid)):
            individual.fitness.values = fitness

        pop = elites + offspring

    best_individual = tools.selBest(pop, 1)[0]
    best_profit, best_weight, overflow = analyze_solution(best_individual, v, w, C)
    return GARunResult(
        config=config,
        best_individual=list(best_individual),
        best_fitness=best_individual.fitness.values[0],
        best_profit=best_profit,
        best_weight=best_weight,
        overflow=overflow,
        history_best=history_best,
        history_avg=history_avg,
    )


def make_run_output_dir(mode: str, instance_index: int, seed: int, output_root: str | Path) -> Path:
    output_dir = Path(output_root) / f"instance_{instance_index}_{mode}_seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def plot_results(result: GARunResult, name: str, optimal_value: float, output_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for plotting. Install it with './.venv/bin/pip install matplotlib'."
        ) from exc

    generations = range(1, len(result.history_best) + 1)
    fig, ax_best = plt.subplots(figsize=(10, 6))
    ax_avg = ax_best.twinx()

    best_line = ax_best.plot(generations, result.history_best, label="Best fitness", linewidth=2, color="tab:blue")
    optimal_line = ax_best.axhline(optimal_value, color="black", linestyle="--", label=f"Optimal = {optimal_value}")
    avg_line = ax_avg.plot(generations, result.history_avg, label="Average fitness", linewidth=2, color="tab:orange", alpha=0.85)

    ax_best.set_xlabel("Generation")
    ax_best.set_ylabel("Best fitness", color="tab:blue")
    ax_avg.set_ylabel("Average fitness", color="tab:orange")
    ax_best.tick_params(axis="y", labelcolor="tab:blue")
    ax_avg.tick_params(axis="y", labelcolor="tab:orange")
    ax_best.set_title(f"GA Results: {name}")

    handles = best_line + avg_line + [optimal_line]
    labels = [handle.get_label() for handle in handles]
    ax_best.legend(handles, labels, loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}.png", dpi=150)
    plt.close(fig)


def plot_comparison(results: list[tuple[str, GARunResult]], output_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for plotting. Install it with './.venv/bin/pip install matplotlib'."
        ) from exc

    plt.figure(figsize=(10, 6))
    for name, result in results:
        generations = range(1, len(result.history_best) + 1)
        plt.plot(generations, result.history_best, label=name, linewidth=2)

    plt.xlabel("Generation")
    plt.ylabel("Best fitness")
    plt.title("GA Variant Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "comparison.png", dpi=150)
    plt.close()


def baseline_config(seed: int) -> GAConfig:
    return GAConfig(seed=seed)


def comparison_configs(seed: int) -> list[tuple[str, GAConfig]]:
    base = baseline_config(seed)
    return [
        ("baseline", base),
        ("tournament_k_2", replace(base, tournament_k=2)),
        ("tournament_k_4", replace(base, tournament_k=4)),
        ("roulette", replace(base, selection="roulette")),
        ("rank", replace(base, selection="rank")),
        ("crossover_one_point", replace(base, crossover="one_point")),
        ("crossover_uniform", replace(base, crossover="uniform")),
        ("mutation_0_01", replace(base, mutation_prob=0.01, mutation_indpb=0.01)),
        ("mutation_0_05", replace(base, mutation_prob=0.05, mutation_indpb=0.05)),
        ("mutation_0_10", replace(base, mutation_prob=0.10, mutation_indpb=0.10)),
    ]


def print_run_summary(name: str, result: GARunResult, optimal_value: float, capacity: float) -> None:
    feasible = result.overflow == 0
    print(f"[{name}]")
    print(f"best_fitness={result.best_fitness:.2f}")
    print(f"best_profit={result.best_profit:.2f}")
    print(f"best_weight={result.best_weight:.2f}")
    print(f"capacity={capacity}")
    print(f"overflow={result.overflow:.2f}")
    print(f"feasible={feasible}")
    print(f"optimal_reference={optimal_value}")
    print(f"last_gen_best={result.history_best[-1]:.2f}")
    print(f"last_gen_avg={result.history_avg[-1]:.2f}")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-constraint DEAP GA for knapsack instances.")
    parser.add_argument("--dataset", default="knap1.txt", help="Path to the knapsack dataset file.")
    parser.add_argument("--instance", type=int, default=0, help="Zero-based instance index inside the dataset.")
    parser.add_argument("--mode", choices=("baseline", "compare"), default="baseline", help="Run baseline or full comparison set.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output", default="plots_single", help="Root directory for saved PNG files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = make_run_output_dir(args.mode, args.instance, args.seed, args.output)
    v, w, C, optimal_value = load_instance(args.dataset, index=args.instance)

    print(f"Loaded instance {args.instance}: n={len(v)}, capacity={C}, optimal={optimal_value}")
    print()

    if args.mode == "baseline":
        base = baseline_config(args.seed)
        result = run(
            v,
            w,
            C,
            gens=base.generations,
            pop_size=base.population_size,
            optimal_value=optimal_value,
            seed=base.seed,
            elitism=base.elitism,
            penalty_multiplier=base.penalty_multiplier,
            selection=base.selection,
            tourn_k=base.tournament_k,
            crossover=base.crossover,
            pc=base.crossover_prob,
            pm=base.mutation_prob,
        )
        plot_results(result, "baseline", optimal_value, output_dir)
        print_run_summary("baseline", result, optimal_value, C)
        print(f"saved_plot={output_dir / 'baseline.png'}")
        return

    collected_results: list[tuple[str, GARunResult]] = []
    for name, config in comparison_configs(args.seed):
        result = run(
            v,
            w,
            C,
            gens=config.generations,
            pop_size=config.population_size,
            optimal_value=optimal_value,
            seed=config.seed,
            elitism=config.elitism,
            penalty_multiplier=config.penalty_multiplier,
            selection=config.selection,
            tourn_k=config.tournament_k,
            crossover=config.crossover,
            pc=config.crossover_prob,
            pm=config.mutation_prob,
        )
        plot_results(result, name, optimal_value, output_dir)
        print_run_summary(name, result, optimal_value, C)
        collected_results.append((name, result))

    plot_comparison(collected_results, output_dir)
    print(f"saved_plots_dir={output_dir}")


if __name__ == "__main__":
    main()
