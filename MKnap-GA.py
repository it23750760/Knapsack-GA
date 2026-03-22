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
    loads: list[float]
    violations: list[float]
    history_best: list[float]
    history_avg: list[float]


@dataclass(frozen=True)
class MKnapsackInstance:
    n_items: int
    n_constraints: int
    optimal_value: float
    profits: list[float]
    weights: list[list[float]]
    capacities: list[float]


def _read_block(tokens: list[str], start: int, size: int) -> tuple[list[float], int]:
    end = start + size
    if end > len(tokens):
        raise ValueError("Unexpected end of file while parsing mknap data.")
    return [float(token) for token in tokens[start:end]], end


def load_mknap_instances(path: str | Path) -> list[MKnapsackInstance]:
    tokens = Path(path).read_text().split()
    if not tokens:
        raise ValueError("Empty dataset file.")

    cursor = 0
    n_instances = int(tokens[cursor])
    cursor += 1

    instances: list[MKnapsackInstance] = []
    for _ in range(n_instances):
        if cursor + 2 >= len(tokens):
            raise ValueError("Unexpected end of file while reading instance header.")

        n_items = int(tokens[cursor])
        n_constraints = int(tokens[cursor + 1])
        optimal_value = float(tokens[cursor + 2])
        cursor += 3

        profits, cursor = _read_block(tokens, cursor, n_items)

        weights: list[list[float]] = []
        for _ in range(n_constraints):
            row, cursor = _read_block(tokens, cursor, n_items)
            weights.append(row)

        capacities, cursor = _read_block(tokens, cursor, n_constraints)

        instances.append(
            MKnapsackInstance(
                n_items=n_items,
                n_constraints=n_constraints,
                optimal_value=optimal_value,
                profits=profits,
                weights=weights,
                capacities=capacities,
            )
        )

    if cursor != len(tokens):
        raise ValueError("Dataset contains trailing tokens after the final instance.")

    return instances


def load_mknap_instance(path: str | Path, index: int = 0) -> MKnapsackInstance:
    instances = load_mknap_instances(path)
    if index < 0 or index >= len(instances):
        raise IndexError(f"Instance index {index} out of range. Found {len(instances)} instances.")
    return instances[index]


def load_instance(path: str | Path = "mknap1.txt", index: int = 0) -> tuple[list[float], list[list[float]], list[float], float]:
    instance = load_mknap_instance(path, index=index)
    return instance.profits, instance.weights, instance.capacities, instance.optimal_value


def _ensure_creator_types() -> None:
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)


def make_fitness(v: list[float], w: list[list[float]], C: list[float], hard: bool = True, M: float = 1000.0):
    def fitness(ind: list[int]) -> tuple[float]:
        val = sum(gene * value for gene, value in zip(ind, v))
        loads = [
            sum(gene * weight for gene, weight in zip(ind, row))
            for row in w
        ]
        overflow = sum(max(0.0, load - capacity) for load, capacity in zip(loads, C))
        if hard:
            return (val - M * overflow,)
        return (val / (1.0 + overflow),)

    return fitness


def evaluate_solution(individual: list[int], instance: MKnapsackInstance, penalty_multiplier: float) -> tuple[float]:
    fitness_fn = make_fitness(
        instance.profits,
        instance.weights,
        instance.capacities,
        hard=True,
        M=penalty_multiplier * max(instance.profits),
    )
    return fitness_fn(individual)


def analyze_solution(individual: list[int], instance: MKnapsackInstance) -> tuple[float, list[float], list[float]]:
    profit = sum(gene * profit for gene, profit in zip(individual, instance.profits))
    loads = [
        sum(gene * weight for gene, weight in zip(individual, row))
        for row in instance.weights
    ]
    violations = [max(0.0, load - capacity) for load, capacity in zip(loads, instance.capacities)]
    return profit, loads, violations


def rank_select(
    population: list[creator.Individual],
    k: int,
) -> list[creator.Individual]:
    ranked = sorted(population, key=lambda individual: individual.fitness.values[0])
    weights = list(range(1, len(ranked) + 1))
    return random.choices(ranked, weights=weights, k=k)


def make_run_output_dir(mode: str, instance_index: int, seed: int) -> Path:
    output_dir = Path("plots") / f"instance_{instance_index}_{mode}_seed_{seed}"
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
    optimal_line = ax_best.axhline(
        optimal_value,
        color="black",
        linestyle="--",
        label=f"Optimal = {optimal_value}",
    )
    avg_line = ax_avg.plot(
        generations,
        result.history_avg,
        label="Average fitness",
        linewidth=2,
        color="tab:orange",
        alpha=0.85,
    )

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
    tb.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        tb.attr_bool,
        n=n,
    )
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
    w: list[list[float]],
    C: list[float],
    gens: int = 200,
    pop_size: int = 150,
    optimal_value: float | None = None,
    seed: int = 42,
    elitism: int = 2,
    penalty_multiplier: float = 10.0,
    **ops,
) -> GARunResult:
    instance = MKnapsackInstance(
        n_items=len(v),
        n_constraints=len(C),
        optimal_value=optimal_value if optimal_value is not None else 0.0,
        profits=v,
        weights=w,
        capacities=C,
    )
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
    fitness_fn = make_fitness(v, w, C, hard=True, M=penalty_multiplier * max(v))
    tb = build_toolbox(
        len(v),
        fitness_fn,
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
    best_profit, loads, violations = analyze_solution(best_individual, instance)
    return GARunResult(
        config=config,
        best_individual=list(best_individual),
        best_fitness=best_individual.fitness.values[0],
        best_profit=best_profit,
        loads=loads,
        violations=violations,
        history_best=history_best,
        history_avg=history_avg,
    )


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


def print_run_summary(name: str, result: GARunResult, instance: MKnapsackInstance) -> None:
    feasible = all(violation == 0 for violation in result.violations)
    print(f"[{name}]")
    print(f"best_fitness={result.best_fitness:.2f}")
    print(f"best_profit={result.best_profit:.2f}")
    print(f"feasible={feasible}")
    print(f"violations={[round(v, 2) for v in result.violations]}")
    print(f"optimal_reference={instance.optimal_value}")
    print(f"last_gen_best={result.history_best[-1]:.2f}")
    print(f"last_gen_avg={result.history_avg[-1]:.2f}")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GA solver for Beasley mknap instances using DEAP.")
    parser.add_argument("--dataset", default="mknap1.txt", help="Path to the mknap dataset file.")
    parser.add_argument("--instance", type=int, default=3, help="Zero-based instance index inside the dataset.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--mode",
        choices=("baseline", "compare"),
        default="baseline",
        help="Run only the baseline or the comparison set from the practical.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    output_dir = make_run_output_dir(args.mode, args.instance, args.seed)
    v, w, C, optimal_value = load_instance(dataset_path, index=args.instance)
    instance = MKnapsackInstance(
        n_items=len(v),
        n_constraints=len(C),
        optimal_value=optimal_value,
        profits=v,
        weights=w,
        capacities=C,
    )

    print(
        f"Loaded instance {args.instance}: n={instance.n_items}, "
        f"m={instance.n_constraints}, optimal={instance.optimal_value}"
    )
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
        print_run_summary("baseline", result, instance)
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
        print_run_summary(name, result, instance)
        collected_results.append((name, result))

    plot_comparison(collected_results, output_dir)
    print(f"saved_plots_dir={output_dir}")


if __name__ == "__main__":
    main()
