# Genetic Algorithm for Multidimensional Knapsack using DEAP

## Problem Introduction

This project solves a **multidimensional knapsack problem (MKP)** using a **Genetic Algorithm (GA)** implemented with **DEAP**.

In the knapsack problem, each item has:
- a profit
- one or more weights

The goal is to choose a subset of items that:
- maximizes total profit
- does not exceed the capacity limits

In this dataset, the problem is **multidimensional**, which means each item consumes resources across multiple constraints instead of just one. A candidate solution is represented as a binary chromosome:
- `1` means the item is selected
- `0` means the item is not selected

The GA searches for good solutions by repeatedly applying:
- selection
- crossover
- mutation
- elitism

The solver also uses a **penalty-based fitness function** so infeasible solutions receive a lower fitness score.

## Project Files

- [ga_mknap.py](/Users/realdulain/Documents/SLIIT/Y3S1/IS%20-%20SE3062%20-%20Intelligent%20Systems/Labs/06/Knapsack/ga_mknap.py): Single Python file containing:
  - dataset loader
  - DEAP GA implementation
  - experiment runner
  - plotting functions
- [mknap1.txt](/Users/realdulain/Documents/SLIIT/Y3S1/IS%20-%20SE3062%20-%20Intelligent%20Systems/Labs/06/Knapsack/mknap1.txt): Dataset file
- [Practical 06.html](/Users/realdulain/Documents/SLIIT/Y3S1/IS%20-%20SE3062%20-%20Intelligent%20Systems/Labs/06/Knapsack/Practical%2006.html): Practical sheet

## Requirements

- Python 3.10 or newer is recommended
- `deap`
- `matplotlib`

## Setup Instructions

### 1. Open the project folder

Make sure your terminal is inside the project directory:

```bash
cd "/path/to/Knapsack"
```

### 2. Create a virtual environment

#### Windows

```powershell
python -m venv .venv
```

#### macOS

```bash
python3 -m venv .venv
```

### 3. Activate the virtual environment

#### Windows

```powershell
.venv\Scripts\activate
```

#### macOS

```bash
source .venv/bin/activate
```

### 4. Install dependencies

#### Windows

```powershell
pip install deap matplotlib
```

#### macOS

```bash
pip install deap matplotlib
```

## Dataset Notes

The project uses `mknap1.txt`, which contains multiple MKP instances.

Each instance includes:
- number of items
- number of constraints
- known optimal value
- profits of all items
- weight rows for each constraint
- capacity values

The loader inside `ga_mknap.py` reads the file using whitespace splitting, so line breaks inside the dataset do not affect parsing.

## How the Solution Works

### Encoding

Each chromosome is a list of `0`s and `1`s.

Example:

```text
[1, 0, 1, 1, 0]
```

This means:
- item 1 selected
- item 2 not selected
- item 3 selected
- item 4 selected
- item 5 not selected

### Fitness Function

The fitness is based on:
- total profit
- total overflow across all constraints

If a solution violates one or more capacity constraints, a penalty is subtracted from its profit.

### Genetic Operators

The implementation supports:
- tournament selection
- roulette selection
- rank selection
- one-point crossover
- two-point crossover
- uniform crossover
- bit-flip mutation
- elitism

## Running the Program

### Run the baseline experiment

#### Windows

```powershell
python ga_mknap.py --instance 3 --mode baseline
```

#### macOS

```bash
python ga_mknap.py --instance 3 --mode baseline
```

### Run all comparison experiments

#### Windows

```powershell
python ga_mknap.py --instance 3 --mode compare
```

#### macOS

```bash
python ga_mknap.py --instance 3 --mode compare
```

## Command Arguments

- `--dataset`
  - path to the dataset file
  - default: `mknap1.txt`

- `--instance`
  - zero-based index of the instance inside the dataset
  - default: `3`

- `--mode`
  - `baseline` runs the default GA setup
  - `compare` runs all configured operator and mutation comparisons

- `--seed`
  - random seed for reproducibility
  - default: `42`

## Example Commands

Run baseline on instance 3:

```bash
python ga_mknap.py --instance 3 --mode baseline
```

Run comparisons on instance 4:

```bash
python ga_mknap.py --instance 4 --mode compare
```

Run with a different random seed:

```bash
python ga_mknap.py --instance 3 --mode compare --seed 7
```

## Output Files

The program saves plots into a separate `plots` directory.

Each program run gets its own subfolder:

```text
plots/
  instance_3_baseline_seed_42/
    baseline.png
  instance_3_compare_seed_42/
    baseline.png
    tournament_k_2.png
    tournament_k_4.png
    roulette.png
    rank.png
    crossover_one_point.png
    crossover_uniform.png
    mutation_0_01.png
    mutation_0_05.png
    mutation_0_10.png
    comparison.png
```

### Plot Types

- Individual run plots:
  - best fitness over generations
  - average fitness over generations
  - dashed horizontal line for known optimal value

- Comparison plot:
  - best fitness curve of every variant on one graph

## Recommended Workflow for the Practical

1. Create and activate the virtual environment.
2. Install `deap` and `matplotlib`.
3. Run the baseline experiment.
4. Check the baseline output and generated plot.
5. Run the comparison experiment.
6. Use the generated PNG files for analysis.
7. Compare:
   - selection methods
   - crossover methods
   - mutation rates
8. Write the report using:
   - your settings
   - output values
   - generated graphs
   - observations about convergence and solution quality

## Suggested Report Structure

### 1. Introduction

Briefly explain:
- what the knapsack problem is
- what makes this dataset multidimensional
- why a genetic algorithm is suitable

### 2. Representation and Fitness

Describe:
- binary chromosome representation
- how profit is calculated
- how penalties are applied for constraint violations

### 3. Baseline Configuration

Include:
- tournament selection with `k=3`
- 2-point crossover
- mutation probability `0.02`
- population size `150`
- `200` generations
- elitism `2`
- random seed used

### 4. Experimental Comparison

Discuss:
- selection variants
- crossover variants
- mutation-rate variants

### 5. Results and Discussion

Explain:
- which variant performed best
- which converged faster
- whether the known optimum was reached
- what happened to average fitness
- how diversity and selection pressure affected performance

## Space for Screenshots / Plots

### Baseline Plot

Insert baseline plot here:

```text
[Add screenshot or baseline.png here]
```

### Comparison Plot

Insert comparison plot here:

```text
[Add screenshot or comparison.png here]
```

### Additional Variant Plots

Insert any useful extra plots here:

```text
[Add crossover or mutation plots here]
```

## Notes

- The dataset contains wrapped lines in some places. This is normal and does not break the loader.
- The known optimal value in the dataset is used only as a reference line in the plots.
- Some average fitness curves may become very negative because infeasible solutions are heavily penalized.

## Troubleshooting

### `ModuleNotFoundError: No module named 'deap'`

Install DEAP:

```bash
pip install deap
```

### `ModuleNotFoundError: No module named 'matplotlib'`

Install matplotlib:

```bash
pip install matplotlib
```

### No plots are generated

Check that:
- `matplotlib` is installed in the active virtual environment
- the script completed successfully
- the `plots` folder was created in the project directory

## Academic Use Note

This code is intended to support the practical implementation and report. You should still explain the method, settings, and results in your own words when preparing the submission.
