# Knapsack Genetic Algorithms with DEAP

## Overview

This project contains two DEAP-based Genetic Algorithm implementations for the knapsack problem:

- [MKnap-GA.py](/Users/realdulain/Documents/SLIIT/Y3S1/IS%20-%20SE3062%20-%20Intelligent%20Systems/Labs/06/Knapsack/MKnap-GA.py)
  - solves the **multidimensional knapsack problem**
  - uses the dataset [mknap1.txt](/Users/realdulain/Documents/SLIIT/Y3S1/IS%20-%20SE3062%20-%20Intelligent%20Systems/Labs/06/Knapsack/mknap1.txt)
  - supports multiple constraints per item

- [Knapsack-GA.py](/Users/realdulain/Documents/SLIIT/Y3S1/IS%20-%20SE3062%20-%20Intelligent%20Systems/Labs/06/Knapsack/Knapsack-GA.py)
  - solves the **single-constraint 0/1 knapsack problem**
  - uses the dataset [knap1.txt](/Users/realdulain/Documents/SLIIT/Y3S1/IS%20-%20SE3062%20-%20Intelligent%20Systems/Labs/06/Knapsack/knap1.txt)
  - uses one weight per item and one capacity value

Both programs:
- use binary chromosomes
- use DEAP for the GA implementation
- support baseline and comparison runs
- generate PNG plots automatically

## The Two Datasets

### 1. `mknap1.txt` — Multidimensional knapsack

This dataset contains **multiple constraints** for each instance.

Each instance includes:
- number of items
- number of constraints
- known optimal value
- profit values
- one weight row for each constraint
- one capacity value for each constraint

This dataset is used by [MKnap-GA.py](/Users/realdulain/Documents/SLIIT/Y3S1/IS%20-%20SE3062%20-%20Intelligent%20Systems/Labs/06/Knapsack/MKnap-GA.py).

### 2. `knap1.txt` — Single-constraint knapsack

This dataset contains **only one constraint**.

Each instance includes:
- number of items
- one constraint
- known optimal value
- profit values
- a single weight row
- a single capacity value

This dataset is used by [Knapsack-GA.py](/Users/realdulain/Documents/SLIIT/Y3S1/IS%20-%20SE3062%20-%20Intelligent%20Systems/Labs/06/Knapsack/Knapsack-GA.py).

## Project Files

- [MKnap-GA.py](/Users/realdulain/Documents/SLIIT/Y3S1/IS%20-%20SE3062%20-%20Intelligent%20Systems/Labs/06/Knapsack/MKnap-GA.py): multidimensional knapsack GA
- [Knapsack-GA.py](/Users/realdulain/Documents/SLIIT/Y3S1/IS%20-%20SE3062%20-%20Intelligent%20Systems/Labs/06/Knapsack/Knapsack-GA.py): single-constraint knapsack GA
- [mknap1.txt](/Users/realdulain/Documents/SLIIT/Y3S1/IS%20-%20SE3062%20-%20Intelligent%20Systems/Labs/06/Knapsack/mknap1.txt): multidimensional dataset
- [knap1.txt](/Users/realdulain/Documents/SLIIT/Y3S1/IS%20-%20SE3062%20-%20Intelligent%20Systems/Labs/06/Knapsack/knap1.txt): single-constraint dataset
- [Practical 06.html](/Users/realdulain/Documents/SLIIT/Y3S1/IS%20-%20SE3062%20-%20Intelligent%20Systems/Labs/06/Knapsack/Practical%2006.html): practical sheet

## Requirements

- Python 3.10 or newer
- `deap`
- `matplotlib`

## Setup

### 1. Open the project folder

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

## Running the Multidimensional Version

Use [MKnap-GA.py](/Users/realdulain/Documents/SLIIT/Y3S1/IS%20-%20SE3062%20-%20Intelligent%20Systems/Labs/06/Knapsack/MKnap-GA.py) with [mknap1.txt](/Users/realdulain/Documents/SLIIT/Y3S1/IS%20-%20SE3062%20-%20Intelligent%20Systems/Labs/06/Knapsack/mknap1.txt).

### Baseline run

#### Windows

```powershell
python MKnap-GA.py --dataset mknap1.txt --instance 3 --mode baseline
```

#### macOS

```bash
python MKnap-GA.py --dataset mknap1.txt --instance 3 --mode baseline
```

### Comparison run

#### Windows

```powershell
python MKnap-GA.py --dataset mknap1.txt --instance 3 --mode compare
```

#### macOS

```bash
python MKnap-GA.py --dataset mknap1.txt --instance 3 --mode compare
```

### PNG output folder for the multidimensional version

The multidimensional script saves PNG files under:

```text
plots/
```

Example run folders:

```text
plots/instance_3_baseline_seed_42/
plots/instance_3_compare_seed_42/
```

Inside a comparison folder you will typically see:

```text
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

## Running the Single-Constraint Version

Use [Knapsack-GA.py](/Users/realdulain/Documents/SLIIT/Y3S1/IS%20-%20SE3062%20-%20Intelligent%20Systems/Labs/06/Knapsack/Knapsack-GA.py) with [knap1.txt](/Users/realdulain/Documents/SLIIT/Y3S1/IS%20-%20SE3062%20-%20Intelligent%20Systems/Labs/06/Knapsack/knap1.txt).

### Baseline run

#### Windows

```powershell
python Knapsack-GA.py --dataset knap1.txt --instance 0 --mode baseline
```

#### macOS

```bash
python Knapsack-GA.py --dataset knap1.txt --instance 0 --mode baseline
```

### Comparison run

#### Windows

```powershell
python Knapsack-GA.py --dataset knap1.txt --instance 0 --mode compare
```

#### macOS

```bash
python Knapsack-GA.py --dataset knap1.txt --instance 0 --mode compare
```

### PNG output folder for the single-constraint version

The single-constraint script saves PNG files under:

```text
plots_single/
```

Example run folders:

```text
plots_single/instance_0_baseline_seed_42/
plots_single/instance_0_compare_seed_42/
```

Inside a comparison folder you will typically see:

```text
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

## Common Command Arguments

Both scripts support:

- `--dataset`
  - path to the dataset file

- `--instance`
  - zero-based index of the instance in the dataset

- `--mode`
  - `baseline` for one run using the default settings
  - `compare` for running the operator and mutation comparisons

- `--seed`
  - random seed for reproducibility
  - default: `42`

The single-constraint script also supports:

- `--output`
  - custom root folder for generated PNG files
  - default: `plots_single`

## What the Scripts Do

Each script:
- loads the selected dataset instance
- creates a binary GA representation
- evaluates each solution using a penalty-based fitness function
- runs the baseline or comparison experiments
- saves plots as PNG files

## Plot Types

For each run, the scripts generate:

- an individual plot for each variant
  - best fitness curve
  - average fitness curve
  - optimal reference line

- one `comparison.png`
  - compares the best-fitness curves of all variants on one graph

## Notes

- `mknap1.txt` is the multidimensional dataset.
- `knap1.txt` is the single-constraint dataset.
- The dataset loaders use whitespace splitting, so wrapped lines in the text files do not cause parsing issues.
- Very negative average fitness values can happen because infeasible solutions receive heavy penalties.

## Suggested Practical Workflow

1. Create and activate the virtual environment.
2. Install `deap` and `matplotlib`.
3. Run the multidimensional version if you want to work directly with `mknap1.txt`.
4. Run the single-constraint version if you want the simpler 0/1 formulation.
5. Use the generated PNG files from `plots/` or `plots_single/` in your report.
6. Compare operator behavior and discuss convergence.

## Space for Screenshots / Plots

### Multidimensional baseline plot

```text
[Insert plot from plots/instance_3_baseline_seed_42/baseline.png]
```

### Multidimensional comparison plot

```text
[Insert plot from plots/instance_3_compare_seed_42/comparison.png]
```

### Single-constraint baseline plot

```text
[Insert plot from plots_single/instance_0_baseline_seed_42/baseline.png]
```

### Single-constraint comparison plot

```text
[Insert plot from plots_single/instance_0_compare_seed_42/comparison.png]
```
