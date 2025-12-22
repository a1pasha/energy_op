# Experimental Runs Guide

## Overview
Run 3 different optimization configurations to validate results and test sensitivity.

## Three Runs Configuration

### Run 1: BASELINE (Account 1)
- **Generations**: 150
- **Population**: 80
- **Seed**: 42 (default)
- **Purpose**: Main thesis results
- **Script**: `COLAB_SCRIPT_1_BASELINE.txt`

### Run 2: DIFFERENT SEED (Account 2)
- **Generations**: 150
- **Population**: 80
- **Seed**: 123
- **Purpose**: Test reproducibility with different random initialization
- **Script**: `COLAB_SCRIPT_2_SEED123.txt`
- **Changes**: Modified `optimization.py` to use seed=123

### Run 3: HIGHER POPULATION (Account 3)
- **Generations**: 150
- **Population**: 100
- **Seed**: 42
- **Purpose**: Test if larger population improves Pareto front
- **Script**: `COLAB_SCRIPT_3_POP100.txt`

## Setup Instructions

### For Each Google Account:

1. **Upload ZIP to Google Drive**
   - Upload `optimization.zip` to `MyDrive/Energy_Optimization/`
   - Do this for all 3 accounts

2. **Open Google Colab**
   - Account 1: https://colab.research.google.com
   - Account 2: https://colab.research.google.com (different account)
   - Account 3: https://colab.research.google.com (third account)

3. **Create New Notebook**
   - File → New Notebook
   - Name it: "Energy Optimization - [BASELINE/SEED123/POP100]"

4. **Copy-Paste Cells**
   - Open the corresponding script file
   - Copy each cell block into Colab
   - Run cells in order

5. **Start All 3 Simultaneously**
   - Run Cell 5 in all 3 notebooks at the same time
   - All will run in parallel (~12 hours each)

## Expected Results

### Comparison Metrics:

After all 3 complete, compare:

1. **Pareto Front Size**
   - Baseline: ~20-30 solutions
   - Seed123: ~20-30 solutions (different points)
   - Pop100: ~25-35 solutions (more diverse)

2. **Objective Ranges**
   - Cost: Should be similar across all runs
   - Emissions: Should be similar
   - Resilience: Should be similar

3. **Convergence**
   - Check `convergence_history.csv`
   - Pop100 should converge slightly better

## Thesis Analysis

Use these 3 runs to:

1. **Validate reproducibility** (compare Run 1 vs Run 2)
2. **Sensitivity analysis** (how seed affects results)
3. **Population size effect** (Run 1 vs Run 3)
4. **Show statistical robustness** (3 independent runs)

## Time Estimates

- **Each run**: 10-14 hours on Colab (2 cores)
- **Total wall time**: 12 hours (parallel execution)
- **Total CPU hours**: ~36 hours (3 × 12)

## Cost

- **All free** (Google Colab free tier)
- Just need 3 Google accounts

## Files Created

After completion, you'll have:
- `baseline_results.zip` (Run 1)
- `seed123_results.zip` (Run 2)
- `pop100_results.zip` (Run 3)

Each contains:
- Pareto solutions
- All evaluated solutions
- Visualizations
- Input parameters

## Python Run Scripts (Local Use)

If you want to run these locally instead:
- `run_seed123.py` - Changes seed to 123
- `run_pop100.py` - Uses population 100
- `run_gen200.py` - Uses 200 generations

Run with: `python run_seed123.py`
