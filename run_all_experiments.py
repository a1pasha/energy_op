#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated script to run all experimental variations
"""

import os
import shutil
import time
from datetime import datetime
import pandas as pd

# Import the main optimization functions
from data.convert_2020_data import convert_2020_xlsx
from optimization import run_nsga2_optimization
from visualization import generate_all_plots
from decision_maker import get_recommended_solutions, print_recommended_solutions
from config import SOLVER


def load_data():
    """Load timeseries and technology parameters"""
    data_dir = "data"
    timeseries_file = os.path.join(data_dir, "energy_system_dataset.csv")
    tech_params_file = os.path.join(data_dir, "technology_parameters.csv")

    timeseries_data = pd.read_csv(timeseries_file, index_col=0, parse_dates=True)
    tech_params = pd.read_csv(tech_params_file, index_col=0)

    date_time_index = pd.date_range(
        start=timeseries_data.index[0],
        periods=len(timeseries_data),
        freq='H'
    )

    return timeseries_data, date_time_index, tech_params


def modify_resilience_weights(weights):
    """
    Temporarily modify resilience weights in optimization.py

    Args:
        weights: tuple of (stirling, redundancy, buffer)
    """
    # Read the file
    with open('optimization.py', 'r') as f:
        lines = f.readlines()

    # Modify lines 89-93
    for i, line in enumerate(lines):
        if i == 88:  # Line 89 (0-indexed)
            lines[i] = f"        resilience_score = (\n"
        elif i == 89:  # Line 90
            lines[i] = f"            {weights[0]} * stirling_norm +\n"
        elif i == 90:  # Line 91
            lines[i] = f"            {weights[1]} * redundancy_norm +\n"
        elif i == 91:  # Line 92
            lines[i] = f"            {weights[2]} * buffer_norm\n"

    # Write back
    with open('optimization.py', 'w') as f:
        f.writelines(lines)


def modify_seed(seed_value):
    """
    Temporarily modify random seed in optimization.py

    Args:
        seed_value: int, the random seed
    """
    # Read the file
    with open('optimization.py', 'r') as f:
        lines = f.readlines()

    # Modify line 303
    for i, line in enumerate(lines):
        if i == 302:  # Line 303 (0-indexed)
            lines[i] = f"            seed={seed_value},\n"

    # Write back
    with open('optimization.py', 'w') as f:
        f.writelines(lines)


def run_single_experiment(run_name, n_gen, pop_size, n_workers,
                          seed=42, weights=(0.33, 0.33, 0.34)):
    """
    Run a single experiment configuration

    Args:
        run_name: str, name of the run (e.g., "RUN_1_Baseline")
        n_gen: int, number of generations
        pop_size: int, population size
        n_workers: int, number of parallel workers
        seed: int, random seed
        weights: tuple, resilience weights (stirling, redundancy, buffer)
    """
    print("\n" + "="*80)
    print(f"STARTING: {run_name}")
    print("="*80)

    # Modify code if needed
    modify_seed(seed)
    modify_resilience_weights(weights)

    # Load data
    print("Loading data...")
    timeseries_data, date_time_index, tech_params = load_data()

    # Create results folder with run name
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_folder = os.path.join("nsga2_results", f"{run_name}_{timestamp}")
    os.makedirs(results_folder, exist_ok=True)

    # Save configuration
    config_path = os.path.join(results_folder, "experiment_config.txt")
    with open(config_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write(f"EXPERIMENT: {run_name}\n")
        f.write("="*70 + "\n")
        f.write(f"Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Generations: {n_gen}\n")
        f.write(f"Population: {pop_size}\n")
        f.write(f"Workers: {n_workers}\n")
        f.write(f"Random Seed: {seed}\n")
        f.write(f"Resilience Weights:\n")
        f.write(f"  - Stirling: {weights[0]}\n")
        f.write(f"  - Redundancy: {weights[1]}\n")
        f.write(f"  - Buffer: {weights[2]}\n")
        f.write("="*70 + "\n")

    # Run optimization
    start_time = time.time()

    res, problem = run_nsga2_optimization(
        timeseries_data,
        date_time_index,
        tech_params,
        n_gen=n_gen,
        pop_size=pop_size,
        results_folder=results_folder,
        n_workers=n_workers
    )

    # Get recommendations
    try:
        recommended_solutions = get_recommended_solutions(res, knee_method='all')
        print_recommended_solutions(recommended_solutions)
        knee_idx = recommended_solutions['knee_point']['index']
        ideal_idx = recommended_solutions['ideal_distance']['index']
    except:
        knee_idx = 0
        ideal_idx = 0
        recommended_solutions = None

    # Generate plots
    try:
        generate_all_plots(res, problem, knee_idx, ideal_idx,
                          results_folder=results_folder,
                          recommended_solutions=recommended_solutions)
    except Exception as e:
        print(f"Warning: Could not generate all plots: {e}")

    # Calculate time
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)

    print(f"\n{run_name} COMPLETED!")
    if hours > 0:
        print(f"Time: {hours}h {minutes}m {seconds}s")
    elif minutes > 0:
        print(f"Time: {minutes}m {seconds}s")
    else:
        print(f"Time: {seconds}s")
    print(f"Results saved to: {results_folder}")

    return results_folder


def main():
    """Run all experiments"""
    # print("="*80)
    # print("AUTOMATED EXPERIMENT RUNNER")
    # print("="*80)
    # print("\nThis script will run 6 experimental variations automatically.")
    # print("Each run will be saved in a separate folder in nsga2_results/")
    # print("\n" + "="*80)

    # Ask user for parameters
    try:
        n_gen = int(input("Enter number of generations for all runs (e.g., 150): ") or "150")
        pop_size = int(input("Enter population size for all runs (e.g., 40): ") or "40")
        n_workers = int(input("Enter number of parallel workers (e.g., 120): ") or "120")
    except:
        print("Invalid input. Using defaults: 150 gen, 40 pop, 120 workers")
        n_gen = 150
        pop_size = 40
        n_workers = 120

    print(f"\nConfiguration: {n_gen} generations, {pop_size} population, {n_workers} workers")

    # Confirm
    confirm = input("\nStart all experiments? (yes/no): ").lower()
    if confirm not in ['yes', 'y']:
        print("Cancelled.")
        return

    # Track all results
    all_results = []
    overall_start = time.time()

    # RUN 1: Baseline
    folder = run_single_experiment(
        "RUN_1_Baseline",
        n_gen=n_gen,
        pop_size=pop_size,
        n_workers=n_workers,
        seed=42,
        weights=(0.33, 0.33, 0.34)
    )
    all_results.append(("RUN 1: Baseline", folder))

    # RUN 2: Different Seed
    folder = run_single_experiment(
        "RUN_2_Seed123",
        n_gen=n_gen,
        pop_size=pop_size,
        n_workers=n_workers,
        seed=123,
        weights=(0.33, 0.33, 0.34)
    )
    all_results.append(("RUN 2: Seed 123", folder))

    # RUN 3: Stirling Priority
    folder = run_single_experiment(
        "RUN_3_Stirling_Priority",
        n_gen=n_gen,
        pop_size=pop_size,
        n_workers=n_workers,
        seed=42,
        weights=(0.50, 0.25, 0.25)
    )
    all_results.append(("RUN 3: Stirling Priority", folder))

    # RUN 4: Redundancy Priority
    folder = run_single_experiment(
        "RUN_4_Redundancy_Priority",
        n_gen=n_gen,
        pop_size=pop_size,
        n_workers=n_workers,
        seed=42,
        weights=(0.25, 0.50, 0.25)
    )
    all_results.append(("RUN 4: Redundancy Priority", folder))

    # RUN 5: Buffer Priority
    folder = run_single_experiment(
        "RUN_5_Buffer_Priority",
        n_gen=n_gen,
        pop_size=pop_size,
        n_workers=n_workers,
        seed=42,
        weights=(0.25, 0.25, 0.50)
    )
    all_results.append(("RUN 5: Buffer Priority", folder))

    # RUN 6: Higher Population
    folder = run_single_experiment(
        "RUN_6_Higher_Population",
        n_gen=n_gen,
        pop_size=pop_size*2,  # Double the population
        n_workers=n_workers,
        seed=42,
        weights=(0.33, 0.33, 0.34)
    )
    all_results.append(("RUN 6: Higher Population", folder))

    # RUN 7: More Generations (only if n_gen < 200)
    if n_gen < 200:
        run7_gen = int(n_gen * 1.67)  # ~250 if n_gen=150
    else:
        run7_gen = n_gen + 50

    folder = run_single_experiment(
        "RUN_7_More_Generations",
        n_gen=run7_gen,
        pop_size=pop_size,
        n_workers=n_workers,
        seed=42,
        weights=(0.33, 0.33, 0.34)
    )
    all_results.append(("RUN 7: More Generations", folder))

    # Restore defaults
    modify_seed(42)
    modify_resilience_weights((0.33, 0.33, 0.34))

    # Summary
    total_time = time.time() - overall_start
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)

    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED!")
    print("="*80)
    if hours > 0:
        print(f"\nTotal time: {hours}h {minutes}m {seconds}s")
    elif minutes > 0:
        print(f"\nTotal time: {minutes}m {seconds}s")
    else:
        print(f"\nTotal time: {seconds}s")

    print("\nResults summary:")
    for i, (name, folder) in enumerate(all_results, 1):
        print(f"  {i}. {name}")
        print(f"     -> {folder}")

    print("\n" + "="*80)
    print("You can now compare results across all runs using the CSV files!")
    print("="*80)


if __name__ == "__main__":
    main()
