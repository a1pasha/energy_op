#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Script for Multi-objective Energy System Optimization
6-Bus System with Temperature-Dependent Efficiencies

Version: 2.0
Data: Real 2020.xlsx with 75% utilization
"""

import warnings
warnings.filterwarnings("ignore")

import os
import time
from datetime import datetime
import oemof
import pandas as pd

from config import SOLVER, DEFAULT_N_GENERATIONS, DEFAULT_POPULATION_SIZE

# Import modules
from data.convert_2020_data import convert_2020_xlsx
from optimization import run_nsga2_optimization
from visualization import generate_all_plots
from decision_maker import get_recommended_solutions, print_recommended_solutions

try:
    print(f"Using oemof version: {oemof.__version__}")
except AttributeError:
    print("Using oemof (version info not available)")


def create_timestamped_results_folder():
    """Create timestamped folder for results"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_folder = os.path.join("nsga2_results", f"run_{timestamp}")
    os.makedirs(results_folder, exist_ok=True)
    return results_folder


def save_input_data(results_folder, timeseries_data, tech_params, n_gen, pop_size, n_workers=None):
    """Save input data and configuration"""
    # Save timeseries
    timeseries_path = os.path.join(results_folder, "input_timeseries_data.csv")
    timeseries_data.to_csv(timeseries_path)
    print(f"Saved input timeseries data to: {timeseries_path}")

    # Save tech params
    tech_params_path = os.path.join(results_folder, "input_technology_parameters.csv")
    tech_params.to_csv(tech_params_path)
    print(f"Saved input technology parameters to: {tech_params_path}")

    # Save config
    config_path = os.path.join(results_folder, "optimization_config.txt")
    with open(config_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("OPTIMIZATION CONFIGURATION\n")
        f.write("="*70 + "\n")
        f.write(f"Version: 2.0 (6-Bus System with Parallel Execution)\n")
        f.write(f"Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data Source: Real 2020.xlsx (75% utilization)\n")
        f.write(f"Solver: {SOLVER}\n")
        f.write(f"Algorithm: NSGA-II (Parallelized)\n")
        f.write(f"Generations: {n_gen}\n")
        f.write(f"Population: {pop_size}\n")
        if n_workers:
            f.write(f"Parallel Workers: {n_workers} cores\n")
        f.write(f"Total Evaluations: {n_gen * pop_size}\n")
        f.write(f"Timesteps: {len(timeseries_data)}\n")
        f.write(f"Technologies: {len(tech_params)}\n")
        f.write("="*70 + "\n")
        f.write("\nFEATURES:\n")
        f.write("  + 6-bus configuration (electricity, pv, heat, heat_grid, gas, hydrogen)\n")
        f.write("  + Time-varying grid CO2 emissions\n")
        f.write("  + Temperature data\n")
        f.write("  + Seasonal gas prices\n")
        f.write("  + Feed-in tariff\n")
        f.write("  + Temperature-dependent heat pump efficiency\n")
        f.write("  + Dynamic capacity limits\n")
        f.write("  + PV routing and grid export tracking\n")
        f.write("  + District heating pump modeling\n")
        f.write("\nObjectives:\n")
        f.write("  1. Minimize Total Cost\n")
        f.write("  2. Minimize CO2 Emissions\n")
        f.write("  3. Maximize Resilience\n")
        f.write("="*70 + "\n")
    print(f"Saved configuration to: {config_path}")


def load_data():
    """
    Load data from local data folder
    """
    # Use local data folder in Project2
    data_dir = "data"
    timeseries_file = os.path.join(data_dir, "energy_system_dataset.csv")
    tech_params_file = os.path.join(data_dir, "technology_parameters.csv")

    if os.path.exists(timeseries_file) and os.path.exists(tech_params_file):
        print("  Loading pre-converted data...")
        timeseries_data = pd.read_csv(timeseries_file, index_col=0, parse_dates=True)
        tech_params = pd.read_csv(tech_params_file, index_col=0)
        # Create explicit date range matching the data length
        date_time_index = pd.date_range(
            start=timeseries_data.index[0],
            periods=len(timeseries_data),
            freq='H'
        )

        print(f"  [OK] Loaded {len(date_time_index)} timesteps")
        print(f"  [OK] Data columns: {len(timeseries_data.columns)}")
        print(f"  [OK] Technologies: {len(tech_params)}")

        # Show what data we have
        features = []
        if 'grid_emission_factor' in timeseries_data.columns:
            features.append("Grid CO2 (time-varying)")
        if 'outdoor_temperature' in timeseries_data.columns:
            features.append("Temperature")
        if 'hp_air_cop' in timeseries_data.columns:
            features.append("HP Air COP (temp-dependent)")
        if 'hp_geo_cop' in timeseries_data.columns:
            features.append("HP Geo COP")

        if features:
            print(f"  Features: {', '.join(features)}")

    else:
        print("  [INFO] Data not found. Converting from 2020.xlsx...")
        # Try to find 2020.xlsx in common relative locations
        project_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            os.path.join(project_dir, "2020.xlsx"),  # Same folder as main.py
            os.path.join(project_dir, "data", "2020.xlsx"),  # In data folder
            os.path.join(project_dir, "..", "..", "oemof Model", "2020.xlsx"),  # Original location (relative)
        ]

        xlsx_path = None
        for path in possible_paths:
            if os.path.exists(path):
                xlsx_path = path
                print(f"  [OK] Found 2020.xlsx at: {path}")
                break

        if xlsx_path is None:
            print("  [ERROR] 2020.xlsx not found in any expected location!")
            print("  Searched:")
            for path in possible_paths:
                print(f"    - {path}")
            print("  [FALLBACK] Using dummy data instead")
            from data.data_loader import create_dummy_timeseries, create_dummy_tech_params
            timeseries_data, date_time_index = create_dummy_timeseries()
            tech_params = create_dummy_tech_params()
        else:
            timeseries_data, date_time_index, tech_params = convert_2020_xlsx(
                xlsx_path,
                output_dir=data_dir
            )

    return timeseries_data, date_time_index, tech_params


def main():
    """Main function for multi-objective optimization"""
    print("="*80)
    print("MULTI-OBJECTIVE ENERGY SYSTEM OPTIMIZATION")
    print("="*80)
    print(f"Version: 2.0 (6-Bus System)")
    print(f"Data: Real 2020.xlsx with temperature-dependent efficiencies")
    print(f"Utilization: 75% of available data")
    print(f"Solver: {SOLVER}")
    print(f"Algorithm: NSGA-II")
    print("\nObjectives:")
    print("  1. Minimize Total Cost")
    print("  2. Minimize CO2 Emissions")
    print("  3. Maximize Resilience (Stirling + Redundancy + Buffer)")
    print("="*80)

    # ========================================================================
    # CREATE RESULTS FOLDER
    # ========================================================================
    results_folder = create_timestamped_results_folder()
    print(f"\nResults will be saved to: {results_folder}")

    start_time = time.time()

    # ========================================================================
    # STEP 1: Load data
    # ========================================================================
    print("\n[STEP 1/5] Loading data...")
    timeseries_data, date_time_index, tech_params = load_data()
    print(f"Loaded {len(date_time_index)} timesteps")

    # ========================================================================
    # STEP 2: Get optimization parameters
    # ========================================================================
    print("\n[STEP 2/5] Setting optimization parameters...")
    print("\nNOTE: For quick testing, use 10-20 generations.")
    print("      For thesis results, use 100-200 generations (recommended: 150)")

    import multiprocessing
    available_cores = multiprocessing.cpu_count()

    try:
        n_gen = int(input(f"\nEnter number of generations (default: {DEFAULT_N_GENERATIONS}): ") or str(DEFAULT_N_GENERATIONS))
        pop_size = int(input(f"Enter population size (default: {DEFAULT_POPULATION_SIZE}): ") or str(DEFAULT_POPULATION_SIZE))
        n_workers = input(f"Enter number of parallel workers (default: {available_cores} - all available cores): ") or str(available_cores)
        n_workers = int(n_workers)
    except:
        n_gen = DEFAULT_N_GENERATIONS
        pop_size = DEFAULT_POPULATION_SIZE
        n_workers = available_cores
        print(f"\nUsing default values: {n_gen} generations, {pop_size} population, {n_workers} workers")

    # ========================================================================
    # STEP 3: Save input data
    # ========================================================================
    print("\n[STEP 3/5] Saving input data and configuration...")
    print("-"*80)
    save_input_data(results_folder, timeseries_data, tech_params, n_gen, pop_size, n_workers)

    # ========================================================================
    # STEP 4: Run optimization
    # ========================================================================
    print("\n[STEP 4/5] Running NSGA-II optimization...")
    print("-"*80)
    print("  [INFO] Using temperature-dependent heat pump efficiencies")
    print("  [INFO] Using time-varying grid emissions")
    print("  [INFO] Using 6-bus system configuration")
    print("-"*80)

    res, problem = run_nsga2_optimization(
        timeseries_data,
        date_time_index,
        tech_params,
        n_gen=n_gen,
        pop_size=pop_size,
        results_folder=results_folder,
        n_workers=n_workers
    )

    # ========================================================================
    # STEP 5: Find recommended solutions
    # ========================================================================
    print("\n[STEP 5/5] Finding recommended solutions...")
    print("-"*80)
    try:
        # Use 'all' to get all three knee point methods
        recommended_solutions = get_recommended_solutions(res, knee_method='all')
        print_recommended_solutions(recommended_solutions)
        knee_idx = recommended_solutions['knee_point']['index']
        ideal_idx = recommended_solutions['ideal_distance']['index']
    except Exception as e:
        print(f"  WARNING: Could not find recommendations: {e}")
        knee_idx = 0
        ideal_idx = 0

    # ========================================================================
    # STEP 6: Generate visualizations
    # ========================================================================
    print("\n[STEP 6/6] Generating results and visualizations...")
    print("-"*80)
    try:
        generate_all_plots(res, problem, knee_idx, ideal_idx,
                          results_folder=results_folder,
                          recommended_solutions=recommended_solutions)
    except Exception as e:
        print(f"  WARNING: Could not generate all plots: {e}")
        import traceback
        traceback.print_exc()

    # ========================================================================
    # Calculate elapsed time
    # ========================================================================
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)

    # ========================================================================
    # DONE
    # ========================================================================
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE!")
    print("="*80)

    if hours > 0:
        print(f"\nTotal time: {hours}h {minutes}m {seconds}s")
    elif minutes > 0:
        print(f"\nTotal time: {minutes}m {seconds}s")
    else:
        print(f"\nTotal time: {seconds}s")

    print(f"\nResults saved in: {results_folder}")
    print("\nGenerated files:")
    print("  INPUT DATA:")
    print("    - input_timeseries_data.csv")
    print("    - input_technology_parameters.csv")
    print("    - optimization_config.txt")
    print("  RESULTS:")
    print("    - all_evaluated_solutions.csv")
    print("    - pareto_solutions.csv")
    print("  VISUALIZATIONS:")
    print("    - pareto_front_3d.png")
    print("    - pareto_front_with_recommendations.png")
    print("    - decision_variables_distribution.png")
    print("    - convergence_plot.png")
    print("\n" + "="*80)
    print("\nFEATURES:")
    print("  + 6-bus system (electricity, pv, heat, heat_grid, gas, hydrogen)")
    print("  + PV routing and grid export tracking")
    print("  + District heating pump modeling")
    print("  + Time-varying grid CO2 emissions")
    print("  + Temperature data")
    print("  + Seasonal gas prices")
    print("  + Temperature-dependent heat pump COP")
    print("  + Actual technology capacities from 2020.xlsx")
    print("\nData utilization: 75%")
    print("="*80)


if __name__ == "__main__":
    main()
