#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-objective optimization using NSGA-II with parallel execution
"""

import os
import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination

from multiprocessing.pool import Pool
from functools import partial

from config import (
    CAPACITY_SCALE_MIN, CAPACITY_SCALE_MAX,
    TECHNOLOGIES, STORAGES, SOLVER,
    RESILIENCE_WEIGHTS, BUFFER_NORM_MAX
)
from energy_system import create_energy_system
from utils import solve_energy_system, extract_results_from_model


# Global worker pool (to be set by run_nsga2_optimization)
_global_pool = None


def evaluate_single_solution(x_and_data):
    """
    Worker function to evaluate a single solution (must be at module level for pickling)

    Args:
        x_and_data: tuple of (x, timeseries_data, date_time_index, tech_params_original, technologies, storages)

    Returns:
        tuple: (objectives, solution_record, success_flag)
    """
    x, timeseries_data, date_time_index, tech_params_original, technologies, storages = x_and_data

    success_flag = True
    try:
        # Scale technology and storage capacities
        scaled_tech_params = tech_params_original.copy()

        for j, tech in enumerate(technologies):
            scaled_tech_params.loc[tech, "capacity_kw"] = \
                tech_params_original.loc[tech, "capacity_kw"] * x[j]

        for j, storage in enumerate(storages):
            idx = len(technologies) + j
            scaled_tech_params.loc[storage, "storage_capacity_kwh"] = \
                tech_params_original.loc[storage, "storage_capacity_kwh"] * x[idx]

        # Create and solve energy system
        energy_system, model = create_energy_system(
            timeseries_data,
            date_time_index,
            scaled_tech_params
        )

        # Solve the model
        success = solve_energy_system(model, solver=SOLVER)

        if not success:
            raise Exception("Solver failed")

        # Extract results
        results = extract_results_from_model(
            energy_system,
            model,
            timeseries_data,
            scaled_tech_params
        )

        # Assign objectives - validate values
        F_cost = results["total_cost"] if np.isfinite(results["total_cost"]) else 1e6
        F_emissions = results["total_emissions"] if np.isfinite(results["total_emissions"]) else 1e6

        # Calculate resilience score using config weights
        stirling_norm = results["stirling_index"] if np.isfinite(results["stirling_index"]) else 0
        redundancy_norm = results["redundancy_index"] if np.isfinite(results["redundancy_index"]) else 0

        # Normalize buffer using BUFFER_NORM_MAX from config (not /10.0)
        # Paper shows buffer ~5-16 hours, using 20 as upper bound
        buffer_norm = min(results["buffer_capacity_index"] / BUFFER_NORM_MAX, 1.0) if np.isfinite(results["buffer_capacity_index"]) else 0

        # NEW: Get balance index (already 0-1 normalized)
        balance_norm = results["balance_index"] if np.isfinite(results["balance_index"]) else 0

        # Use config weights (now includes balance component)
        resilience_score = (
            RESILIENCE_WEIGHTS['stirling'] * stirling_norm +
            RESILIENCE_WEIGHTS['redundancy'] * redundancy_norm +
            RESILIENCE_WEIGHTS['buffer'] * buffer_norm +
            RESILIENCE_WEIGHTS['balance'] * balance_norm  # NEW: Breaks emissions correlation
        )

        # NEW: Apply penalty for boundary saturation (prevents clustering at bounds)
        # Count how many variables are at boundaries (within 5% of min/max)
        n_vars = len(x)
        n_at_lower = sum([1 for xi in x if xi <= CAPACITY_SCALE_MIN * 1.05])
        n_at_upper = sum([1 for xi in x if xi >= CAPACITY_SCALE_MAX * 0.95])
        saturation_fraction = (n_at_lower + n_at_upper) / n_vars

        # If >40% of variables are saturated, apply penalty to resilience
        if saturation_fraction > 0.4:
            saturation_penalty = 0.1 * (saturation_fraction - 0.4)  # Up to 6% penalty
            resilience_score = resilience_score * (1 - saturation_penalty)

        # Negate for minimization
        F_resilience = -resilience_score if np.isfinite(resilience_score) else -1e-6

    except Exception as e:
        print(f"Solution evaluation failed: {e}")
        success_flag = False
        F_cost = 1e6
        F_emissions = 1e6
        F_resilience = -1e-6

    # Build solution record
    solution_record = {'Status': 'Success' if success_flag else 'Failed'}
    for j, tech in enumerate(technologies):
        solution_record[tech] = x[j]
    for j, storage in enumerate(storages):
        idx = len(technologies) + j
        solution_record[storage] = x[idx]
    solution_record['Cost'] = F_cost
    solution_record['Emissions_kg'] = F_emissions
    solution_record['Resilience'] = -F_resilience  # Convert back to positive

    # Final validation to prevent NaN/Inf
    F_cost = np.nan_to_num(F_cost, nan=1e6, posinf=1e6, neginf=1e6)
    F_emissions = np.nan_to_num(F_emissions, nan=1e6, posinf=1e6, neginf=1e6)
    F_resilience = np.nan_to_num(F_resilience, nan=-1e-6, posinf=-1e-6, neginf=-1e-6)

    return [F_cost, F_emissions, F_resilience], solution_record, success_flag


class EnergySystemOptimizationProblem(Problem):
    """
    NSGA-II Problem Definition for Multi-Objective Energy System Optimization

    Uses multiprocessing Pool for efficient parallelization across multiple cores.

    Decision Variables: Capacity scaling factors for each technology (0.1 to 2.0)
    Objectives:
        1. Minimize Total Cost
        2. Minimize CO2 Emissions
        3. Maximize Resilience (negated for minimization)
    """

    def __init__(self, timeseries_data, date_time_index, tech_params_original, pool=None):
        """
        Initialize the optimization problem

        Args:
            timeseries_data (DataFrame): Time series data
            date_time_index: Time index
            tech_params_original (DataFrame): Original technology parameters
            pool: multiprocessing.Pool for parallel evaluation
        """
        self.timeseries_data = timeseries_data
        self.date_time_index = date_time_index
        self.tech_params_original = tech_params_original.copy()
        self.pool = pool

        self.technologies = TECHNOLOGIES
        self.storages = STORAGES

        n_var = len(self.technologies) + len(self.storages)
        n_obj = 3  # Cost, Emissions, Resilience
        n_constr = 0

        xl = np.array([CAPACITY_SCALE_MIN] * n_var)
        xu = np.array([CAPACITY_SCALE_MAX] * n_var)

        # Track all evaluated solutions
        self.all_solutions = []
        self.current_generation = 0
        self.solution_counter = 0

        # Track convergence history manually
        self.convergence_history = []

        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu)

    def _evaluate(self, X, out, *args, **kwargs):
        """
        Evaluate objective functions for a batch of solutions (with optional parallelization)

        Args:
            X: array of shape (n_solutions, n_variables)
            out: output dictionary
        """
        n_solutions = X.shape[0]

        # Prepare data for parallel evaluation
        eval_data = [
            (X[i], self.timeseries_data, self.date_time_index,
             self.tech_params_original, self.technologies, self.storages)
            for i in range(n_solutions)
        ]

        # Evaluate solutions in parallel if pool is available
        if self.pool is not None:
            results = self.pool.map(evaluate_single_solution, eval_data)
        else:
            # Sequential evaluation if no pool
            results = [evaluate_single_solution(data) for data in eval_data]

        # Unpack results
        F = []
        for i, (objectives, solution_record, success_flag) in enumerate(results):
            F.append(objectives)

            # Add generation and solution_id to record
            solution_record['Generation'] = self.current_generation
            solution_record['Solution_ID'] = self.solution_counter
            self.all_solutions.append(solution_record)
            self.solution_counter += 1

        # Convert to numpy array
        out["F"] = np.array(F)


def run_nsga2_optimization(timeseries_data, date_time_index, tech_params,
                          n_gen=50, pop_size=40, results_folder=None, n_workers=None):
    """
    Run NSGA-II multi-objective optimization with parallel execution

    Args:
        timeseries_data (DataFrame): Time series data
        date_time_index: Time index
        tech_params (DataFrame): Technology parameters
        n_gen (int): Number of generations
        pop_size (int): Population size
        results_folder (str): Path to save results
        n_workers (int): Number of parallel workers. If None, uses all available CPU cores.
                        Set to 1 to disable parallelization.

    Returns:
        tuple: (optimization results, problem object)
    """
    import multiprocessing

    # Determine number of workers
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()

    use_parallel = n_workers > 1

    print(f"Starting NSGA-II Optimization" + (" with Parallel Execution" if use_parallel else ""))
    print(f"  Generations: {n_gen}")
    print(f"  Population size: {pop_size}")
    print(f"  Total evaluations: {n_gen * pop_size}")
    if use_parallel:
        print(f"  Parallel workers: {n_workers} cores")
    else:
        print(f"  Parallelization: Disabled (sequential execution)")
    print("This may take several minutes...")

    # Create a multiprocessing pool if using parallel execution
    if use_parallel:
        pool = Pool(n_workers)
    else:
        pool = None

    try:
        # Define the problem with optional pool
        problem = EnergySystemOptimizationProblem(
            timeseries_data,
            date_time_index,
            tech_params,
            pool=pool
        )

        # Define the algorithm
        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=10),      # Lower eta for more exploration
            mutation=PM(eta=15, prob=0.3),        # Explicit 30% mutation probability
            eliminate_duplicates=True
        )

        # Define termination criterion
        termination = get_termination("n_gen", n_gen)

        # Custom callback to track generation number and convergence
        class GenerationCallback:
            def __init__(self, problem_ref):
                self.problem = problem_ref

            def __call__(self, algorithm):
                self.problem.current_generation = algorithm.n_gen

                # Track convergence history
                if algorithm.opt is not None and hasattr(algorithm.opt, 'get'):
                    F = algorithm.opt.get("F")
                    if F is not None and len(F) > 0:
                        convergence_data = {
                            'n_gen': algorithm.n_gen,
                            'n_eval': algorithm.evaluator.n_eval,
                            'best_cost': float(F[:, 0].min()),
                            'best_emissions': float(F[:, 1].min()),
                            'best_resilience': float(-F[:, 2].max()),  # Convert back to positive
                            'n_pareto': len(F)
                        }
                        self.problem.convergence_history.append(convergence_data)

        callback = GenerationCallback(problem)

        # Run optimization (disable save_history when using parallel execution to avoid pickling issues)
        res = minimize(
            problem,
            algorithm,
            termination,
            seed=42,
            verbose=True,
            save_history=False if use_parallel else True,
            callback=callback
        )

        print(f"\nOptimization complete!")
        print(f"Number of Pareto optimal solutions: {len(res.F)}")
        print(f"Total solutions evaluated: {len(problem.all_solutions)}")

        # Save all evaluated solutions to CSV
        if results_folder and len(problem.all_solutions) > 0:
            all_solutions_df = pd.DataFrame(problem.all_solutions)
            csv_path = os.path.join(results_folder, "all_evaluated_solutions.csv")
            all_solutions_df.to_csv(csv_path, index=False)
            print(f"Saved all {len(problem.all_solutions)} evaluated solutions to: {csv_path}")

        # Save convergence history to CSV
        if results_folder and len(problem.convergence_history) > 0:
            convergence_df = pd.DataFrame(problem.convergence_history)
            conv_csv_path = os.path.join(results_folder, "convergence_history.csv")
            convergence_df.to_csv(conv_csv_path, index=False)
            print(f"Saved convergence history to: {conv_csv_path}")

        return res, problem

    finally:
        # Clean up the pool
        if pool is not None:
            pool.close()
            pool.join()
