#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization module for optimization results
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

from config import RESULTS_PATH, PLOT_DPI
from decision_maker import find_knee_point, find_ideal_point_solution


def plot_pareto_front_3d(res, save_path=None):
    """
    Create 3D visualization of Pareto front

    Args:
        res: NSGA-II optimization results
        save_path (str): Path to save the plot
    """
    F = res.F
    cost = F[:, 0]
    emissions = F[:, 1]
    resilience = -F[:, 2]  # Convert back to positive

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(cost, emissions, resilience, c=resilience, cmap='viridis',
                        s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Total Cost', fontsize=12, labelpad=10)
    ax.set_ylabel('CO2 Emissions (kg)', fontsize=12, labelpad=10)
    ax.set_zlabel('Resilience Score', fontsize=12, labelpad=10)
    ax.set_title('3D Pareto Front: Cost vs. Emissions vs. Resilience',
                 fontsize=14, pad=20)

    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('Resilience Score', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches='tight')
        print(f"Saved 3D Pareto front to: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_pareto_front_2d(res, save_path=None):
    """
    Create 2D projections of Pareto front

    Args:
        res: NSGA-II optimization results
        save_path (str): Path to save the plot
    """
    F = res.F
    cost = F[:, 0]
    emissions = F[:, 1]
    resilience = -F[:, 2]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Cost vs Emissions
    axes[0].scatter(cost, emissions, c=resilience, cmap='viridis', s=80,
                   alpha=0.6, edgecolors='black', linewidth=0.5)
    axes[0].set_xlabel('Total Cost', fontsize=11)
    axes[0].set_ylabel('CO2 Emissions (kg)', fontsize=11)
    axes[0].set_title('Cost vs. Emissions', fontsize=12)
    axes[0].grid(True, alpha=0.3)

    # Cost vs Resilience
    axes[1].scatter(cost, resilience, c=emissions, cmap='plasma', s=80,
                   alpha=0.6, edgecolors='black', linewidth=0.5)
    axes[1].set_xlabel('Total Cost', fontsize=11)
    axes[1].set_ylabel('Resilience Score', fontsize=11)
    axes[1].set_title('Cost vs. Resilience', fontsize=12)
    axes[1].grid(True, alpha=0.3)

    # Emissions vs Resilience
    scatter = axes[2].scatter(emissions, resilience, c=cost, cmap='coolwarm', s=80,
                             alpha=0.6, edgecolors='black', linewidth=0.5)
    axes[2].set_xlabel('CO2 Emissions (kg)', fontsize=11)
    axes[2].set_ylabel('Resilience Score', fontsize=11)
    axes[2].set_title('Emissions vs. Resilience', fontsize=12)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches='tight')
        print(f"Saved 2D Pareto projections to: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_decision_variables(res, problem, save_path=None, n_solutions=10):
    """
    Plot decision variables for selected Pareto solutions

    Args:
        res: NSGA-II optimization results
        problem: Problem object
        save_path (str): Path to save the plot
        n_solutions (int): Number of solutions to plot
    """
    n_solutions_to_plot = min(n_solutions, len(res.X))
    indices = np.linspace(0, len(res.X)-1, n_solutions_to_plot, dtype=int)

    fig, ax = plt.subplots(figsize=(14, 8))

    tech_labels = problem.technologies + problem.storages
    x_pos = np.arange(len(tech_labels))
    width = 0.08

    colors = cm.tab10(np.linspace(0, 1, n_solutions_to_plot))

    for i, idx in enumerate(indices):
        offset = (i - n_solutions_to_plot/2) * width
        ax.bar(x_pos + offset, res.X[idx], width, label=f'Solution {idx+1}',
              alpha=0.8, color=colors[i])

    ax.set_xlabel('Technologies and Storage', fontsize=12)
    ax.set_ylabel('Capacity Scaling Factor', fontsize=12)
    ax.set_title('Decision Variables (Capacity Scaling) for Selected Pareto Solutions',
                 fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(tech_labels, rotation=45, ha='right')
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.7,
               label='Baseline')
    ax.legend(loc='upper right', fontsize=9, ncol=2)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches='tight')
        print(f"Saved decision variables plot to: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_convergence(res, problem=None, save_path=None):
    """
    Plot convergence curves for all objectives

    Args:
        res: NSGA-II optimization results
        problem: Problem object with convergence_history (optional)
        save_path (str): Path to save the plot
    """
    # Try to use custom convergence history first (for parallel execution)
    if problem is not None and hasattr(problem, 'convergence_history') and len(problem.convergence_history) > 0:
        conv_history = problem.convergence_history
        n_evals = [entry['n_eval'] for entry in conv_history]
        opt_cost = [entry['best_cost'] for entry in conv_history]
        opt_emissions = [entry['best_emissions'] for entry in conv_history]
        opt_resilience = [entry['best_resilience'] for entry in conv_history]
    # Fallback to pymoo history
    elif hasattr(res, 'history') and res.history is not None:
        n_evals = []
        opt_cost = []
        opt_emissions = []
        opt_resilience = []

        for entry in res.history:
            n_evals.append(entry.evaluator.n_eval)
            F_gen = entry.opt.get("F")
            opt_cost.append(F_gen[:, 0].min())
            opt_emissions.append(F_gen[:, 1].min())
            opt_resilience.append(-F_gen[:, 2].max())  # Convert back to positive
    else:
        print("No convergence history available for convergence plot")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(n_evals, opt_cost, linewidth=2, color='blue', marker='o', markersize=4)
    axes[0].set_xlabel('Function Evaluations', fontsize=11)
    axes[0].set_ylabel('Best Cost', fontsize=11)
    axes[0].set_title('Cost Convergence', fontsize=12)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(n_evals, opt_emissions, linewidth=2, color='green', marker='o', markersize=4)
    axes[1].set_xlabel('Function Evaluations', fontsize=11)
    axes[1].set_ylabel('Best Emissions (kg)', fontsize=11)
    axes[1].set_title('Emissions Convergence', fontsize=12)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(n_evals, opt_resilience, linewidth=2, color='orange', marker='o', markersize=4)
    axes[2].set_xlabel('Function Evaluations', fontsize=11)
    axes[2].set_ylabel('Best Resilience Score', fontsize=11)
    axes[2].set_title('Resilience Convergence', fontsize=12)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches='tight')
        print(f"Saved convergence plot to: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_pareto_with_recommendations(res, knee_idx, ideal_idx, save_path=None,
                                     knee_ce_idx=None, knee_cr_idx=None, knee_er_idx=None):
    """
    Plot 3D Pareto front with highlighted recommended solutions

    Args:
        res: NSGA-II optimization results
        knee_idx (int): Index of knee point solution (backward compatibility)
        ideal_idx (int): Index of ideal distance solution
        save_path (str): Path to save the plot
        knee_ce_idx (int): Index of cost-emissions knee point
        knee_cr_idx (int): Index of cost-resilience knee point
        knee_er_idx (int): Index of emissions-resilience knee point
    """
    F = res.F
    cost = F[:, 0]
    emissions = F[:, 1]
    resilience = -F[:, 2]  # Convert back to positive

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot all Pareto solutions
    scatter = ax.scatter(cost, emissions, resilience, c=resilience, cmap='viridis',
                        s=80, alpha=0.4, edgecolors='gray', linewidth=0.5,
                        label='Pareto Solutions')

    # Check if all three knee points are provided
    if knee_ce_idx is not None and knee_cr_idx is not None and knee_er_idx is not None:
        # Plot all three knee points with different shapes
        # Knee Point 1: Cost-Emissions (Square)
        ax.scatter(cost[knee_ce_idx], emissions[knee_ce_idx], resilience[knee_ce_idx],
                  c='blue', s=80, marker='s', edgecolors='black', linewidth=1.5,
                  label=f'Knee: Cost-Emissions (#{knee_ce_idx})', zorder=5)

        # Knee Point 2: Cost-Resilience (Triangle)
        ax.scatter(cost[knee_cr_idx], emissions[knee_cr_idx], resilience[knee_cr_idx],
                  c='green', s=80, marker='^', edgecolors='black', linewidth=1.5,
                  label=f'Knee: Cost-Resilience (#{knee_cr_idx})', zorder=5)

        # Knee Point 3: Emissions-Resilience (Pentagon)
        ax.scatter(cost[knee_er_idx], emissions[knee_er_idx], resilience[knee_er_idx],
                  c='orange', s=80, marker='p', edgecolors='black', linewidth=1.5,
                  label=f'Knee: Emissions-Resilience (#{knee_er_idx})', zorder=5)

        # Ideal Distance (Red Circle)
        ax.scatter(cost[ideal_idx], emissions[ideal_idx], resilience[ideal_idx],
                  c='red', s=80, marker='o', edgecolors='black', linewidth=1.5,
                  label=f'Ideal Distance (#{ideal_idx})', zorder=5)

        title_text = ('3D Pareto Front with Recommended Solutions\n'
                     'Blue Square=Cost-Emissions | Green Triangle=Cost-Resilience | '
                     'Orange Pentagon=Emissions-Resilience | Red Circle=Ideal')

    else:
        # Backward compatibility: use single knee point
        ax.scatter(cost[knee_idx], emissions[knee_idx], resilience[knee_idx],
                  c='blue', s=80, marker='s', edgecolors='black', linewidth=1.5,
                  label=f'Knee Point (#{knee_idx})', zorder=5)

        ax.scatter(cost[ideal_idx], emissions[ideal_idx], resilience[ideal_idx],
                  c='red', s=80, marker='o', edgecolors='black', linewidth=1.5,
                  label=f'Ideal Distance (#{ideal_idx})', zorder=5)

        title_text = ('3D Pareto Front with Recommended Solutions\n'
                     'Blue Square=Knee Point | Red Circle=Ideal Distance')

    ax.set_xlabel('Total Cost', fontsize=12, labelpad=10)
    ax.set_ylabel('CO2 Emissions (kg)', fontsize=12, labelpad=10)
    ax.set_zlabel('Resilience Score', fontsize=12, labelpad=10)
    ax.set_title(title_text, fontsize=13, pad=20)

    ax.legend(loc='upper left', fontsize=9)

    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('Resilience Score', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches='tight')
        print(f"Saved Pareto front with recommendations to: {save_path}")
    else:
        plt.show()

    plt.close()


def save_pareto_solutions_csv(res, problem, save_path=None, knee_idx=None, ideal_idx=None,
                              knee_ce_idx=None, knee_cr_idx=None, knee_er_idx=None):
    """
    Save Pareto optimal solutions to CSV

    Args:
        res: NSGA-II optimization results
        problem: Problem object
        save_path (str): Path to save CSV
        knee_idx (int): Index of knee point solution (backward compatibility)
        ideal_idx (int): Index of ideal distance solution
        knee_ce_idx (int): Index of cost-emissions knee point
        knee_cr_idx (int): Index of cost-resilience knee point
        knee_er_idx (int): Index of emissions-resilience knee point
    """
    tech_labels = problem.technologies + problem.storages
    df = pd.DataFrame(res.X, columns=tech_labels)

    df['Cost'] = res.F[:, 0]
    df['Emissions_kg'] = res.F[:, 1]
    df['Resilience'] = -res.F[:, 2]

    # Add recommendation columns
    if knee_ce_idx is not None:
        df['Knee_Cost_Emissions'] = ['YES' if i == knee_ce_idx else 'NO' for i in range(len(df))]
    if knee_cr_idx is not None:
        df['Knee_Cost_Resilience'] = ['YES' if i == knee_cr_idx else 'NO' for i in range(len(df))]
    if knee_er_idx is not None:
        df['Knee_Emissions_Resilience'] = ['YES' if i == knee_er_idx else 'NO' for i in range(len(df))]

    # Backward compatibility
    if knee_idx is not None and knee_ce_idx is None:
        df['Knee_Point'] = ['YES' if i == knee_idx else 'NO' for i in range(len(df))]

    if ideal_idx is not None:
        df['Ideal_Distance'] = ['YES' if i == ideal_idx else 'NO' for i in range(len(df))]

    if save_path:
        df.to_csv(save_path, index_label='Solution_ID')
        print(f"Saved Pareto solutions to: {save_path}")
    else:
        default_path = os.path.join(RESULTS_PATH, 'pareto_solutions.csv')
        df.to_csv(default_path, index_label='Solution_ID')
        print(f"Saved Pareto solutions to: {default_path}")

    return df


def print_pareto_summary(res):
    """
    Print summary statistics of Pareto front

    Args:
        res: NSGA-II optimization results
    """
    print("\n" + "="*70)
    print("PARETO FRONT SUMMARY STATISTICS")
    print("="*70)
    print(f"\nNumber of Pareto optimal solutions: {len(res.F)}")
    print(f"\nCost Range: ${res.F[:, 0].min():.2f} - ${res.F[:, 0].max():.2f}")
    print(f"Emissions Range: {res.F[:, 1].min():.2f} - {res.F[:, 1].max():.2f} kg CO2")
    print(f"Resilience Range: {-res.F[:, 2].max():.4f} - {-res.F[:, 2].min():.4f}")

    # Identify extreme solutions
    idx_min_cost = res.F[:, 0].argmin()
    idx_min_emissions = res.F[:, 1].argmin()
    idx_max_resilience = res.F[:, 2].argmin()

    print("\n" + "-"*70)
    print("EXTREME SOLUTIONS:")
    print("-"*70)

    print(f"\nMost Cost-Effective Solution (ID {idx_min_cost}):")
    print(f"  Cost: ${res.F[idx_min_cost, 0]:.2f}")
    print(f"  Emissions: {res.F[idx_min_cost, 1]:.2f} kg CO2")
    print(f"  Resilience: {-res.F[idx_min_cost, 2]:.4f}")

    print(f"\nLowest Emissions Solution (ID {idx_min_emissions}):")
    print(f"  Cost: ${res.F[idx_min_emissions, 0]:.2f}")
    print(f"  Emissions: {res.F[idx_min_emissions, 1]:.2f} kg CO2")
    print(f"  Resilience: {-res.F[idx_min_emissions, 2]:.4f}")

    print(f"\nMost Resilient Solution (ID {idx_max_resilience}):")
    print(f"  Cost: ${res.F[idx_max_resilience, 0]:.2f}")
    print(f"  Emissions: {res.F[idx_max_resilience, 1]:.2f} kg CO2")
    print(f"  Resilience: {-res.F[idx_max_resilience, 2]:.4f}")
    print("="*70)


def generate_all_plots(res, problem, knee_idx=None, ideal_idx=None, results_folder=None,
                       recommended_solutions=None):
    """
    Generate all visualization plots and save to results directory

    Args:
        res: NSGA-II optimization results
        problem: Problem object
        knee_idx (int): Index of knee point solution (backward compatibility)
        ideal_idx (int): Index of ideal distance solution
        results_folder (str): Custom results folder path (optional)
        recommended_solutions (dict): Dictionary with all recommended solutions
    """
    # Use custom folder if provided, otherwise use default
    save_path = results_folder if results_folder else RESULTS_PATH
    os.makedirs(save_path, exist_ok=True)

    print("\nGenerating visualizations...")

    plot_pareto_front_3d(res, os.path.join(save_path, 'pareto_front_3d.png'))
    plot_pareto_front_2d(res, os.path.join(save_path, 'pareto_front_2d_projections.png'))

    # Extract knee point indices if recommended_solutions is provided
    knee_ce_idx = None
    knee_cr_idx = None
    knee_er_idx = None

    if recommended_solutions is not None:
        if 'knee_cost_emissions' in recommended_solutions:
            knee_ce_idx = recommended_solutions['knee_cost_emissions']['index']
        if 'knee_cost_resilience' in recommended_solutions:
            knee_cr_idx = recommended_solutions['knee_cost_resilience']['index']
        if 'knee_emissions_resilience' in recommended_solutions:
            knee_er_idx = recommended_solutions['knee_emissions_resilience']['index']
        if 'ideal_distance' in recommended_solutions:
            ideal_idx = recommended_solutions['ideal_distance']['index']

    # Add plot with recommended solutions highlighted
    if ideal_idx is not None:
        if knee_ce_idx is not None and knee_cr_idx is not None and knee_er_idx is not None:
            # Plot all three knee points
            plot_pareto_with_recommendations(res, knee_idx, ideal_idx,
                                            os.path.join(save_path, 'pareto_front_with_recommendations.png'),
                                            knee_ce_idx, knee_cr_idx, knee_er_idx)
        elif knee_idx is not None:
            # Backward compatibility: single knee point
            plot_pareto_with_recommendations(res, knee_idx, ideal_idx,
                                            os.path.join(save_path, 'pareto_front_with_recommendations.png'))

    plot_decision_variables(res, problem, os.path.join(save_path, 'decision_variables_distribution.png'))
    plot_convergence(res, problem, os.path.join(save_path, 'convergence_plot.png'))
    save_pareto_solutions_csv(res, problem, os.path.join(save_path, 'pareto_solutions.csv'),
                             knee_idx, ideal_idx, knee_ce_idx, knee_cr_idx, knee_er_idx)
    print_pareto_summary(res)

    print(f"\nAll visualizations saved to: {save_path}")
