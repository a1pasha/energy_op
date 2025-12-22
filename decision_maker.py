#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Decision making methods for selecting single best solution from Pareto front
Implements Knee Point and Distance from Ideal methods
"""

import numpy as np
from scipy.spatial.distance import cdist


def normalize_objectives(F):
    """
    Normalize objectives to 0-1 range

    Args:
        F (ndarray): Objective values (n_solutions x n_objectives)

    Returns:
        ndarray: Normalized objectives
    """
    F_norm = np.copy(F)
    for i in range(F.shape[1]):
        min_val = np.min(F[:, i])
        max_val = np.max(F[:, i])
        if max_val - min_val > 0:
            F_norm[:, i] = (F[:, i] - min_val) / (max_val - min_val)
        else:
            F_norm[:, i] = 0
    return F_norm


def find_knee_point(F, objective_pair='cost_emissions'):
    """
    Find the knee point in the Pareto front
    The knee point is where the trade-off curve has maximum curvature
    This represents the best compromise solution

    Method: Uses perpendicular distance from line connecting extreme points

    Args:
        F (ndarray): Objective values (n_solutions x n_objectives)
                     Assumes minimization for all objectives
        objective_pair (str): Which objectives to analyze for knee point
                             Options: 'cost_emissions', 'cost_resilience', 'emissions_resilience'

    Returns:
        int: Index of knee point solution
    """
    # Normalize objectives to 0-1 range
    F_norm = normalize_objectives(F)

    # For 3D objectives, project to 2D for knee detection
    # Select which 2 objectives to analyze
    if F_norm.shape[1] == 3:
        if objective_pair == 'cost_emissions':
            F_2d = F_norm[:, [0, 1]]  # Cost vs Emissions (original)
        elif objective_pair == 'cost_resilience':
            F_2d = F_norm[:, [0, 2]]  # Cost vs Resilience
        elif objective_pair == 'emissions_resilience':
            F_2d = F_norm[:, [1, 2]]  # Emissions vs Resilience
        else:
            F_2d = F_norm[:, :2]  # Default to cost vs emissions
    else:
        F_2d = F_norm

    # Find extreme points
    idx_min_obj1 = np.argmin(F_2d[:, 0])  # Best in objective 1
    idx_min_obj2 = np.argmin(F_2d[:, 1])  # Best in objective 2

    point_a = F_2d[idx_min_obj1]
    point_b = F_2d[idx_min_obj2]

    # Calculate perpendicular distance from each point to line AB
    # Distance = |cross_product| / |AB|
    distances = []
    for i in range(len(F_2d)):
        point = F_2d[i]
        # Vector from A to point
        ap = point - point_a
        # Vector from A to B
        ab = point_b - point_a

        # Perpendicular distance
        if np.linalg.norm(ab) > 0:
            distance = np.abs(np.cross(ap, ab)) / np.linalg.norm(ab)
        else:
            distance = 0
        distances.append(distance)

    # Knee point is the one with maximum distance
    knee_idx = np.argmax(distances)

    return knee_idx


def find_ideal_point_solution(F):
    """
    Find solution closest to ideal point (utopia point)
    Ideal point = best value in each objective (usually unattainable)

    Method: Euclidean distance to ideal point in normalized space

    Args:
        F (ndarray): Objective values (n_solutions x n_objectives)
                     Assumes minimization for all objectives

    Returns:
        int: Index of solution closest to ideal point
    """
    # Normalize objectives to 0-1 range
    F_norm = normalize_objectives(F)

    # Ideal point (best in each objective) = all zeros in normalized space
    ideal_point = np.zeros(F_norm.shape[1])

    # Calculate Euclidean distance to ideal point for each solution
    distances = np.linalg.norm(F_norm - ideal_point, axis=1)

    # Solution with minimum distance
    ideal_idx = np.argmin(distances)

    return ideal_idx


def get_recommended_solutions(res, knee_method='cost_emissions'):
    """
    Get recommended solutions using multiple methods

    Args:
        res: Optimization result from pymoo
        knee_method (str): Which knee point to use
                          Options: 'cost_emissions', 'cost_resilience', 'emissions_resilience', 'all'

    Returns:
        dict: Dictionary with knee and ideal solution indices and details
    """
    F = res.F  # Objective values
    X = res.X  # Decision variables

    # Find solutions using different methods
    ideal_idx = find_ideal_point_solution(F)

    results = {
        'ideal_distance': {
            'index': ideal_idx,
            'objectives': F[ideal_idx],
            'decision_vars': X[ideal_idx],
            'description': 'Closest to ideal point (balanced optimization)'
        }
    }

    # Get knee points for all three combinations if 'all' is selected
    if knee_method == 'all':
        knee_cost_emissions_idx = find_knee_point(F, 'cost_emissions')
        knee_cost_resilience_idx = find_knee_point(F, 'cost_resilience')
        knee_emissions_resilience_idx = find_knee_point(F, 'emissions_resilience')

        results['knee_cost_emissions'] = {
            'index': knee_cost_emissions_idx,
            'objectives': F[knee_cost_emissions_idx],
            'decision_vars': X[knee_cost_emissions_idx],
            'description': 'Best trade-off between cost and emissions'
        }
        results['knee_cost_resilience'] = {
            'index': knee_cost_resilience_idx,
            'objectives': F[knee_cost_resilience_idx],
            'decision_vars': X[knee_cost_resilience_idx],
            'description': 'Best trade-off between cost and resilience'
        }
        results['knee_emissions_resilience'] = {
            'index': knee_emissions_resilience_idx,
            'objectives': F[knee_emissions_resilience_idx],
            'decision_vars': X[knee_emissions_resilience_idx],
            'description': 'Best trade-off between emissions and resilience'
        }

        # Default knee point for backward compatibility
        results['knee_point'] = results['knee_cost_emissions']

    else:
        # Single knee point based on selection
        knee_idx = find_knee_point(F, knee_method)

        description_map = {
            'cost_emissions': 'Best trade-off between cost and emissions',
            'cost_resilience': 'Best trade-off between cost and resilience',
            'emissions_resilience': 'Best trade-off between emissions and resilience'
        }

        results['knee_point'] = {
            'index': knee_idx,
            'objectives': F[knee_idx],
            'decision_vars': X[knee_idx],
            'description': description_map.get(knee_method, 'Maximum trade-off efficiency')
        }

    return results


def print_recommended_solutions(results):
    """
    Print recommended solutions in a formatted way

    Args:
        results: Dictionary from get_recommended_solutions()
    """
    print("\n" + "="*80)
    print("RECOMMENDED SINGLE BEST SOLUTIONS")
    print("="*80)

    method_num = 1

    # Check if all knee points are present
    has_all_knees = ('knee_cost_emissions' in results and
                     'knee_cost_resilience' in results and
                     'knee_emissions_resilience' in results)

    if has_all_knees:
        # Print all three knee points
        knee_ce = results['knee_cost_emissions']
        print(f"\n[METHOD {method_num}: KNEE POINT - Cost vs Emissions]")
        print(f"Description: {knee_ce['description']}")
        print(f"Solution Index: {knee_ce['index']}")
        print(f"Objectives:")
        print(f"  - Total Cost: ${knee_ce['objectives'][0]:,.2f}")
        print(f"  - CO2 Emissions: {knee_ce['objectives'][1]:,.2f} kg")
        print(f"  - Resilience Score: {-knee_ce['objectives'][2]:.4f}")  # Negate back to positive
        method_num += 1

        knee_cr = results['knee_cost_resilience']
        print(f"\n[METHOD {method_num}: KNEE POINT - Cost vs Resilience]")
        print(f"Description: {knee_cr['description']}")
        print(f"Solution Index: {knee_cr['index']}")
        print(f"Objectives:")
        print(f"  - Total Cost: ${knee_cr['objectives'][0]:,.2f}")
        print(f"  - CO2 Emissions: {knee_cr['objectives'][1]:,.2f} kg")
        print(f"  - Resilience Score: {-knee_cr['objectives'][2]:.4f}")
        method_num += 1

        knee_er = results['knee_emissions_resilience']
        print(f"\n[METHOD {method_num}: KNEE POINT - Emissions vs Resilience]")
        print(f"Description: {knee_er['description']}")
        print(f"Solution Index: {knee_er['index']}")
        print(f"Objectives:")
        print(f"  - Total Cost: ${knee_er['objectives'][0]:,.2f}")
        print(f"  - CO2 Emissions: {knee_er['objectives'][1]:,.2f} kg")
        print(f"  - Resilience Score: {-knee_er['objectives'][2]:.4f}")
        method_num += 1

    else:
        # Print single knee point
        knee = results['knee_point']
        print(f"\n[METHOD {method_num}: KNEE POINT]")
        print(f"Description: {knee['description']}")
        print(f"Solution Index: {knee['index']}")
        print(f"Objectives:")
        print(f"  - Total Cost: ${knee['objectives'][0]:,.2f}")
        print(f"  - CO2 Emissions: {knee['objectives'][1]:,.2f} kg")
        print(f"  - Resilience Score: {-knee['objectives'][2]:.4f}")
        method_num += 1

    # Ideal Distance
    ideal = results['ideal_distance']
    print(f"\n[METHOD {method_num}: DISTANCE FROM IDEAL]")
    print(f"Description: {ideal['description']}")
    print(f"Solution Index: {ideal['index']}")
    print(f"Objectives:")
    print(f"  - Total Cost: ${ideal['objectives'][0]:,.2f}")
    print(f"  - CO2 Emissions: {ideal['objectives'][1]:,.2f} kg")
    print(f"  - Resilience Score: {-ideal['objectives'][2]:.4f}")

    # Comparison
    print("\n" + "-"*80)
    print("RECOMMENDATION GUIDE")
    print("-"*80)
    if has_all_knees:
        print("Choose based on your priority:")
        print("  - Cost-Emissions balance -> Knee Point (Cost vs Emissions)")
        print("  - Cost-Resilience balance -> Knee Point (Cost vs Resilience)")
        print("  - Emissions-Resilience balance -> Knee Point (Emissions vs Resilience)")
        print("  - Equal weight to all 3 objectives -> Distance from Ideal")
    else:
        print("Which to choose?")
        print("  - If prioritizing the specific trade-off -> Use Knee Point")
        print("  - If equal weight to all objectives -> Use Ideal Distance")

    print("="*80)
