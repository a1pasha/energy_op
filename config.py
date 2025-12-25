#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration file for Energy System Optimization
Contains all paths, constants, and parameters
"""

import os
import pyomo.environ
from pyomo.opt import SolverFactory

# ============================================================================
# PATHS
# ============================================================================
# Get the directory where this config.py file is located
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_PATH, "data")
RESULTS_PATH = os.path.join(BASE_PATH, "nsga2_results")
TIMESERIES_FILE = os.path.join(DATA_PATH, "energy_system_dataset.csv")
TECHNOLOGY_FILE = os.path.join(DATA_PATH, "technology_parameters.csv")

# ============================================================================
# SOLVER CONFIGURATION
# ============================================================================
def get_available_solver():
    """Detect and return the first available solver (oemof compatible)"""
    available_solvers = []
    # Check oemof-compatible solvers only (highs not compatible with oemof's LP format)
    # Order: gurobi > cplex > cbc > glpk
    for solver in ['gurobi', 'cplex', 'cbc', 'glpk']:
        try:
            if SolverFactory(solver).available():
                available_solvers.append(solver)
        except:
            pass

    if available_solvers:
        print(f"Available solvers: {available_solvers}")
        return available_solvers[0]
    else:
        print("No solvers found, defaulting to glpk")
        return 'glpk'

SOLVER = get_available_solver()

# ============================================================================
# TIME SERIES PARAMETERS
# ============================================================================
START_DATE = "2025-01-01 00:00:00"
END_DATE = "2025-01-31 23:00:00"  # 1 month (744 hours)
# For quick testing, use: END_DATE = "2025-01-02 12:00:00"  # 1.5 days

# ============================================================================
# RESILIENCE CALCULATION PARAMETERS
# ============================================================================
# Stirling Index parameters
ALPHA = 1  # Weight for disparity in diversity calculation
BETA = 1   # Weight for variety and balance in diversity calculation

# Redundancy Index parameters (deprecated - now using MGA formula)
DISPERSION_WEIGHT = 0.5  # Weight for dispersion in redundancy calculation
CAPACITY_MARGIN_WEIGHT = 0.5  # Weight for capacity margin in redundancy calculation

# Resilience Score Weighting (for combining Stirling, Redundancy, Buffer, Balance)
# FIXED: Increased balance weight to 0.4 to better break emissions-resilience correlation
# Balance component is critical: prevents high-emission = high-resilience trap
RESILIENCE_WEIGHTS = {
    'stirling': 0.20,      # Technology diversity
    'redundancy': 0.20,    # Capacity margin
    'buffer': 0.20,        # Storage capacity
    'balance': 0.40        # INCREASED: Renewable-fossil balance (rewards mixing both types)
}

# Buffer Capacity Normalization (for reasonable range 0-1)
# Paper shows buffer capacity can range from ~5-16 hours
# Using 20 hours as upper bound for normalization
BUFFER_NORM_MAX = 20.0  # hours

# ============================================================================
# OPTIMIZATION PARAMETERS
# ============================================================================
# NSGA-II default parameters
DEFAULT_N_GENERATIONS = 150  # Thesis-quality run
DEFAULT_POPULATION_SIZE = 80

# Capacity scaling bounds for decision variables
# FIXED: Widened bounds to enable real variation while preventing zero-capacity exploits
CAPACITY_SCALE_MIN = 0.2  # Minimum 20% of nominal (prevents unrealistic zero-capacity solutions)
CAPACITY_SCALE_MAX = 3.0  # Maximum 300% of nominal (wider than old 2.0 to enable more exploration)

# ============================================================================
# TECHNOLOGY LISTS
# ============================================================================
# REMOVED: electrolyser - no H2 usage pathway
TECHNOLOGIES = [
    "pv", "gas_boiler", "gas_chp",
    "heat_pump_air", "heat_pump_ground"
]

# REMOVED: hydrogen_storage - no H2 usage pathway
STORAGES = [
    "battery_storage",
    "thermal_storage"
]

# ============================================================================
# VISUALIZATION PARAMETERS
# ============================================================================
PLOT_DPI = 300
PLOT_STYLE = 'seaborn-v0_8-darkgrid'  # Matplotlib style

# ============================================================================
# DUMMY DATA GENERATION PARAMETERS
# ============================================================================
BASE_DEMAND_ELECTRICITY_KW = 100
BASE_DEMAND_HEAT_KW = 80
RANDOM_SEED = 42  # For reproducibility
