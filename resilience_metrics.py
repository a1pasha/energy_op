#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Resilience metrics calculation module
Implements Stirling Index, Redundancy, and Buffer Capacity (based on MGA paper)
"""

import numpy as np
import itertools
from config import (
    ALPHA, BETA
)


def calculate_stirling_index(generation_mix, tech_params):
    """
    Calculate the Stirling Index for diversity assessment
    Considers both variety (number of technologies) and disparity (how different they are)

    Args:
        generation_mix (dict): Dictionary of {technology: generation_amount}
        tech_params (DataFrame): Technology parameters with disparity_factor

    Returns:
        float: Stirling diversity index
    """
    total_generation = sum(generation_mix.values())
    if total_generation == 0:
        return 0

    normalized_mix = {tech: gen / total_generation for tech, gen in generation_mix.items()}

    disparity_values = {tech: tech_params.loc[tech, "disparity_factor"]
                        for tech in generation_mix.keys() if tech in tech_params.index}

    stirling_index = 0
    for tech1, tech2 in itertools.combinations(normalized_mix.keys(), 2):
        if tech1 in disparity_values and tech2 in disparity_values:
            disparity = abs(disparity_values[tech1] - disparity_values[tech2])**ALPHA
            prop_product = (normalized_mix[tech1] * normalized_mix[tech2])**BETA
            stirling_index += disparity * prop_product

    return stirling_index


def calculate_redundancy(generation_capacity, demand, tech_params):
    """
    Calculate redundancy index based on MGA paper Equation 3
    R = (1 - P_demand/P_inst)^α · Σ(P_i/P_total)^(2·β)

    Grid_electricity is already excluded from generation_capacity dict in utils.py

    Args:
        generation_capacity (dict): Dictionary of {technology: capacity} (excludes grid)
        demand (float): Peak demand in kW
        tech_params (DataFrame): Technology parameters with dispatchable flag

    Returns:
        float: Redundancy index
    """
    # Get dispatchable technologies only
    # (grid_electricity already excluded from generation_capacity dict)
    dispatchable_techs = [tech for tech in generation_capacity.keys()
                         if tech in tech_params.index and tech_params.loc[tech, "dispatchable"] == 1]

    # Total secured installed capacity (dispatchable only)
    P_inst_total = sum([generation_capacity[tech] for tech in dispatchable_techs])

    if P_inst_total == 0 or demand == 0:
        return 0

    # First term: (1 - P_demand/P_inst)^α
    capacity_margin_term = (1 - demand / P_inst_total)**ALPHA if P_inst_total > demand else 0

    # Second term: Σ(P_i/P_total)^(2·β) - distribution of capacity
    dispersion_sum = 0
    for tech in dispatchable_techs:
        P_i = generation_capacity[tech]
        dispersion_sum += (P_i / P_inst_total)**(2 * BETA)

    # Combine according to paper formula (Equation 3, page 9)
    redundancy_index = capacity_margin_term * dispersion_sum

    return redundancy_index


def calculate_buffer_capacity(storage_capacity, demand):
    """
    Calculate buffer capacity index based on MGA paper Equation 4
    BSC = E_inst,BSC / P_demand

    Represents hours of storage available to maintain system performance

    Args:
        storage_capacity (dict): Dictionary of {storage_type: capacity_kwh}
        demand (float): Peak demand in kW

    Returns:
        float: Buffer capacity index (hours of storage available)
    """
    # E_inst,BSC: Total installed buffer and storage capacity in kWh
    E_inst_BSC = sum(storage_capacity.values())

    # BSC = E_inst,BSC / P_demand (Equation 4, page 9)
    # Result is in hours - how long the system can maintain max load from storage alone
    buffer_capacity_index = E_inst_BSC / demand if demand > 0 else 0

    return buffer_capacity_index


def calculate_renewable_fossil_balance(generation_capacity, tech_params):
    """
    Calculate renewable-fossil balance index to break emissions-resilience correlation

    This metric rewards systems that have BOTH renewable and dispatchable fossil capacity,
    enabling high resilience with varying emission levels.

    Balance = 4 * (P_renewable / P_total) * (P_fossil / P_total)

    Maximum value (1.0) when renewable and fossil are equal (50%-50%)
    Low value when dominated by one type (prevents correlation with emissions)

    Args:
        generation_capacity (dict): Dictionary of {technology: capacity} (excludes grid)
        tech_params (DataFrame): Technology parameters

    Returns:
        float: Balance index (0 to 1, where 1 = perfect 50-50 mix)
    """
    # Define renewable technologies (zero emissions)
    renewable_techs = ['pv', 'heat_pump_air', 'heat_pump_ground']

    # Define fossil/dispatchable technologies (have emissions)
    fossil_techs = ['gas_boiler', 'gas_chp']

    # Calculate total capacity for each type
    P_renewable = sum([generation_capacity.get(tech, 0) for tech in renewable_techs])
    P_fossil = sum([generation_capacity.get(tech, 0) for tech in fossil_techs])
    P_total = P_renewable + P_fossil

    if P_total == 0:
        return 0

    # Calculate balance: maximum when 50-50 split
    # Formula: 4 * p * (1-p) where p = renewable fraction
    renewable_fraction = P_renewable / P_total
    fossil_fraction = P_fossil / P_total

    balance_index = 4 * renewable_fraction * fossil_fraction

    return balance_index


