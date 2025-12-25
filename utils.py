#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for result extraction and processing
"""

import pandas as pd
import pyomo.environ
import oemof.solph.processing as processing
from resilience_metrics import (
    calculate_stirling_index,
    calculate_redundancy, calculate_buffer_capacity,
    calculate_renewable_fossil_balance
)


def calculate_emissions_time_varying(results, timeseries_data, tech_params):
    """
    Calculate total emissions with time-varying grid emission factor

    Args:
        results: oemof results dictionary
        timeseries_data (DataFrame): Timeseries with grid_emission_factor column
        tech_params (DataFrame): Technology parameters

    Returns:
        float: Total emissions (kg CO2)
    """
    total_emissions = 0

    for key, flow_data in results.items():
        if isinstance(key, tuple) and len(key) == 2:
            source, target = key

            if hasattr(source, 'label'):
                tech_name = source.label

                if 'sequences' in flow_data:
                    flow_series = flow_data['sequences']

                    # Check if it's a DataFrame or Series
                    if isinstance(flow_series, pd.DataFrame) and 'flow' in flow_series.columns:
                        flow_values = flow_series['flow']
                    elif isinstance(flow_series, pd.Series):
                        flow_values = flow_series
                    else:
                        continue

                    # GRID ELECTRICITY: Use time-varying emissions
                    if tech_name == 'grid_electricity' and 'grid_emission_factor' in timeseries_data.columns:
                        # Multiply hourly grid usage by hourly emission factor
                        hourly_emissions = flow_values * timeseries_data['grid_emission_factor'].values[:len(flow_values)]
                        emissions = hourly_emissions.sum()
                        total_emissions += emissions

                    # OTHER TECHNOLOGIES: Use fixed emissions from tech_params
                    elif tech_name in tech_params.index and 'co2_emissions_kg_per_kwh' in tech_params.columns:
                        emissions = flow_values.sum() * tech_params.loc[tech_name, 'co2_emissions_kg_per_kwh']
                        total_emissions += emissions

    return total_emissions


def extract_results_from_model(energy_system, model, timeseries_data, tech_params):
    """
    Extract and process results from solved oemof model

    Args:
        energy_system: oemof energy system object
        model: Solved oemof model
        timeseries_data (DataFrame): Original timeseries data
        tech_params (DataFrame): Technology parameters

    Returns:
        dict: Dictionary containing all results and metrics
    """
    # Try to use oemof's processing module
    try:
        results = processing.results(model)
    except Exception as e:
        print(f"Warning: Could not use oemof processing: {e}")
        # Fallback to manual extraction
        results = _manual_result_extraction(model)

    # Extract generation data
    electricity_generation = {}
    heat_generation = {}

    for key, flow_data in results.items():
        if isinstance(key, tuple) and len(key) == 2:
            source, target = key

            if target is not None and hasattr(target, 'label'):
                if target.label == "electricity_bus":
                    tech_name = source.label if hasattr(source, 'label') else str(source)
                    if 'sequences' in flow_data:
                        # sequences is a DataFrame; sum and get 'flow' column
                        flow_sum = flow_data['sequences'].sum()
                        if 'flow' in flow_sum:
                            electricity_generation[tech_name] = flow_sum['flow']

                elif target.label == "heat_bus":
                    tech_name = source.label if hasattr(source, 'label') else str(source)
                    if 'sequences' in flow_data:
                        flow_sum = flow_data['sequences'].sum()
                        if 'flow' in flow_sum:
                            heat_generation[tech_name] = flow_sum['flow']

    # Combine electricity and heat generation
    # EXCLUDE grid_electricity for resilience calculations
    total_generation = {}
    for tech, value in electricity_generation.items():
        if tech != 'grid_electricity':  # Exclude grid
            total_generation[tech] = value

    for tech, value in heat_generation.items():
        if tech != 'grid_electricity':  # Exclude grid
            if tech in total_generation:
                total_generation[tech] += value
            else:
                total_generation[tech] = value

    # Calculate demands
    max_electricity_demand = max(timeseries_data["demand_electricity_kw"])
    max_heat_demand = max(timeseries_data["demand_heat_kw"])
    max_total_demand = max_electricity_demand + max_heat_demand

    # Get capacities (wind and fuel_cell removed)
    # Generation capacity dict - EXCLUDES grid_electricity for resilience calculations
    # Grid is not optimized and should not be counted in redundancy/diversity metrics
    generation_capacity = {
        "pv": tech_params.loc["pv", "capacity_kw"],
        "gas_boiler": tech_params.loc["gas_boiler", "capacity_kw"],
        "gas_chp": tech_params.loc["gas_chp", "capacity_kw"],
        "heat_pump_air": tech_params.loc["heat_pump_air", "capacity_kw"],
        "heat_pump_ground": tech_params.loc["heat_pump_ground", "capacity_kw"],
    }

    storage_capacity = {
        "battery_storage": tech_params.loc["battery_storage", "storage_capacity_kwh"],
        "thermal_storage": tech_params.loc["thermal_storage", "storage_capacity_kwh"],
    }

    # Calculate resilience metrics (Shannon Index removed)
    stirling_index = calculate_stirling_index(total_generation, tech_params)
    # MODIFIED: Pass storage_capacity to redundancy calculation (storage now counts as backup)
    redundancy_index = calculate_redundancy(generation_capacity, max_total_demand, tech_params, storage_capacity)
    buffer_capacity_index = calculate_buffer_capacity(storage_capacity, max_total_demand)

    # NEW: Calculate renewable-fossil balance to break emissions-resilience correlation
    balance_index = calculate_renewable_fossil_balance(generation_capacity, tech_params)

    # Calculate total cost: Variable operational costs + CAPEX + fixed OPEX
    # FIXED 3rd attempt: Combine oemof's variable costs (fuel, grid) with fixed infrastructure costs
    # This breaks cost-resilience correlation: fuel usage varies independently of installed capacity

    # Get variable operational costs from oemof objective (fuel, grid electricity, etc.)
    try:
        import pyomo.environ
        variable_operational_cost = pyomo.environ.value(model.objective)
    except:
        try:
            variable_operational_cost = model.objective()
        except:
            variable_operational_cost = 0.0  # Fallback if objective not available

    # Add fixed annual infrastructure costs (CAPEX + OPEX per kW installed)
    fixed_infrastructure_cost = 0.0
    for tech in ["pv", "gas_boiler", "gas_chp", "heat_pump_air", "heat_pump_ground"]:
        capacity = tech_params.loc[tech, "capacity_kw"]
        capex = tech_params.loc[tech, "capex_per_kw_year"]
        opex = tech_params.loc[tech, "opex_per_kw_year"]
        fixed_infrastructure_cost += capacity * (capex + opex)

    # Storage CAPEX + OPEX (fixed annual costs per kWh capacity)
    for storage in ["battery_storage", "thermal_storage"]:
        storage_cap = tech_params.loc[storage, "storage_capacity_kwh"]
        capex = tech_params.loc[storage, "capex_per_kwh_year"]
        opex = tech_params.loc[storage, "opex_per_kwh_year"]
        fixed_infrastructure_cost += storage_cap * (capex + opex)

    # Total cost = variable operational + fixed infrastructure
    total_cost = variable_operational_cost + fixed_infrastructure_cost

    # Calculate emissions with time-varying grid emission factor
    total_emissions = calculate_emissions_time_varying(results, timeseries_data, tech_params)

    return {
        "electricity_generation": electricity_generation,
        "heat_generation": heat_generation,
        "total_generation": total_generation,
        "total_cost": total_cost,
        "total_emissions": total_emissions,
        "stirling_index": stirling_index,
        "redundancy_index": redundancy_index,
        "buffer_capacity_index": buffer_capacity_index,
        "balance_index": balance_index,  # NEW
    }


def _manual_result_extraction(model):
    """
    Manually extract results from Pyomo model when oemof processing fails

    Args:
        model: Pyomo model object

    Returns:
        dict: Results dictionary
    """
    results = {}

    try:
        for name, var in model.component_map(ctype=pyomo.environ.Var).items():
            if name == 'flow':
                for (node_i, node_o, t), val in var.items():
                    if (node_i, node_o) not in results:
                        results[(node_i, node_o)] = {
                            'sequences': pd.Series(index=model.timeindex, dtype=float)
                        }
                    if val.value is not None:
                        results[(node_i, node_o)]['sequences'].iloc[t] = val.value
    except Exception as e:
        print(f"Manual extraction failed: {e}")

    return results


def solve_energy_system(model, solver='glpk'):
    """
    Solve the energy system optimization model

    Args:
        model: oemof Model object
        solver (str): Solver name

    Returns:
        bool: True if solved successfully, False otherwise
    """
    try:
        solver_result = model.solve(solver=solver, tee=False)
        return True
    except Exception as e:
        print(f"Solver failed: {e}")
        return False
