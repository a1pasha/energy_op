#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data loading and generation module
Handles both dummy data generation and real data loading
"""

import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
from config import (
    START_DATE, END_DATE, BASE_DEMAND_ELECTRICITY_KW,
    BASE_DEMAND_HEAT_KW, RANDOM_SEED, TIMESERIES_FILE, TECHNOLOGY_FILE
)


def create_dummy_timeseries():
    """
    Create realistic synthetic timeseries data with:
    - Weekday/weekend patterns
    - Seasonal variations
    - Peak/off-peak pricing
    - Realistic solar/wind profiles

    Returns:
        tuple: (timeseries_data DataFrame, date_time_index)
    """
    np.random.seed(RANDOM_SEED)

    # Define time period
    date_time_index = pd.date_range(START_DATE, END_DATE, freq="1H")
    dummy_data = pd.DataFrame(index=date_time_index)

    # Extract time features
    time_of_day = dummy_data.index.hour
    day_of_week = dummy_data.index.dayofweek  # 0=Monday, 6=Sunday
    day_of_year = dummy_data.index.dayofyear
    is_weekend = day_of_week >= 5  # Saturday and Sunday

    # ========================================================================
    # ELECTRICITY DEMAND - Realistic pattern
    # ========================================================================
    # Base load varies by time of day
    morning_peak = np.exp(-((time_of_day - 8)**2) / 8)  # Peak at 8am
    evening_peak = np.exp(-((time_of_day - 19)**2) / 8)  # Peak at 7pm
    night_valley = 1 - np.exp(-((time_of_day - 3)**2) / 8)  # Low at 3am

    daily_profile = 0.5 + 0.25 * morning_peak + 0.35 * evening_peak + 0.1 * night_valley

    # Weekend reduction (20% lower)
    weekend_factor = np.where(is_weekend, 0.8, 1.0)

    # Seasonal variation (winter higher for heating-related electricity)
    seasonal_factor = 0.9 + 0.2 * np.cos(2 * np.pi * (day_of_year - 15) / 365)

    # Random daily variation (±10%)
    daily_noise = 1.0 + 0.1 * np.random.randn(len(date_time_index))

    dummy_data["demand_electricity_kw"] = (
        BASE_DEMAND_ELECTRICITY_KW * daily_profile *
        weekend_factor * seasonal_factor * daily_noise
    )

    # ========================================================================
    # HEAT DEMAND - Temperature dependent
    # ========================================================================
    # Higher in morning (6-9am) and evening (17-23pm)
    heat_morning = np.exp(-((time_of_day - 7)**2) / 6)
    heat_evening = np.exp(-((time_of_day - 20)**2) / 10)
    heat_profile = 0.3 + 0.35 * heat_morning + 0.4 * heat_evening

    # Strong seasonal variation (much higher in winter)
    heat_seasonal = 1.3 - 0.8 * np.cos(2 * np.pi * (day_of_year - 15) / 365)

    # Weekend slightly higher (people at home)
    heat_weekend_factor = np.where(is_weekend, 1.15, 1.0)

    # Random variation
    heat_noise = 1.0 + 0.15 * np.random.randn(len(date_time_index))

    heat_demand_raw = (
        BASE_DEMAND_HEAT_KW * heat_profile *
        heat_seasonal * heat_weekend_factor * heat_noise
    )
    dummy_data["demand_heat_kw"] = np.clip(heat_demand_raw, 0, None)

    # ========================================================================
    # SOLAR AVAILABILITY - Realistic solar irradiance
    # ========================================================================
    # January: shorter days, lower angle
    sunrise_hour = 7.5 - 1.5 * np.cos(2 * np.pi * (day_of_year - 172) / 365)
    sunset_hour = 17.5 + 1.5 * np.cos(2 * np.pi * (day_of_year - 172) / 365)
    day_length = sunset_hour - sunrise_hour

    # Solar intensity (0 at night, peak at solar noon)
    solar_noon = (sunrise_hour + sunset_hour) / 2
    solar_intensity = np.zeros(len(time_of_day))

    for i in range(len(time_of_day)):
        if sunrise_hour[i] <= time_of_day[i] <= sunset_hour[i]:
            # Cosine curve for solar intensity
            time_from_noon = (time_of_day[i] - solar_noon[i]) / (day_length[i] / 2)
            solar_intensity[i] = np.cos(time_from_noon * np.pi / 2) ** 1.5

    # Seasonal variation (lower in winter)
    seasonal_solar = 0.6 + 0.4 * np.cos(2 * np.pi * (day_of_year - 172) / 365)

    # Cloud cover (random, some days cloudy)
    cloud_factor = 0.7 + 0.3 * np.random.beta(5, 2, len(date_time_index))

    solar_avail_raw = solar_intensity * seasonal_solar * cloud_factor
    dummy_data["solar_availability"] = np.clip(solar_avail_raw, 0, 1)

    # ========================================================================
    # WIND AVAILABILITY - Weather patterns
    # ========================================================================
    # Base wind pattern (often stronger at night and in winter)
    wind_diurnal = 0.55 + 0.15 * np.sin(2 * np.pi * (time_of_day - 3) / 24)
    wind_seasonal = 0.85 + 0.3 * np.cos(2 * np.pi * (day_of_year - 15) / 365)

    # Weather fronts (multi-day patterns)
    num_hours = len(date_time_index)
    weather_pattern = np.zeros(num_hours)
    for i in range(0, num_hours, 6):  # 6-hour blocks
        weather_pattern[i:i+6] = np.random.beta(2, 2)

    # Smooth transitions
    weather_pattern = gaussian_filter1d(weather_pattern, sigma=3)

    # Hourly turbulence
    turbulence = 1.0 + 0.2 * np.random.randn(num_hours)

    wind_avail_raw = wind_diurnal * wind_seasonal * weather_pattern * turbulence
    dummy_data["wind_availability"] = np.clip(wind_avail_raw, 0, 1)

    # ========================================================================
    # GRID AVAILABILITY - Very high but occasional outages
    # ========================================================================
    grid_avail = np.ones(len(date_time_index)) * 0.98
    # Rare outages (0.1% chance per hour)
    outages = np.random.random(len(date_time_index)) < 0.001
    grid_avail[outages] = 0.5  # Reduced capacity during issues
    dummy_data["grid_availability"] = grid_avail

    # ========================================================================
    # ELECTRICITY PRICES - Time-of-use pricing
    # ========================================================================
    # Peak hours: 17-21 (evening)
    # Mid-peak: 7-17, 21-23
    # Off-peak: 23-7
    base_price = 0.12  # $/kWh

    price_multiplier = np.ones(len(time_of_day))
    # Off-peak (11pm - 7am): 0.7x
    price_multiplier[(time_of_day >= 23) | (time_of_day < 7)] = 0.7
    # Peak (5pm - 9pm): 1.5x
    price_multiplier[(time_of_day >= 17) & (time_of_day < 21)] = 1.5
    # Mid-peak: 1.0x (default)

    # Weekend discount (10% cheaper)
    weekend_price_factor = np.where(is_weekend, 0.9, 1.0)

    # Seasonal variation (higher in summer for AC)
    seasonal_price = 1.0 + 0.15 * np.sin(2 * np.pi * (day_of_year - 80) / 365)

    # Daily market variation (±5%)
    daily_price_noise = 1.0 + 0.05 * np.random.randn(len(date_time_index))

    elec_price_raw = (
        base_price * price_multiplier * weekend_price_factor *
        seasonal_price * daily_price_noise
    )
    dummy_data["electricity_price"] = np.clip(elec_price_raw, 0.05, 0.50)

    # ========================================================================
    # GAS PRICES - More stable but seasonal
    # ========================================================================
    base_gas_price = 0.06  # $/kWh
    # Higher in winter (heating season)
    gas_seasonal = 1.0 + 0.25 * np.cos(2 * np.pi * (day_of_year - 15) / 365)
    # Small random variation
    gas_noise = 1.0 + 0.02 * np.random.randn(len(date_time_index))

    gas_price_raw = base_gas_price * gas_seasonal * gas_noise
    dummy_data["gas_price"] = np.clip(gas_price_raw, 0.04, 0.10)

    return dummy_data, date_time_index


def create_dummy_tech_params():
    """
    Create dummy technology parameters for testing

    Returns:
        DataFrame: Technology parameters indexed by technology name
    """
    tech_data = {
        "pv": {
            "capacity_kw": 200,
            "opex_per_kw_year": 15,
            "efficiency": 1.0,
            "disparity_factor": 1.0,
            "dispatchable": 0,
            "co2_emissions_kg_per_kwh": 0.0
        },
        "wind": {
            "capacity_kw": 150,
            "opex_per_kw_year": 25,
            "efficiency": 1.0,
            "disparity_factor": 0.8,
            "dispatchable": 0,
            "co2_emissions_kg_per_kwh": 0.0
        },
        "gas_boiler": {
            "capacity_kw": 120,
            "opex_per_kw_year": 10,
            "efficiency": 0.9,
            "disparity_factor": 0.3,
            "dispatchable": 1,
            "co2_emissions_kg_per_kwh": 0.2
        },
        "gas_chp": {
            "capacity_kw": 100,
            "opex_per_kw_year": 30,
            "efficiency": 0.85,
            "disparity_factor": 0.4,
            "dispatchable": 1,
            "co2_emissions_kg_per_kwh": 0.18
        },
        "heat_pump_air": {
            "capacity_kw": 80,
            "opex_per_kw_year": 20,
            "efficiency": 3.0,
            "disparity_factor": 0.6,
            "dispatchable": 1,
            "co2_emissions_kg_per_kwh": 0.0
        },
        "heat_pump_ground": {
            "capacity_kw": 60,
            "opex_per_kw_year": 25,
            "efficiency": 4.0,
            "disparity_factor": 0.7,
            "dispatchable": 1,
            "co2_emissions_kg_per_kwh": 0.0
        },
        "electrolyser": {
            "capacity_kw": 50,
            "opex_per_kw_year": 40,
            "efficiency": 0.7,
            "disparity_factor": 0.8,
            "dispatchable": 1,
            "co2_emissions_kg_per_kwh": 0.0
        },
        "fuel_cell": {
            "capacity_kw": 40,
            "opex_per_kw_year": 50,
            "efficiency": 0.6,
            "disparity_factor": 0.9,
            "dispatchable": 1,
            "co2_emissions_kg_per_kwh": 0.0
        },
        "grid_electricity": {
            "capacity_kw": 250,
            "opex_per_kw_year": 5,
            "efficiency": 1.0,
            "disparity_factor": 0.2,
            "dispatchable": 1,
            "co2_emissions_kg_per_kwh": 0.35
        },
        "battery_storage": {
            "storage_capacity_kwh": 300,
            "efficiency": 0.95,
            "disparity_factor": 0.7,
            "dispatchable": 0,
            "co2_emissions_kg_per_kwh": 0.0  # Storage doesn't emit
        },
        "thermal_storage": {
            "storage_capacity_kwh": 400,
            "efficiency": 0.9,
            "disparity_factor": 0.5,
            "dispatchable": 0,
            "co2_emissions_kg_per_kwh": 0.0  # Storage doesn't emit
        },
        "hydrogen_storage": {
            "storage_capacity_kwh": 200,
            "efficiency": 0.8,
            "disparity_factor": 0.8,
            "dispatchable": 0,
            "co2_emissions_kg_per_kwh": 0.0  # Storage doesn't emit
        }
    }

    return pd.DataFrame.from_dict(tech_data, orient="index")


def load_real_timeseries():
    """
    Load real timeseries data from CSV file

    Returns:
        tuple: (timeseries_data DataFrame, date_time_index)
    """
    try:
        data = pd.read_csv(TIMESERIES_FILE, index_col=0, parse_dates=True)
        return data, data.index
    except FileNotFoundError:
        print(f"Real data file not found: {TIMESERIES_FILE}")
        print("Falling back to dummy data...")
        return create_dummy_timeseries()


def load_real_tech_params():
    """
    Load real technology parameters from CSV file

    Returns:
        DataFrame: Technology parameters
    """
    try:
        data = pd.read_csv(TECHNOLOGY_FILE, index_col=0)
        return data
    except FileNotFoundError:
        print(f"Technology parameters file not found: {TECHNOLOGY_FILE}")
        print("Falling back to dummy parameters...")
        return create_dummy_tech_params()


def load_data(use_dummy=True):
    """
    Load data - either dummy or real based on parameter

    Args:
        use_dummy (bool): If True, use dummy data; if False, try to load real data

    Returns:
        tuple: (timeseries_data, date_time_index, tech_params)
    """
    if use_dummy:
        print("Using dummy data for testing...")
        timeseries, date_index = create_dummy_timeseries()
        tech_params = create_dummy_tech_params()
    else:
        print("Attempting to load real data...")
        timeseries, date_index = load_real_timeseries()
        tech_params = load_real_tech_params()

    print(f"Loaded timeseries: {len(date_index)} timesteps")
    print(f"Loaded {len(tech_params)} technologies/storages")

    return timeseries, date_index, tech_params
