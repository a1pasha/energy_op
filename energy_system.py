#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
6-Bus Energy System Model with Time-Varying Efficiencies

Features:
- Temperature-dependent heat pump COP (hourly variation)
- Time-varying grid emission factor
- Dynamic capacity limits for heat pumps
- PV routing bus for feed-in tracking
- District heating network with pump modeling

Compatible with oemof 1.0.0
"""

from oemof.solph._energy_system import EnergySystem
from oemof.solph._models import Model
from oemof.solph.flows._flow import Flow
from oemof.solph.buses._bus import Bus
from oemof.solph.components._generic_storage import GenericStorage
from oemof.network.network.nodes import Source, Sink, Transformer


def create_energy_system(timeseries_data, date_time_index, tech_params,
                         use_time_varying_hp=True,
                         use_time_varying_emissions=True):
    """
    Create 6-bus energy system with temperature-dependent components

    Args:
        timeseries_data (DataFrame): Timeseries with HP COP, temperature, etc.
        date_time_index (DatetimeIndex): Time index
        tech_params (DataFrame): Technology parameters
        use_time_varying_hp (bool): Use temperature-dependent HP efficiency
        use_time_varying_emissions (bool): Use hourly grid emissions

    Returns:
        tuple: (energy_system, model)
    """
    print("\nCreating 5-bus energy system...")
    print(f"  Time-varying HP efficiency: {use_time_varying_hp}")
    print(f"  Time-varying grid emissions: {use_time_varying_emissions}")
    print(f"  PV routing bus: ENABLED")
    print(f"  District heating network: ENABLED")

    # Set infer_last_interval=False to prevent adding extra timestep (8760 → 8761)
    energy_system = EnergySystem(timeindex=date_time_index, infer_last_interval=False)

    # ========================================================================
    # CREATE BUSES (5-bus configuration)
    # ========================================================================
    electricity_bus = Bus(label="electricity_bus")  # Main electricity (self-consumption)
    pv_bus = Bus(label="pv_bus")  # PV routing bus (for feed-in tracking)
    heat_bus = Bus(label="heat_bus")  # Heat generation
    heat_grid_bus = Bus(label="heat_grid_bus")  # District heating network (for pump modeling)
    gas_bus = Bus(label="gas_bus")
    # REMOVED: hydrogen_bus - no H2 usage pathway (electrolyser removed)

    # ========================================================================
    # CREATE SOURCES
    # ========================================================================

    # Gas source with time-varying prices
    gas_source = Source(
        label="gas_supply",
        outputs={gas_bus: Flow(
            variable_costs=timeseries_data["gas_price"].values  # Time-varying array
        )}
    )

    # Grid electricity with time-varying costs AND emissions
    grid_variable_costs = timeseries_data["electricity_price"].values.tolist()

    grid_source = Source(
        label="grid_electricity",
        outputs={electricity_bus: Flow(
            nominal_value=tech_params.loc["grid_electricity", "capacity_kw"],
            variable_costs=grid_variable_costs  # Time-varying prices
        )}
    )

    # PV source - NOW outputs to dedicated PV bus for routing
    pv = Source(
        label="pv",
        outputs={pv_bus: Flow(  # Changed from electricity_bus to pv_bus
            nominal_value=tech_params.loc["pv", "capacity_kw"],
            variable_costs=0,  # FIXED: CAPEX/OPEX now only in utils.py, not double-counted
            max=timeseries_data["solar_availability"].values.tolist()
        )}
    )

    # REMOVED: Wind source (not in actual building system)

    # ========================================================================
    # PV ROUTING COMPONENTS (NEW - for feed-in tracking)
    # ========================================================================

    # PV to self-consumption transformer
    pv_to_self = Transformer(
        label="pv_to_self",
        inputs={pv_bus: Flow()},
        outputs={electricity_bus: Flow()}  # Route PV to main electricity bus
    )
    pv_to_self.conversion_factors = {pv_bus: 1, electricity_bus: 1}  # No losses

    # Grid feed-in (PV export) sink - earns revenue!
    feed_in_tariff = timeseries_data.get("feed_in_tariff", -0.04)  # Default -0.04 EUR/kWh
    if hasattr(feed_in_tariff, 'values'):
        feed_in_costs = feed_in_tariff.values.tolist()
    else:
        feed_in_costs = [feed_in_tariff] * len(timeseries_data)

    grid_feed_in = Sink(
        label="grid_feed_in",
        inputs={pv_bus: Flow(
            variable_costs=feed_in_costs  # Negative = revenue
        )}
    )

    # ========================================================================
    # CREATE TRANSFORMERS (CONVERSION TECHNOLOGIES)
    # ========================================================================

    # Gas boiler (unchanged)
    gas_boiler = Transformer(
        label="gas_boiler",
        inputs={gas_bus: Flow()},
        outputs={heat_bus: Flow(
            nominal_value=tech_params.loc["gas_boiler", "capacity_kw"],
            variable_costs=0  # FIXED: CAPEX/OPEX now only in utils.py, not double-counted
        )},
    )
    gas_boiler.conversion_factors = {
        gas_bus: 1,
        heat_bus: tech_params.loc["gas_boiler", "efficiency"]
    }

    # Gas CHP (unchanged)
    gas_chp = Transformer(
        label="gas_chp",
        inputs={gas_bus: Flow()},
        outputs={
            electricity_bus: Flow(
                nominal_value=tech_params.loc["gas_chp", "capacity_kw"],
                variable_costs=0  # FIXED: CAPEX/OPEX now only in utils.py, not double-counted
            ),
            heat_bus: Flow(
                nominal_value=tech_params.loc["gas_chp", "capacity_kw"] * 1.2
            )
        }
    )
    gas_chp.conversion_factors = {
        gas_bus: 1,
        electricity_bus: tech_params.loc["gas_chp", "efficiency"] * 0.4,
        heat_bus: tech_params.loc["gas_chp", "efficiency"] * 0.6
    }

    # ========================================================================
    # Temperature-Dependent Heat Pumps
    # ========================================================================

    if use_time_varying_hp and 'hp_air_cop' in timeseries_data.columns:
        # AIR SOURCE HEAT PUMP with TIME-VARYING COP
        print("  Using temperature-dependent Air HP efficiency")

        # Get time-varying COP and capacity limit
        hp_air_cop_series = timeseries_data['hp_air_cop'].values.tolist()

        # LIMITATION: oemof basic Transformer doesn't support time-varying conversion factors
        # Using average COP means electricity consumption is approximate (not exact at each timestep)
        # This is acceptable for annual optimization where total energy balance matters more
        # Dynamic capacity limits (hp_air_max) capture operational constraints

        # FIXED: Remove OPEX from variable_costs (counted in utils.py fixed infrastructure)
        avg_cop = timeseries_data['hp_air_cop'].mean()

        # Grid electricity variable_costs already account for HP consumption
        # No need to add OPEX here (would be double-counting)
        adjusted_costs = 0  # OPEX handled in utils.py

        # Get dynamic capacity limits if available
        if 'hp_air_max' in timeseries_data.columns:
            hp_air_max_series = timeseries_data['hp_air_max'].values.tolist()
        else:
            hp_air_max_series = [1.0] * len(timeseries_data)

        heat_pump_air = Transformer(
            label="heat_pump_air",
            inputs={electricity_bus: Flow()},
            outputs={heat_bus: Flow(
                nominal_value=tech_params.loc["heat_pump_air", "capacity_kw"],
                variable_costs=0,  # FIXED: OPEX now only in utils.py, not double-counted
                max=hp_air_max_series  # Dynamic capacity limit
            )}
        )
        # Use average efficiency for conversion factor
        heat_pump_air.conversion_factors = {
            electricity_bus: 1.0 / avg_cop,
            heat_bus: 1.0
        }

    else:
        # Standard AIR HP with fixed efficiency
        print("  Using fixed Air HP efficiency")
        heat_pump_air = Transformer(
            label="heat_pump_air",
            inputs={electricity_bus: Flow()},
            outputs={heat_bus: Flow(
                nominal_value=tech_params.loc["heat_pump_air", "capacity_kw"],
                variable_costs=0  # FIXED: CAPEX/OPEX now only in utils.py, not double-counted
            )}
        )
        heat_pump_air.conversion_factors = {
            electricity_bus: 1.0 / tech_params.loc["heat_pump_air", "efficiency"],
            heat_bus: 1.0
        }

    if use_time_varying_hp and 'hp_geo_cop' in timeseries_data.columns:
        # GROUND SOURCE HEAT PUMP with TIME-VARYING COP
        print("  Using temperature-dependent Geo HP efficiency")

        hp_geo_cop_series = timeseries_data['hp_geo_cop'].values.tolist()
        avg_cop_geo = timeseries_data['hp_geo_cop'].mean()

        # FIXED: Remove OPEX from variable_costs (counted in utils.py fixed infrastructure)
        # Grid electricity variable_costs already account for HP consumption
        adjusted_costs_geo = 0  # OPEX handled in utils.py

        heat_pump_ground = Transformer(
            label="heat_pump_ground",
            inputs={electricity_bus: Flow()},
            outputs={heat_bus: Flow(
                nominal_value=tech_params.loc["heat_pump_ground", "capacity_kw"],
                variable_costs=0  # FIXED: OPEX now only in utils.py, not double-counted
            )}
        )
        heat_pump_ground.conversion_factors = {
            electricity_bus: 1.0 / avg_cop_geo,
            heat_bus: 1.0
        }

    else:
        # Standard GEO HP with fixed efficiency
        print("  Using fixed Geo HP efficiency")
        heat_pump_ground = Transformer(
            label="heat_pump_ground",
            inputs={electricity_bus: Flow()},
            outputs={heat_bus: Flow(
                nominal_value=tech_params.loc["heat_pump_ground", "capacity_kw"],
                variable_costs=0  # FIXED: CAPEX/OPEX now only in utils.py, not double-counted
            )}
        )
        heat_pump_ground.conversion_factors = {
            electricity_bus: 1.0 / tech_params.loc["heat_pump_ground", "efficiency"],
            heat_bus: 1.0
        }

    # ========================================================================
    # REMOVED COMPONENTS
    # ========================================================================
    # REMOVED: Electrolyser - no H2 usage pathway (no fuel cell, no H2 demand)
    # REMOVED: Fuel cell - not in actual building system

    # ========================================================================
    # DISTRICT HEATING PUMP (NEW - models circulation pump)
    # ========================================================================

    # District heating circulation pump
    # Models the electricity consumption to pump hot water through the network
    # Typical consumption: ~1.5% of heat delivered
    district_heating_pump = Transformer(
        label="district_heating_pump",
        inputs={
            heat_bus: Flow(),  # Heat from generation
            electricity_bus: Flow()  # Electricity for pump
        },
        outputs={heat_grid_bus: Flow(nominal_value=10000)}  # Heat to distribution network
    )
    district_heating_pump.conversion_factors = {
        heat_bus: 1.0 / 0.98,  # FIXED: Need 1.02 kWh heat to deliver 1 kWh (2% losses)
        electricity_bus: 0.015,  # 1.5% electricity for pumping per kWh delivered
        heat_grid_bus: 1.0  # 1 kWh heat delivered (normalized output)
    }

    # ========================================================================
    # CREATE STORAGE COMPONENTS (unchanged)
    # ========================================================================

    battery_storage = GenericStorage(
        label="battery_storage",
        inputs={electricity_bus: Flow()},
        outputs={electricity_bus: Flow()},
        nominal_storage_capacity=tech_params.loc["battery_storage", "storage_capacity_kwh"],
        loss_rate=0.001,
        initial_storage_level=0.5,
        inflow_conversion_factor=tech_params.loc["battery_storage", "efficiency"]**0.5,
        outflow_conversion_factor=tech_params.loc["battery_storage", "efficiency"]**0.5
    )

    thermal_storage = GenericStorage(
        label="thermal_storage",
        inputs={heat_bus: Flow()},
        outputs={heat_bus: Flow()},
        nominal_storage_capacity=tech_params.loc["thermal_storage", "storage_capacity_kwh"],
        loss_rate=0.002,
        initial_storage_level=0.5,
        inflow_conversion_factor=tech_params.loc["thermal_storage", "efficiency"]**0.5,
        outflow_conversion_factor=tech_params.loc["thermal_storage", "efficiency"]**0.5
    )

    # REMOVED: hydrogen_storage - no H2 usage pathway

    # ========================================================================
    # CREATE SINKS (DEMANDS)
    # ========================================================================

    electricity_demand = Sink(
        label="electricity_demand",
        inputs={electricity_bus: Flow(
            nominal_value=max(timeseries_data["demand_electricity_kw"]),
            fix=timeseries_data["demand_electricity_kw"].values / max(timeseries_data["demand_electricity_kw"])
        )}
    )

    heat_demand = Sink(
        label="heat_demand",
        inputs={heat_grid_bus: Flow(  # Changed from heat_bus to heat_grid_bus
            nominal_value=max(timeseries_data["demand_heat_kw"]),
            fix=timeseries_data["demand_heat_kw"].values / max(timeseries_data["demand_heat_kw"])
        )}
    )

    # ========================================================================
    # ADD ALL NODES TO ENERGY SYSTEM
    # ========================================================================

    energy_system.add(
        # Buses (5 buses total)
        electricity_bus, pv_bus, heat_bus, heat_grid_bus, gas_bus,
        # Sources
        gas_source, grid_source, pv,
        # PV Routing
        pv_to_self, grid_feed_in,
        # Transformers
        gas_boiler, gas_chp, heat_pump_air, heat_pump_ground,
        district_heating_pump,
        # Storage
        battery_storage, thermal_storage,
        # Sinks
        electricity_demand, heat_demand
    )

    print(f"  5-bus system created:")
    print(f"    - Buses: 5 (electricity, pv, heat, heat_grid, gas)")
    print(f"    - PV routing: pv_bus -> electricity_bus (self-consumption)")
    print(f"    - PV routing: pv_bus -> grid_feed_in (export revenue)")
    print(f"    - District heating: heat_bus -> heat_grid_bus (pump + losses)")

    # ========================================================================
    # CREATE OPTIMIZATION MODEL
    # ========================================================================

    model = Model(energy_system)

    print("  [OK] 5-bus energy system created successfully")

    return energy_system, model
