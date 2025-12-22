# Cost Calculation Fix - 2025-12-22

## Problem
After removing H2 system (electrolyser, hydrogen_storage, hydrogen_bus), all optimization runs showed **Cost = 0.0 EUR** for every solution.

## Root Cause
- Old code relied on `model.objective` from oemof to get costs
- oemof's objective only includes costs when using `Investment` objects
- This system uses **fixed nominal_value** (pre-scaled capacities), not Investment
- Without Investment objects, oemof's objective = 0 (only operational variable costs, which don't accumulate as CAPEX)

## Fix Applied
**File**: `utils.py` (lines 152-168)

Changed from:
```python
# Get total cost from objective function
try:
    total_cost = pyomo.environ.value(model.objective)
except:
    total_cost = model.objective()
```

To:
```python
# Calculate total cost manually from capacities (CAPEX + OPEX)
total_cost = 0.0

# Technology costs (capacity-based)
for tech in ["pv", "gas_boiler", "gas_chp", "heat_pump_air", "heat_pump_ground"]:
    capacity = tech_params.loc[tech, "capacity_kw"]
    capex = tech_params.loc[tech, "capex_per_kw_year"]
    opex = tech_params.loc[tech, "opex_per_kw_year"]
    total_cost += capacity * (capex + opex)

# Storage costs (energy-based)
for storage in ["battery_storage", "thermal_storage"]:
    storage_cap = tech_params.loc[storage, "storage_capacity_kwh"]
    capex = tech_params.loc[storage, "capex_per_kwh_year"]
    opex = tech_params.loc[storage, "opex_per_kwh_year"]
    total_cost += storage_cap * (capex + opex)
```

## Results After Fix
- **Cost range**: $1.9M - $3.2M EUR/year ✓
- **Proper trade-offs**: Cost vs Emissions correlation = -0.29 ✓
- **5 Pareto solutions** found (was only 1 when costs = 0) ✓
- All objectives working correctly ✓

## Why This Happened
When removing electrolyser references from `utils.py`, the cost calculation remained unchanged but was already broken. The previous working runs (e.g., `run_2025-12-21_19-13-21`) had costs because they still included the H2 system with Investment-based sizing. After H2 removal, the underlying issue became visible.

## Verification
Test run (2 gen, 5 pop): Costs properly calculated at $1.9M-$3.2M/year
- Battery storage: 511k EUR/year (largest component)
- PV: 251k EUR/year
- Gas CHP: 232k EUR/year
- Heat pumps: 253k EUR/year combined
