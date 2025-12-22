# Multi-Objective Optimization Fixes

**Date:** 2025-12-21
**Problem:** Previous optimization results showed poor tradeoffs between objectives (emissions-resilience correlation = 0.999)

---

## Issues Identified

### 1. **Tight Bounds Caused Variable Saturation**
- **Old bounds:** 0.1 - 2.0 (or 0.0 - 5.0 in broken version)
- **Problem:** 6 out of 9 decision variables stuck at upper bound (2.0)
- **Result:** Only gas_chp, hydrogen_storage, and battery_storage had real variation

### 2. **Emissions-Resilience Correlation (r=0.999)**
- **Problem:** Resilience metric rewarded total capacity indiscriminately
- **Result:** More gas CHP → Higher emissions AND higher resilience (no tradeoff)
- **Impact:** Effectively a 2-objective problem instead of 3-objective

### 3. **Zero-Capacity Exploits (New Runs)**
- **Problem:** Allowing CAPACITY_SCALE_MIN = 0.0 enabled unrealistic solutions
- **Result:** Cost = €0, Emissions = 0 kg (physically impossible)

---

## Fixes Implemented

### Fix #1: Widened Capacity Bounds
**File:** `config.py` (lines 84-86)

```python
# BEFORE (BROKEN):
CAPACITY_SCALE_MIN = 0.0  # Allowed zero capacity
CAPACITY_SCALE_MAX = 5.0  # Too wide, caused instability

# AFTER (FIXED):
CAPACITY_SCALE_MIN = 0.2  # Minimum 20% of nominal (prevents unrealistic solutions)
CAPACITY_SCALE_MAX = 3.0  # Maximum 300% (wider than old 2.0, enables exploration)
```

**Rationale:**
- Prevents zero-capacity exploits (min = 0.2 instead of 0.0)
- Enables more variation than old bounds (max = 3.0 instead of 2.0)
- Reduces variable saturation at boundaries

---

### Fix #2: Added Renewable-Fossil Balance Metric
**Files:**
- `resilience_metrics.py` (new function: `calculate_renewable_fossil_balance()`)
- `utils.py` (integrated balance calculation)
- `optimization.py` (included in resilience score)

**New Function:**
```python
def calculate_renewable_fossil_balance(generation_capacity, tech_params):
    """
    Rewards systems with BOTH renewable AND fossil capacity.

    Balance = 4 * (P_renewable / P_total) * (P_fossil / P_total)

    - Maximum (1.0) when 50-50 renewable-fossil mix
    - Low when dominated by one type
    - Breaks correlation: high balance possible with LOW emissions (renewable-heavy)
                          or HIGH emissions (fossil-heavy)
    """
```

**Resilience Weights Updated:**
```python
# BEFORE:
RESILIENCE_WEIGHTS = {
    'stirling': 0.33,
    'redundancy': 0.33,
    'buffer': 0.34
}

# AFTER:
RESILIENCE_WEIGHTS = {
    'stirling': 0.25,      # Technology diversity
    'redundancy': 0.25,    # Capacity margin
    'buffer': 0.25,        # Storage capacity
    'balance': 0.25        # NEW: Renewable-fossil balance
}
```

**Expected Impact:**
- High resilience achievable with LOW emissions (diverse renewable portfolio with some gas backup)
- High resilience also achievable with HIGH emissions (fossil-heavy with renewables)
- **Breaks the 0.999 correlation** between emissions and resilience

---

### Fix #3: Boundary Saturation Penalty
**File:** `optimization.py` (lines 104-114)

```python
# NEW: Apply penalty for boundary saturation (prevents clustering at bounds)
n_vars = len(x)
n_at_lower = sum([1 for xi in x if xi <= CAPACITY_SCALE_MIN * 1.05])
n_at_upper = sum([1 for xi in x if xi >= CAPACITY_SCALE_MAX * 0.95])
saturation_fraction = (n_at_lower + n_at_upper) / n_vars

# If >40% of variables are saturated, apply penalty to resilience
if saturation_fraction > 0.4:
    saturation_penalty = 0.1 * (saturation_fraction - 0.4)  # Up to 6% penalty
    resilience_score = resilience_score * (1 - saturation_penalty)
```

**Rationale:**
- Discourages solutions where >40% of variables hit bounds
- Forces optimizer to explore interior of search space
- Encourages diverse capacity distributions

---

## Expected Improvements

### Objective Independence
| Objective Pair | Old Correlation | Expected New Correlation |
|---------------|-----------------|--------------------------|
| Cost vs Emissions | 0.003 (good) | 0.01-0.10 (good) |
| Cost vs Resilience | 0.016 (good) | 0.05-0.15 (good) |
| **Emissions vs Resilience** | **0.999 (broken)** | **0.20-0.50 (acceptable)** |

### Variable Variation
- Old: 6/9 variables stuck at 2.0 (std < 0.01)
- Expected: 0-2/9 variables at boundaries (wider distribution)

### Solution Diversity
- Old: Gas CHP only varying technology
- Expected: Multiple technologies vary independently

### Realistic Costs
- Old (0.1-2.0 bounds): €20-408 ✓ Valid
- Broken (0.0-5.0 bounds): €0-1,617 ✗ Invalid
- Expected (0.2-3.0 bounds): €15-500 ✓ Valid

---

## Testing Plan

### Quick Validation Run
**Parameters:**
- Workers: 4 cores
- Generations: 10
- Population: 20
- Total evaluations: 200

**Success Criteria:**
1. ✅ No solutions with Cost < €5 (validates minimum capacity works)
2. ✅ Emissions-Resilience correlation < 0.7 (validates balance metric works)
3. ✅ At least 4/9 decision variables have std > 0.1 (validates saturation penalty works)
4. ✅ Pareto front shows clear tradeoffs in 3D objective space

---

## Files Modified

1. `config.py` - Updated bounds and resilience weights
2. `resilience_metrics.py` - Added renewable-fossil balance function
3. `utils.py` - Integrated balance index calculation
4. `optimization.py` - Added balance to resilience score + saturation penalty

---

## Backward Compatibility

⚠️ **Results from new runs are NOT comparable to old runs** due to:
- Different bounds (0.2-3.0 vs 0.1-2.0 or 0.0-5.0)
- Different resilience formulation (4 components vs 3)
- Different selection pressure (saturation penalty)

**Old results should be marked as "preliminary" or "baseline" in thesis.**

---

## Next Steps

1. Run quick validation (4 cores, 10 gen, 20 pop)
2. Analyze correlation matrix and decision variable distributions
3. If validated, run full optimization (64 cores, 150 gen, 80 pop)
4. Compare Pareto fronts visually to confirm improvements
