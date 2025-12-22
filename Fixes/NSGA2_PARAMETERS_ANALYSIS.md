# NSGA-II Parameters Analysis

## Current Settings (Line 284-289)

```python
algorithm = NSGA2(
    pop_size=pop_size,
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=0.9, eta=15),      # Simulated Binary Crossover
    mutation=PM(eta=20),                   # Polynomial Mutation
    eliminate_duplicates=True
)
```

## Parameter Analysis

### 1. **Crossover: SBX(prob=0.9, eta=15)**
- **prob=0.9**: 90% crossover probability ✅ GOOD (standard)
- **eta=15**: Distribution index - controls how close offspring are to parents
  - **PROBLEM**: eta=15 is CONSERVATIVE (offspring stay close to parents)
  - **Lower eta** = more exploration, wider offspring spread
  - **Typical range**: 5-30
  - **Recommendation**: Try eta=10 or eta=5 for more diversity

### 2. **Mutation: PM(eta=20)**
- **NO prob parameter!** ❌ **MAJOR ISSUE**
  - PM mutation probability defaults to **1/n_var** (very low!)
  - With 9 variables: prob = 1/9 = 11% chance per solution
  - **PROBLEM**: Very weak mutation pressure
- **eta=20**: High value = small mutations (conservative)
  - **Typical range**: 10-30
  - **Recommendation**: Try eta=10 for larger mutations

### 3. **Missing: Mutation Probability**
- **Should explicitly set**: `PM(eta=20, prob=0.2)` or higher
- **Standard values**: 0.1-0.3 (10-30% per variable)
- **Current implicit**: ~0.11 (too low for this problem)

### 4. **eliminate_duplicates=True**
- ✅ GOOD - prevents wasted evaluations
- But may contribute to diversity loss if search space is small

## Why Premature Convergence Occurs

1. **Low mutation probability** (~11%) → not enough new genetic material
2. **Conservative crossover** (eta=15) → offspring too similar to parents
3. **Conservative mutation** (eta=20) → small perturbations only
4. **Small population** (20) → limited diversity pool
5. **Gurobi LP solver** → may find same optimal solutions repeatedly

## Recommended Fixes

### **Option 1: Increase Mutation Strength (Quick Fix)**
```python
crossover=SBX(prob=0.9, eta=10),    # Lower eta = more exploration
mutation=PM(eta=15, prob=0.3),      # Explicit prob + lower eta
```

### **Option 2: Aggressive Exploration (Best for your case)**
```python
crossover=SBX(prob=0.9, eta=5),     # Very exploratory crossover
mutation=PM(eta=10, prob=0.4),      # High mutation rate
```

### **Option 3: Larger Population (Combine with Option 2)**
```python
pop_size=100,  # Was 20
n_gen=50,      # Was 10
crossover=SBX(prob=0.9, eta=10),
mutation=PM(eta=15, prob=0.3),
```

## Expected Impact

### With Current Settings:
- Gen 0-3: Random search finds decent solutions
- Gen 4+: Population converges due to weak exploration
- Result: Stuck at local optimum

### With Recommended Settings (Option 2):
- More diverse offspring (eta=5 crossover)
- Frequent mutations (prob=0.3-0.4)
- Continuous exploration throughout generations
- Better Pareto front coverage

## References
- SBX eta typical: 5-20 (lower = more exploration)
- PM eta typical: 10-30 (lower = larger mutations)
- PM prob typical: 0.1-0.3 per variable
- Population size: 50-200 for multi-objective problems
