#!/usr/bin/env python3
"""
Run optimization with different random seed
Seed = 123 (instead of default 42)
150 generations, 80 population
"""

import subprocess
import sys

# Modify optimization.py to use seed 123
print("Setting random seed to 123...")

with open('optimization.py', 'r') as f:
    code = f.read()

# Replace seed setting
code = code.replace('np.random.seed(42)', 'np.random.seed(123)')
code = code.replace('random_seed=42', 'random_seed=123')

with open('optimization.py', 'w') as f:
    f.write(code)

print("✓ Seed set to 123")
print("Starting optimization: 150 gen, 80 pop, seed=123\n")

# Run main.py with parameters
inputs = "150\n80\n2\n"
result = subprocess.run(
    ['python', 'main.py'],
    input=inputs,
    text=True
)

print("\n✓ Optimization complete!")
