#!/usr/bin/env python3
"""
Run optimization with different population size
Population = 100 (instead of 80)
150 generations, seed=42
"""

import subprocess
import sys

print("Starting optimization: 150 gen, 100 pop, seed=42\n")

# Run main.py with parameters
inputs = "150\n100\n2\n"
result = subprocess.run(
    ['python', 'main.py'],
    input=inputs,
    text=True
)

print("\n✓ Optimization complete!")
