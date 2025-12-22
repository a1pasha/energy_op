#!/usr/bin/env python3
"""
Run optimization with more generations
Generations = 200 (instead of 150)
80 population, seed=42
"""

import subprocess
import sys

print("Starting optimization: 200 gen, 80 pop, seed=42\n")

# Run main.py with parameters
inputs = "200\n80\n2\n"
result = subprocess.run(
    ['python', 'main.py'],
    input=inputs,
    text=True
)

print("\n✓ Optimization complete!")
