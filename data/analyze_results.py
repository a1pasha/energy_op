import pandas as pd
import numpy as np

df = pd.read_csv('nsga2_results/run_2025-12-21_19-13-21/pareto_solutions.csv')

print('=== PARETO FRONT SUMMARY ===')
print(f'Solutions: {len(df)}')
print(f'\nCost range: EUR {df["Cost"].min():.2f} - EUR {df["Cost"].max():.2f}')
print(f'Emissions range: {df["Emissions_kg"].min():.0f} - {df["Emissions_kg"].max():.0f} kg')
print(f'Resilience range: {df["Resilience"].min():.4f} - {df["Resilience"].max():.4f}')

corr = np.corrcoef(df['Emissions_kg'], df['Resilience'])[0,1]
print(f'\nEmissions-Resilience correlation: {corr:.4f}')

print(f'\n=== DECISION VARIABLES DIVERSITY ===')
vars_cols = ['pv', 'gas_boiler', 'gas_chp', 'heat_pump_air', 'heat_pump_ground',
             'battery_storage', 'thermal_storage']

for col in vars_cols:
    std = df[col].std()
    mean = df[col].mean()
    min_val = df[col].min()
    max_val = df[col].max()
    print(f'{col:20s}: mean={mean:.2f}, std={std:.3f}, range=[{min_val:.2f}, {max_val:.2f}]')
