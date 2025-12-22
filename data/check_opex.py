import pandas as pd

tech = pd.read_csv('technology_parameters.csv', index_col=0)

print('=== UPDATED TECH PARAMETERS WITH OPEX ===\n')
print(tech[['capacity_kw', 'capex_per_kw_year', 'opex_per_kw_year',
             'storage_capacity_kwh', 'capex_per_kwh_year', 'opex_per_kwh_year']].to_string())

print('\n\n=== TOTAL ANNUAL COSTS (CAPEX + OPEX) ===')
for idx in tech.index:
    if pd.notna(tech.loc[idx, 'capacity_kw']):
        capex = tech.loc[idx, 'capex_per_kw_year']
        opex = tech.loc[idx, 'opex_per_kw_year']
        total = capex + opex
        pct = (opex / total * 100) if total > 0 else 0
        print(f'{idx:20s}: {total:.3f} EUR/kW/year (OPEX = {opex:.3f}, {pct:.1f}%)')
    elif pd.notna(tech.loc[idx, 'storage_capacity_kwh']):
        capex = tech.loc[idx, 'capex_per_kwh_year']
        opex = tech.loc[idx, 'opex_per_kwh_year']
        total = capex + opex
        pct = (opex / total * 100) if total > 0 else 0
        print(f'{idx:20s}: {total:.3f} EUR/kWh/year (OPEX = {opex:.3f}, {pct:.1f}%)')
