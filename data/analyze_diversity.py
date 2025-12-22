import pandas as pd

df = pd.read_csv('nsga2_results/run_2025-12-21_19-13-21/all_evaluated_solutions.csv')

print('=== SOLUTION DIVERSITY ANALYSIS ===')
print(f'Total evaluations: {len(df)}')
print('\nSolutions per generation:')
print(df.groupby('Generation')['Solution_ID'].count())

print('\n=== EXPLORATION BY GENERATION ===')
for gen in range(10):
    gen_df = df[df['Generation'] == gen]
    cost_range = gen_df['Cost'].max() - gen_df['Cost'].min()
    emis_range = gen_df['Emissions_kg'].max() - gen_df['Emissions_kg'].min()
    res_range = gen_df['Resilience'].max() - gen_df['Resilience'].min()

    print(f'\nGen {gen}: {len(gen_df)} solutions')
    print(f'  Cost: [{gen_df["Cost"].min():.1f}, {gen_df["Cost"].max():.1f}] (spread: {cost_range:.1f})')
    print(f'  Emissions: [{gen_df["Emissions_kg"].min():.0f}, {gen_df["Emissions_kg"].max():.0f}] (spread: {emis_range:.0f})')
    print(f'  Resilience: [{gen_df["Resilience"].min():.4f}, {gen_df["Resilience"].max():.4f}] (spread: {res_range:.4f})')

print('\n=== BEST VALUES FOUND EACH GENERATION ===')
for gen in range(10):
    gen_df = df[df['Generation'] == gen]
    best_cost = gen_df['Cost'].min()
    best_emis = gen_df['Emissions_kg'].min()
    best_res = gen_df['Resilience'].max()
    print(f'Gen {gen}: Cost={best_cost:.2f}, Emissions={best_emis:.0f}, Resilience={best_res:.4f}')
