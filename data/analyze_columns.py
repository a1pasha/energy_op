import pandas as pd

df = pd.read_excel('2020.xlsx', sheet_name='Timeseries')

print('='*80)
print('ALL TIMESERIES COLUMNS ANALYSIS')
print('='*80)
print()

for col in df.columns:
    print(f'{col}:')
    vals = df[col]
    print(f'  Type: {vals.dtype}')
    print(f'  Unique values: {vals.nunique()}')

    if vals.dtype in ['float64', 'int64']:
        print(f'  Min: {vals.min():.6f}')
        print(f'  Max: {vals.max():.6f}')
        print(f'  Mean: {vals.mean():.6f}')
        print(f'  Std: {vals.std():.6f}')

        # Sample values at different times
        samples = []
        for i in [0, 100, 500, 1000, 5000]:
            if i < len(vals):
                samples.append(f'{i}: {vals.iloc[i]:.4f}')
        print(f'  Samples: {", ".join(samples)}')
    else:
        print(f'  First value: {vals.iloc[0]}')

    print()
