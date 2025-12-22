import pandas as pd

xl = pd.ExcelFile('2020.xlsx')

print('='*80)
print('ALL SHEETS IN 2020.xlsx')
print('='*80)
print()

for sheet in xl.sheet_names:
    print(f'Sheet: {sheet}')
    df = pd.read_excel('2020.xlsx', sheet_name=sheet)
    print(f'  Rows: {len(df)}, Columns: {len(df.columns)}')
    print(f'  Column names: {list(df.columns[:10])}...' if len(df.columns) > 10 else f'  Columns: {list(df.columns)}')
    print()

print()
print('='*80)
print('DETAILED ANALYSIS OF KEY SHEETS')
print('='*80)
print()

# Analyze Source sheet
print('=== SOURCE SHEET ===')
df_source = pd.read_excel('2020.xlsx', sheet_name='Source')
print(df_source.to_string())
print()

# Analyze Source_fix sheet
print('=== SOURCE_FIX SHEET ===')
df_source_fix = pd.read_excel('2020.xlsx', sheet_name='Source_fix')
print(df_source_fix.to_string())
print()

# Analyze Transformer sheet
print('=== TRANSFORMER SHEET ===')
df_trans = pd.read_excel('2020.xlsx', sheet_name='Transformer')
print(df_trans.to_string())
print()

# Check for columns with "cost" or "price" in any sheet
print('='*80)
print('SEARCHING FOR COST/PRICE COLUMNS IN ALL SHEETS')
print('='*80)
print()

for sheet in xl.sheet_names:
    df = pd.read_excel('2020.xlsx', sheet_name=sheet)
    cost_cols = [c for c in df.columns if 'cost' in str(c).lower() or 'price' in str(c).lower()]
    if cost_cols:
        print(f'Sheet: {sheet}')
        for col in cost_cols:
            print(f'  - {col}')
            vals = df[col].dropna()
            if len(vals) > 0:
                if vals.dtype in ['float64', 'int64']:
                    print(f'    Unique: {vals.nunique()}, Range: [{vals.min():.4f}, {vals.max():.4f}]')
                else:
                    print(f'    Type: {vals.dtype}, Sample: {vals.iloc[0]}')
        print()
