import pandas as pd

# file CSV
df = pd.read_csv('spectral-senggigi-raw.csv')

# new columns 1
new_columns = [
    'Kode Sampel',
    'Data XRF - Ni',
    'Data XRF - Fe',
    'Data XRF - SiO2',
    'Data XRF - MgO',
    'Dataset'
]

# new columns 2 (WL)
wl_columns = [f'WL_{i}' for i in range(350, 2501)]
new_columns.extend(wl_columns)

# filter
df = df[new_columns]

# drop N/A data from WL data
df = df.dropna(subset=wl_columns, how='all')

# savefile CSV
df.to_csv('new-spectral-senggigi-raw.csv', index=False)