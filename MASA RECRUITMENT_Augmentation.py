import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv('new-spectral-senggigi-raw.csv')

input_cols = [f'WL_{i}' for i in range(350, 2501)]
output_cols = ['Data XRF - Ni', 'Data XRF - Fe', 'Data XRF - SiO2', 'Data XRF - MgO']

df_filtered = df[input_cols + output_cols].dropna()

def augment_spectral_data(data, noise_level_input=0.02, noise_level_output=0.01, n_augment=5):
    augmented = []
    for _ in range(n_augment):
        noisy_input = data[input_cols] + np.random.normal(0, noise_level_input * data[input_cols].std(), data[input_cols].shape)
        noisy_output = data[output_cols] + np.random.normal(0, noise_level_output * data[output_cols].std(), data[output_cols].shape)
        augmented.append(pd.concat([noisy_input, noisy_output.reset_index(drop=True)], axis=1))
    return pd.concat(augmented, ignore_index=True)

augmented_df = augment_spectral_data(df_filtered, 0.02, 0.01, 5)

combined_df = pd.concat([df_filtered, augmented_df], ignore_index=True)

train, test = train_test_split(combined_df, test_size=0.25, random_state=42)
train['Dataset'] = 'Train'
test['Dataset'] = 'Test'

final_df = pd.concat([train, test])
final_df.to_csv('augmented-new-spectral-senggigi-raw.csv', index=False)
