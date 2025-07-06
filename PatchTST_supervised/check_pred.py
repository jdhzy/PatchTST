# %% [markdown]
# ## 1. Imports and Setup
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler

# %% [markdown]
# ## 2. Define File Paths
# %%
# Path to prediction output
result_dir = '/Users/jdhzy/Desktop/Surfy/forecast/models/PatchTST/PatchTST_supervised/test_results/hawaii_allstations_1440_2160_1440_2160_PatchTST_custom_ftM_sl1440_ll336_pl2160_dm64_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_exp_hawaii_full_0'
pred_path = os.path.join(result_dir, 'pred.npy')

# Path to ground truth data
csv_path = '/Users/jdhzy/Desktop/Surfy/forecast/model_data/hawaii_All.csv'

# %% [markdown]
# ## 3. Load Prediction and Ground Truth
# %%
# Load predicted results
pred = np.load(pred_path, allow_pickle=True)
print("Shape of prediction:", pred.shape)
print("Sample prediction (index 0):\n", pred[0])

# Load ground truth data
df = pd.read_csv(csv_path)
wvht_series = df['WVHT'].values  # Make sure column name is exactly 'WVHT'

# %% [markdown]
# ## 4. Choose Sample and Denormalize
# %%
# Choose prediction sample
sample_index = 0
pred_wvht = pred[sample_index, :, 0]  # Assuming WVHT is index 0

# Define PatchTST parameters
seq_len = 1440
label_len = 336
pred_len = 2160

# Calculate the time window for ground truth
start = seq_len + label_len + sample_index * pred_len
end = start + pred_len
true_wvht = wvht_series[start:end]

# Denormalize predicted WVHT
scaler = StandardScaler()
scaler.fit(wvht_series.reshape(-1, 1))  # Fit scaler on full WVHT data
pred_wvht_real = scaler.inverse_transform(pred_wvht.reshape(-1, 1)).flatten()

# %% [markdown]
# ## 5. Plot Comparison
# %%
plt.figure(figsize=(14, 6))
plt.plot(true_wvht, label='Ground Truth WVHT', linewidth=2)
plt.plot(pred_wvht_real, label='Predicted WVHT (Denorm)', linestyle='--')
plt.title('WVHT Prediction vs. Ground Truth (Sample 0)')
plt.xlabel('Time Step')
plt.ylabel('WVHT (m)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# %%
print("Predicted WVHT mean:", np.mean(pred_wvht_real))
print("Predicted WVHT std deviation:", np.std(pred_wvht_real))
# %%
