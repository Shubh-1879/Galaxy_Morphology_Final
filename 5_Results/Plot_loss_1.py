import numpy as np
import matplotlib.pyplot as plt
import os

# --- Configuration ---
# Paths are relative to your main project folder
MODEL_1_LOSS_PATH = os.path.join('5_Results', 'Model_1', 'train_loss_history.npy')
MODEL_2_LOSS_PATH = os.path.join('5_Results', 'Model_2', 'train_loss_history.npy')
SAVE_PATH = 'training_loss_comparison.png'

# --- Plotting ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 6))
data_loaded = False

# Load and plot data for Model 1
try:
    loss_model_1 = np.load(MODEL_1_LOSS_PATH, allow_pickle=True)
    epochs_1 = range(1, len(loss_model_1) + 1)
    ax.plot(epochs_1, loss_model_1, marker='o', linestyle='-', label='Custom CNN (Model 1)')
    print(f"Loaded {MODEL_1_LOSS_PATH} successfully.")
    data_loaded = True
except Exception as e:
    print(f"Error loading Model 1 data: {e}")

# Load and plot data for Model 2
try:
    loss_model_2 = np.load(MODEL_2_LOSS_PATH, allow_pickle=True)
    epochs_2 = range(1, len(loss_model_2) + 1)
    ax.plot(epochs_2, loss_model_2, marker='s', linestyle='--', label='Fine-Tuned ResNet-18 (Model 2)')
    print(f"Loaded {MODEL_2_LOSS_PATH} successfully.")
    data_loaded = True
except Exception as e:
    print(f"Error loading Model 2 data: {e}")

# --- Formatting ---
ax.set_title('Training Loss Curve Comparison', fontsize=16)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Mean Squared Error (MSE) Loss', fontsize=12)

if data_loaded:
    ax.legend(fontsize=12)
else:
    # Add a message to the plot if no data could be loaded
    ax.text(0.5, 0.5, 'No data found to plot.\nCheck file paths and names.', 
            horizontalalignment='center', verticalalignment='center', 
            transform=ax.transAxes, fontsize=14, color='red')

ax.grid(True)

# Ensure x-axis ticks are integers
from matplotlib.ticker import MaxNLocator
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

plt.tight_layout()
plt.savefig(SAVE_PATH, dpi=300)

print(f"\nPlot saved to {SAVE_PATH}")
plt.show()