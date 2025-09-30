import torch
import pandas as pd
import numpy as np
import os
import sys
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend for HPC
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import torch.multiprocessing as mp

mp.set_sharing_strategy('file_system')

# --- 1. Set Up Paths and Imports ---

# Add the scripts directory to the Python path to allow importing from other scripts
scripts_dir = "/home/shubham.agarwal_phd24/Galaxy_Classification/2_Scripts"
sys.path.append(scripts_dir)

# Now you can import your custom classes
from Model_1_final import StableCNN, GalaxyDataset, val_transform, target_cols

# Define project directories
project_dir = "/home/shubham.agarwal_phd24/Galaxy_Classification"
data_dir = os.path.join(project_dir, "1_Data")
model_dir = os.path.join(project_dir, "4_Models", "Model_1")
results_dir = os.path.join(project_dir, "5_Results", "Model_1")

# Define full paths to required files
images_dir = os.path.join(data_dir, "raw", "images")
test_set_path = os.path.join(data_dir, "processed", "final_galaxy_test_set.csv")
final_model_path = os.path.join(model_dir, 'final_stable_cnn_model.pth')
plot_path = os.path.join(results_dir, 'final_performance_plot.png')

print("--- Starting Evaluation ---")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --- 2. Load Model and Test Data ---
print(f"Loading test data from: {test_set_path}")
test_df = pd.read_csv(test_set_path)

test_dataset = GalaxyDataset(test_df, images_dir, target_cols, transform=val_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=8)

print(f"Loading trained model from: {final_model_path}")
model = StableCNN()
model.load_state_dict(torch.load(final_model_path, map_location=device))
model.to(device)
model.eval() # Set model to evaluation mode

# --- 3. Make Predictions on the Test Set ---
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Making predictions on Test Set"):
        inputs = inputs.to(device)
        outputs = model(inputs)
        all_preds.append(outputs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

all_preds = np.concatenate(all_preds, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

# --- 4. Calculate Final Performance ---
overall_rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
print(f"\n>>> Final Test Set RMSE: {overall_rmse:.4f} <<<")

# --- 5. Plot and Save Performance Figure ---
per_task_rmse = np.sqrt(np.mean((all_labels - all_preds)**2, axis=0))
plt.figure(figsize=(15, 8))
plt.bar(range(len(target_cols)), per_task_rmse)
plt.xlabel("Task (Morphology Feature)")
plt.ylabel("RMSE Score")
plt.title(f"Per-Task RMSE on Test Set (Overall RMSE: {overall_rmse:.4f})")
plt.xticks(ticks=range(len(target_cols)), labels=[col.split('_')[-2] for col in target_cols], rotation=90, fontsize=8)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

plt.savefig(plot_path)
print(f"Performance plot saved to: {plot_path}")
