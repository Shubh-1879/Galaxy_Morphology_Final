import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from scipy import ndimage

from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


import random

import torch

import torch.nn as nn
import torch.nn.functional as F


import torch.optim as optim
from tqdm import tqdm 

import sys


# Paths
if 'linux' in sys.platform:
    print(">>> Running on HPC (Linux detected)")
    # Define paths for the HPC environment
    home_dir = "/home/shubham.agarwal_phd24"
    project_dir = os.path.join(home_dir, "Model_1")

    # Data is in the home directory
    mapping_path = os.path.join(home_dir, "gz2_filename_mapping.csv")
    labels_path  = os.path.join(home_dir, "gz2_hart16.csv")
    images_dir   = os.path.join(home_dir, "images")
    
    # Output file will be created inside the project folder
    out_merged_csv = os.path.join(project_dir, "merged_labels_assets.csv")

else: # Catches Windows ('win32'), macOS ('darwin'), etc.
    print(">>> Running on Local PC")
    # Define paths for your local Windows PC
    data_dir = r"C:\Users\shubh\Downloads\MSc 3rd Sem\Galaxy_Data"
    project_dir = r"C:\Users\shubh\Downloads\MSc 3rd Sem\Galaxy_Morphology_Final"
    
    mapping_path = os.path.join(data_dir, "gz2_filename_mapping.csv")
    labels_path  = os.path.join(data_dir, "gz2_hart16.csv")
    images_dir   = os.path.join(data_dir, "images")
    out_merged_csv = os.path.join(project_dir, "merged_labels_assets.csv")

# --- You can now use these path variables throughout the rest of your script ---
print(f"Using mapping path: {mapping_path}")
print(f"Using labels path: {labels_path}")
print(f"Using images directory: {images_dir}")



# List of tasks and their debiased columns
tasks = {
    "t01_smooth_or_features": [
        "t01_smooth_or_features_a01_smooth_debiased",
        "t01_smooth_or_features_a02_features_or_disk_debiased",
        "t01_smooth_or_features_a03_star_or_artifact_debiased"
    ],
    "t02_edgeon": [
        "t02_edgeon_a04_yes_debiased",
        "t02_edgeon_a05_no_debiased"
    ],
    "t03_bar": [
        "t03_bar_a06_bar_debiased",
        "t03_bar_a07_no_bar_debiased"
    ],
    "t04_spiral": [
        "t04_spiral_a08_spiral_debiased",
        "t04_spiral_a09_no_spiral_debiased"
    ],
    "t05_bulge_prominence": [
        "t05_bulge_prominence_a10_no_bulge_debiased",
        "t05_bulge_prominence_a11_just_noticeable_debiased",
        "t05_bulge_prominence_a12_obvious_debiased",
        "t05_bulge_prominence_a13_dominant_debiased"
    ],
    "t06_odd": [
        "t06_odd_a14_yes_debiased",
        "t06_odd_a15_no_debiased"
    ],
    "t07_rounded": [
        "t07_rounded_a16_completely_round_debiased",
        "t07_rounded_a17_in_between_debiased",
        "t07_rounded_a18_cigar_shaped_debiased"
    ],
    "t08_odd_feature": [
        "t08_odd_feature_a19_ring_debiased",
        "t08_odd_feature_a20_lens_or_arc_debiased",
        "t08_odd_feature_a21_disturbed_debiased",
        "t08_odd_feature_a22_irregular_debiased",
        "t08_odd_feature_a23_other_debiased",
        "t08_odd_feature_a24_merger_debiased"
    ],
    "t09_bulge_shape": [
        "t09_bulge_shape_a25_round_debiased",
        "t09_bulge_shape_a26_boxy_debiased",
        "t09_bulge_shape_a27_no_bulge_debiased"
    ],
    "t10_spiral_winding": [
        "t10_spiral_winding_a28_tight_debiased",
        "t10_spiral_winding_a29_medium_debiased",
        "t10_spiral_winding_a30_loose_debiased"
    ],
    "t11_spiral_arm_count": [
        "t11_spiral_arm_count_a31_1_debiased",
        "t11_spiral_arm_count_a32_2_debiased",
        "t11_spiral_arm_count_a33_3_debiased",
        "t11_spiral_arm_count_a34_4_debiased",
        "t11_spiral_arm_count_a36_more_than_4_debiased",
        "t11_spiral_arm_count_a37_cant_tell_debiased"
    ]
}




# Now, we need to finalize the target columns, extract the 37 debiased probability columns (make sure they sum to 1 within each task, but not necessarily across all 37 since some tasks are conditional) and save them in a NumPy array (shape: [num_samples, 37]).


# --- 2. Data Loading and Merging ---
df_map = pd.read_csv(mapping_path, dtype={"objid": str, "asset_id": str})
df_lbl = pd.read_csv(labels_path, dtype={"dr7objid": str})
df = pd.merge(df_map, df_lbl, left_on="objid", right_on="dr7objid", how="inner")

# --- 3. CRITICAL FIX: Filter Missing Files BEFORE Splitting ---
print("Checking for existing image files...")
df["filename"] = df["asset_id"].astype(str) + ".jpg"
df["file_path"] = df["filename"].apply(lambda f: os.path.join(images_dir, f))
df["file_exists"] = df["file_path"].apply(os.path.exists)

df_filtered = df[df["file_exists"]].copy()
print(f"Found {len(df_filtered)} images out of {len(df)} total records.")



# --- 4. Prepare Labels and Perform Stratified Splitting on Filtered Data ---
target_cols = [col for col in df_filtered.columns if col.endswith("_debiased")]
df_filtered[target_cols] = df_filtered[target_cols].fillna(0.0)
Y_filtered = df_filtered[target_cols].to_numpy()

# Save the cleaned, merged, and filtered dataframe
df_filtered.to_csv(out_merged_csv, index=False)

print("Performing stratified split...")
# Outer split to create a smaller subset for train/val
msss_outer = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=15000, random_state=42)
_, subset_idx = next(msss_outer.split(np.zeros(len(Y_filtered)), Y_filtered))

# Inner split on the subset
subset_Y = Y_filtered[subset_idx]


print(subset_idx.shape)
print(subset_Y.shape)


msss_inner = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=5000, random_state=42)
train_idx, val_idx = next(msss_inner.split(np.zeros(len(subset_Y)), subset_Y))

# Map indices back to the original filtered dataframe's index
# CORRECTED code
subset_train_idx = subset_idx[train_idx]
subset_val_idx   = subset_idx[val_idx]

print(f"Created train set with {len(subset_train_idx)} samples and val set with {len(subset_val_idx)} samples.")


# Create the final train and validation dataframes from the filtered dataframe
train_df = df_filtered.iloc[subset_train_idx]
val_df   = df_filtered.iloc[subset_val_idx]

# --- 5. Define Parameters ---
img_size = (128, 128)

import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import random
from scipy import ndimage # You will need to install this

class FullImageCanonicalRotation:
    """
    A transform that finds the brightest region of the WHOLE image,
    rotates the WHOLE image to a canonical orientation,
    and then takes a random crop.
    """
    def __init__(self, output_size, brightness_threshold=0.75):
        self.output_size = output_size
        self.brightness_threshold = brightness_threshold

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            img = transforms.ToPILImage()(img)
        
        # Use a smaller version for faster centroid calculation
        thumb = img.resize((64, 64))
        thumb_gray = thumb.convert("L")
        thumb_np = np.array(thumb_gray)

        # Find the center of mass of the brightest pixels
        threshold_value = np.max(thumb_np) * self.brightness_threshold
        bright_pixels = thumb_np > threshold_value
        
        if np.sum(bright_pixels) > 0:
            # ndimage.center_of_mass gives (row, col) which is (y, x)
            y, x = ndimage.center_of_mass(bright_pixels)
            
            # Determine quadrant and rotation angle based on the thumbnail's center (32,32)
            if y < 32 and x < 32: rotation_angle = 180   # Top-left -> Bottom-right
            elif y < 32 and x >= 32: rotation_angle = -90  # Top-right -> Bottom-right
            elif y >= 32 and x < 32: rotation_angle = 90   # Bottom-left -> Bottom-right
            else: rotation_angle = 0
            
            # Rotate the ORIGINAL, full-sized image
            img = img.rotate(rotation_angle)

      
        return img
    
canonical_full_image_transform = FullImageCanonicalRotation(output_size=img_size)



train_transform = transforms.Compose([
    canonical_full_image_transform,
    transforms.RandomRotation(180),  
    transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor()
])



val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])



class GalaxyDataset(Dataset):
    def __init__(self, dataframe, img_dir, target_cols, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.target_cols = target_cols
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["filename"])
        image = Image.open(img_path).convert("RGB")  # keep RGB for augmentation
        label = row[self.target_cols].values.astype("float32")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label)



# Build datasets from merged dataframe (df) and target_cols
train_df = df_filtered.iloc[subset_train_idx]
val_df   = df_filtered.iloc[subset_val_idx]

train_dataset = GalaxyDataset(train_df, images_dir, target_cols, transform=train_transform)
val_dataset   = GalaxyDataset(val_df, images_dir, target_cols, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)


# In[24]:


images, labels = next(iter(train_loader))

fig, axes = plt.subplots(1, 8, figsize=(20, 4))
for i, ax in enumerate(axes):
    img = images[i].permute(1, 2, 0).numpy()
    ax.imshow(img)
    ax.axis("off")
plt.show()


# In[25]:


class StableCNN(nn.Module): # Renamed for clarity
    def __init__(self):
        super(StableCNN, self).__init__()

        # Convolutional Block 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32) # Add BatchNorm for 32 channels
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Block 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64) # Add BatchNorm for 64 channels
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Block 3
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128) # Add BatchNorm for 128 channels
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()

        # Create a dummy forward pass to find the flattened size
        dummy_input = torch.randn(1, 3, 128, 128)
        flattened_size = self._forward_features(dummy_input).shape[1]

        # Use the dynamic size for fully connected head
        self.fc1 = nn.Linear(flattened_size, 256)
        self.dropout = nn.Dropout(p=0.5) # 50% dropout rate
        self.fc2 = nn.Linear(256, 37)

    def _forward_features(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.flatten(x)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x

model = StableCNN()











# After creating your train_loader, before the training loop
print("--- Checking a batch of data ---")
inputs, labels = next(iter(train_loader))
print(f"Inputs min: {inputs.min()}, max: {inputs.max()}")
print(f"Labels min: {labels.min()}, max: {labels.max()}")

# Check if any values are nan
print(f"Any nans in inputs? {torch.isnan(inputs).any()}")
print(f"Any nans in labels? {torch.isnan(labels).any()}")
print("---------------------------------")




# --- 1. SETUP ---
# (Assuming model, train_loader, and val_loader are already defined)

model = StableCNN()
# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, verbose=True)


# Set the device (use GPU if available, otherwise CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Move the model to the chosen device
model.to(device)

# --- 2. TRAINING ---
num_epochs = 150
best_val_loss = float('inf') # Initialize with a very high value

for epoch in range(num_epochs):
    # --- Training Phase ---
    model.train() # Set the model to training mode
    running_train_loss = 0.0

    # Use tqdm for a progress bar
    train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]")

    for batch_idx, (inputs, labels) in enumerate(train_progress_bar):
        # Move data to the device
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)


        if torch.isnan(loss):
            print(f"\n!!! NaN loss detected at batch index: {batch_idx} !!!")

            # Save the problematic batch for inspection
            torch.save(inputs, 'bad_batch_inputs.pt')
            torch.save(labels, 'bad_batch_labels.pt')

            print("Problematic batch saved to 'bad_batch_inputs.pt' and 'bad_batch_labels.pt'")
            print("Stopping training.")
            
            sys.exit()

        # Backward pass and optimize
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Update running loss and progress bar
        running_train_loss += loss.item()
        train_progress_bar.set_postfix(loss=loss.item())

    # --- Validation Phase ---
    model.eval() # Set the model to evaluation mode
    running_val_loss = 0.0

    # Disable gradient calculations for validation
    with torch.no_grad():
        val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]")
        for inputs, labels in val_progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_val_loss += loss.item()
            val_progress_bar.set_postfix(loss=loss.item())

    # --- 3. REPORTING & SAVING ---
    # Calculate average losses
    avg_train_loss = running_train_loss / len(train_loader)
    avg_val_loss = running_val_loss / len(val_loader)

    scheduler.step(avg_val_loss)
    # Calculate RMSE for the validation set
    val_rmse = torch.sqrt(torch.tensor(avg_val_loss))

    print(f"Epoch {epoch+1}/{num_epochs} -> "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | "
          f"Val RMSE: {val_rmse:.4f}")

    # Save the model if it has the best validation loss so far
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_galaxy_model.pth')
        print(f"🎉 New best model saved with Val Loss: {best_val_loss:.4f}")

print("\n--- Training Finished ---")

