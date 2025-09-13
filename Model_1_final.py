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
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm 
import sys
from torchvision import transforms

# Paths
if 'linux' in sys.platform:
    print(">>> Running on HPC (Linux detected)")
    # Define paths for the HPC environment

    # NEW, ORGANIZED path definitions
    project_dir = "/home/shubham.agarwal_phd24/Galaxy_Classification"
    home_dir = "/home/shubham.agarwal_phd24"
    # Define subdirectories
    data_dir = os.path.join(project_dir, "1_Data")
    model_dir = os.path.join(project_dir, "4_Models", "Model_1")
    results_dir = os.path.join(project_dir, "5_Results", "Model_1")
    checkpoint_dir = os.path.join(model_dir, "checkpoints")
    # Ensure model and checkpoint directories exist
    os.makedirs(os.path.join(model_dir, "checkpoints"), exist_ok=True)

    # Paths to data
    mapping_path = os.path.join(data_dir, "raw", "gz2_filename_mapping.csv")
    labels_path  = os.path.join(data_dir, "raw", "gz2_hart16.csv")
    images_dir   = os.path.join(data_dir, "raw", "images")
        
    train_history_path = os.path.join(results_dir, 'train_loss_history.npy')
    # Path for processed CSV
    out_merged_csv = os.path.join(data_dir, "processed", "merged_labels_assets.csv")
    checkpoint_path_template = os.path.join(checkpoint_dir, 'checkpoint_epoch_{}.pth')
    test_set_path = os.path.join(data_dir, "processed", "final_galaxy_test_set.csv")

    # Path to save the final model
    final_model_path = os.path.join(model_dir, 'final_stable_cnn_model.pth')
    
   
    
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



img_size = (128, 128)


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


# --- 4. Prepare Labels and Perform Final Train/Test Split on Filtered Data ---
target_cols = [col for col in df_filtered.columns if col.endswith("_debiased")]
df_filtered[target_cols] = df_filtered[target_cols].fillna(0.0)
Y_filtered = df_filtered[target_cols].to_numpy()

print("Performing final 80/20 train/test split...")
# Use a single split for the entire filtered dataset
msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

# This gives you the indices for your final training and test sets
train_idx, test_idx = next(msss.split(np.zeros(len(Y_filtered)), Y_filtered))

# Create the final train and test dataframes
train_df = df_filtered.iloc[train_idx]
test_df  = df_filtered.iloc[test_idx]

print(f"Created final train set with {len(train_df)} samples.")
print(f"Created final test set with {len(test_df)} samples.")

# --- Save the test set for later evaluation ---
test_df.to_csv(test_set_path, index=False)
print("Held-out test set saved to 'final_galaxy_test_set.csv'")



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






# (Make sure your transform classes and variables are defined before this point)
train_dataset = GalaxyDataset(train_df, images_dir, target_cols, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)

# We do not use a validation loader for the final training run
val_loader = None

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




if __name__ == "__main__":

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


    # --- FINAL TRAINING LOOP ---
    num_epochs = 250 # Set the total number of epochs for final training
    model.to(device)

    # --- Save Training History ---
    train_loss_history = []

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Final Training]")

        for inputs, labels in train_progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
            train_progress_bar.set_postfix(loss=loss.item())

        # --- Reporting & Saving ---
        avg_train_loss = running_train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)
        print(f"Epoch {epoch+1}/{num_epochs} -> Train Loss: {avg_train_loss:.4f}")
        
        # Optional: Save a checkpoint every few epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), checkpoint_path_template.format(epoch + 1))
            print(f"--- Checkpoint saved for epoch {epoch+1} ---")


    torch.save(model.state_dict(), final_model_path)
    np.save(train_history_path, np.array(train_loss_history))

    print(f"\n--- Final model saved to {final_model_path} ---")
    print(f"--- Training history saved to train_loss_history.npy ---")
    print("\n--- Training Finished ---")

