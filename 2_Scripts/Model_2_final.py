
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

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm 
import sys
from torchvision import transforms
import torchvision.models as models




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



def create_resnet18_model(num_outputs=37, pretrained=True):
    """
    Creates a ResNet18 model for fine-tuning.
    
    Args:
        num_outputs (int): The number of output classes.
        pretrained (bool): Whether to use pre-trained ImageNet weights.
    
    Returns:
        A PyTorch model ready for fine-tuning.
    """
    # 1. Load a ResNet18 model
    if pretrained:
        print("Loading PRE-TRAINED ResNet18 model...")
        model = models.resnet18(pretrained=True)
    else:
        print("Loading ResNet18 model from SCRATCH...")
        model = models.resnet18(pretrained=False)

    # 2. Freeze all the parameters in the pre-trained model
    if pretrained:
        for param in model.parameters():
            param.requires_grad = False

    # 3. Replace the classifier head for our specific problem
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, num_outputs), #512->37
        nn.Sigmoid()
    )
    
    return model


img_size = (128, 128)


        
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





# This block allows you to run this script directly to test the model
if __name__ == '__main__':
    # Paths
    if 'linux' in sys.platform:
        print(">>> Running on HPC (Linux detected)")
        # Define paths for the HPC environment

        # NEW, ORGANIZED path definitions
        project_dir = "/home/shubham.agarwal_phd24/Galaxy_Classification"
        home_dir = "/home/shubham.agarwal_phd24"
        # Define subdirectories
        data_dir = os.path.join(project_dir, "1_Data")
        model_dir = os.path.join(project_dir, "4_Models", "Model_2")
        results_dir = os.path.join(project_dir, "5_Results", "Model_2")
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
        
        model_dir = os.path.join(project_dir, "4_Models", "Model_2")
        results_dir = os.path.join(project_dir, "5_Results", "Model_2")
        checkpoint_dir = os.path.join(model_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

        mapping_path = os.path.join(data_dir, "gz2_filename_mapping.csv")
        labels_path  = os.path.join(data_dir, "gz2_hart16.csv")
        images_dir   = os.path.join(data_dir, "images")
        out_merged_csv = os.path.join(project_dir, "merged_labels_assets.csv")
        test_set_path = os.path.join(project_dir, "final_galaxy_test_set.csv")

        # Paths for model outputs
        final_model_path = os.path.join(model_dir, 'final_stable_cnn_model.pth')
        checkpoint_path_template = os.path.join(checkpoint_dir, 'checkpoint_epoch_{}.pth')
        train_history_path = os.path.join(results_dir, 'train_loss_history.npy')


    print(f"Using mapping path: {mapping_path}")
    print(f"Using labels path: {labels_path}")
    print(f"Using images directory: {images_dir}")


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


    # Create an instance of Model 2
    model_2 = create_resnet18_model(pretrained=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model_2.fc.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, verbose=True)

    # Set the device (use GPU if available, otherwise CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Print a summary of the model
    print("\n--- Model 2 (ResNet18 Fine-Tuned) Architecture ---")
    print(model_2)


    
    print("\n--- Trainable Parameters ---")
    for name, param in model_2.named_parameters():
        if param.requires_grad:
            print(name)

    print("\n--- Starting Training for Model 2 ---") 
    # Move the model to the selected device
    model_2.to(device)

    # Number of epochs to train for
    num_epochs = 250

    # To store the training loss history
    train_loss_history = []

    for epoch in range(num_epochs):
        # Set the model to training mode
        model_2.train()
        
        running_loss = 0.0
        
        
        # Loop over the training data
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # Move images and labels to the device (GPU or CPU)
            images = images.to(device)
            labels = labels.to(device)
            
            # 1. Zero the gradients
            optimizer.zero_grad()
            
            # 2. Forward pass: compute predicted outputs
            outputs = model_2(images)
            
            # 3. Calculate the loss
            loss = criterion(outputs, labels)
            
            # 4. Backward pass: compute gradient of the loss
            loss.backward()
            
            # 5. Update weights: call step() on the optimizer
            optimizer.step()
            
            # Update the running loss
            running_loss += loss.item() * images.size(0)
        
        # Calculate the average loss for the epoch
        epoch_loss = running_loss / len(train_loader.dataset)
        train_loss_history.append(epoch_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {epoch_loss:.6f}")
        
        # Step the learning rate scheduler based on the training loss
        scheduler.step(epoch_loss)

        # Save a checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = checkpoint_path_template.format(epoch+1)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model_2.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    print("\n--- Training Finished ---")

    # Save the final trained model
    torch.save(model_2.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

    # Save the training loss history
    
    np.save(train_history_path, np.array(train_loss_history))
    print(f"Training loss history saved to {train_history_path}")
