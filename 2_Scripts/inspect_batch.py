import torch
import matplotlib.pyplot as plt
import numpy as np

print("--- Loading problematic batch ---")

try:
    # Load the saved tensors to the CPU
    inputs = torch.load('bad_batch_inputs.pt', map_location=torch.device('cpu'))
    labels = torch.load('bad_batch_labels.pt', map_location=torch.device('cpu'))

    print(f"Inputs tensor shape: {inputs.shape}")
    print(f"Labels tensor shape: {labels.shape}")

    # --- Data Inspection ---
    print(f"\nPixel value range in inputs: min={inputs.min()}, max={inputs.max()}")
    print(f"Label value range: min={labels.min()}, max={labels.max()}")

    # Check for non-finite values (NaN or infinity)
    if not torch.isfinite(inputs).all():
        print("CRITICAL: Inputs contain non-finite values (NaNs or infinities)!")
    if not torch.isfinite(labels).all():
        print("CRITICAL: Labels contain non-finite values (NaNs or infinities)!")

    # Check for label sums (for classification, sometimes useful)
    print(f"Example label sums: {labels.sum(dim=1)[:5]}")

    # --- Visualization ---
    batch_size = inputs.shape[0]
    fig, axes = plt.subplots(batch_size, 1, figsize=(10, batch_size * 3))
    if batch_size == 1:
        axes = [axes]

    print("\n--- Visualizing Images and Labels ---")
    for i in range(batch_size):
        img_tensor = inputs[i]
        label_vector = labels[i]
        
        img_np = img_tensor.cpu().permute(1, 2, 0).numpy()
        
        axes[i].imshow(img_np)
        axes[i].set_title(f"Image #{i} in Batch")
        axes[i].axis('off')
        
        print(f"\nLabels for Image #{i}:")
        print(label_vector.cpu().numpy())

    plt.tight_layout()
    plt.savefig("problematic_batch_visualization.png")
    print("\n>>> Visualization saved to 'problematic_batch_visualization.png'")

except FileNotFoundError:
    print("Error: Could not find 'bad_batch_inputs.pt' or 'bad_batch_labels.pt'.")
except Exception as e:
    print(f"An error occurred: {e}")
