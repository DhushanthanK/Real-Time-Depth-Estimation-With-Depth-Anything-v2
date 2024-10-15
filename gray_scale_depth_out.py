import os
import cv2
import numpy as np
import torch
from depth_anything_v2.dpt import DepthAnythingV2
import torch

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'

# Model configurations
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

# Choose the encoder you want to use
encoder = 'vits'  # options: 'vits', 'vitb', 'vitl', 'vitg'

# Initialize the model
model = DepthAnythingV2(**model_configs[encoder])

# Load the model weights
model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu', weights_only=True))
model = model.to(DEVICE).eval()

# Create output directory if it doesn't exist
output_dir = 'depth_output'
os.makedirs(output_dir, exist_ok=True)

raw_img = cv2.imread('assets/examples/demo16.jpg')
raw_img = cv2.resize(raw_img, (512, 512))  # Resize to 512x512 or your desired size

if raw_img is None:
    print("Error: Could not read the image. Check the file path.")
else:
    # Perform depth estimation
    depth = model.infer_image(raw_img)  # HxW raw depth map in numpy

    # Check the depth map
    print("Depth map shape:", depth.shape)
    print("Depth map min:", depth.min(), "max:", depth.max())

    # Save the depth output as an image
    depth_image = (depth * 255).astype(np.uint8)  # Scale depth for visualization
    depth_output_path = os.path.join(output_dir, 'depth_map.png')
    cv2.imwrite(depth_output_path, depth_image)

    print(f"Depth estimation completed. Output saved to {depth_output_path}.")