import os
import cv2
import numpy as np
import torch
from depth_anything_v2.dpt import DepthAnythingV2

# Set device
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

# Open the webcam
cap = cv2.VideoCapture(0)  # 0 is usually the default webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Create output directory if it doesn't exist
output_dir = 'depth_output'
os.makedirs(output_dir, exist_ok=True)

while True:
    # Capture frame-by-frame from the webcam
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to capture frame from webcam.")
        break

    # Resize the frame to the required input size for the model
    frame_resized = cv2.resize(frame, (512, 512))

    # Perform depth estimation
    depth = model.infer_image(frame_resized)  # HxW raw depth map in numpy

    # Normalize the depth map to a 0-255 range for visualization
    depth_min = depth.min()
    depth_max = depth.max()
    depth_normalized = (depth - depth_min) / (depth_max - depth_min)  # Normalize between 0 and 1
    depth_normalized = (depth_normalized * 255).astype(np.uint8)      # Convert to 0-255 scale

    # Apply colormap (e.g., INFERNO)
    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)

    # Display the original frame and depth map side by side
    combined_frame = np.hstack((frame_resized, depth_colored))  # Combine original frame and depth map
    cv2.imshow('Real-Time Depth Estimation', combined_frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()