# Real-Time Depth Estimation with Depth Anything V2

This project utilizes the Depth Anything V2 model to perform real-time depth estimation using a webcam or to process static images. The code provides three main functionalities:

1. **Real-time Camera Depth Map**: Captures video from the webcam and displays the depth map in real time.
   <img width="1024" alt="Screenshot 2024-10-15 at 13 35 00" src="https://github.com/user-attachments/assets/9f7f3521-06ad-4ad1-ac8a-473869321525">
3. **Colored Depth Output**: Takes an input image and outputs a colored depth map.
   ![depth_colored_map](https://github.com/user-attachments/assets/72aaf72b-7689-4300-9042-81a8ac7c5142)
5. **Grayscale Depth Output**: Takes an input image and outputs a grayscale depth map.
   ![depth_map](https://github.com/user-attachments/assets/9e43f9d4-fa21-4164-a6eb-01f2b5ef3896)

## Requirements

To run this project, you need to install the following Python packages. You can use `pip` to install them. A `requirements.txt` file is provided for your convenience:

gradio_imageslider
gradio==4.29.0
matplotlib
opencv-python
torch
torchvision


To install the dependencies, run:

```bash
pip install -r requirements.txt
```


## Pre-trained Models

Download the following pre-trained models and place them in the `checkpoints` directory.

- **Depth-Anything-V2-Small (24.8M)**:

  - [Download](https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true)

- **Depth-Anything-V2-Base (97.5M)**:

  - [Download](https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth?download=true)

- **Depth-Anything-V2-Large (335.3M)**:
  - [Download](https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true)

### Instructions for Downloading

You can download the models using the following commands:

````bash
mkdir -p checkpoints  # Create the checkpoints directory if it doesn't exist

# Download the pre-trained models
wget -O checkpoints/depth_anything_v2_vits.pth https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true

wget -O checkpoints/depth_anything_v2_vitb.pth https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth?download=true

wget -O checkpoints/depth_anything_v2_vitl.pth https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true
````


## Use

1. Real-Time Camera Depth Map

To run the real-time depth estimation from the webcam, execute the following script:
```bash
python real_time_camera_depth_map.py
````
Ensure that your webcam is connected and functioning. Press ‘q’ to exit the webcam window.

2. Colored Depth Output

To generate a colored depth map from a static image, use:

```bash
python colored_depth_out.py
```

Make sure to place your input image at assets/examples/demo16.jpg or modify the script to point to your desired image path.

3. Grayscale Depth Output

For generating a grayscale depth map from a static image, execute:

```bash
python gray_scale_depth_out.py
```
Again, ensure your input image is correctly specified in the script.


## Model Configuration

The model supports several encoders. You can choose from the following options:

    •	vits
    •	vitb
    •	vitl
    •	vitg

The encoder can be selected by changing the encoder variable in the script.


## Output

All depth maps are saved in the depth_output directory. If the directory does not exist, it will be created automatically.

Acknowledgments

## Licensing

This project is licensed under the [Apache License 2.0](LICENSE-APACHE) for the original Depth Anything V2 framework.

Additional contributions are licensed under the [MIT License](LICENSE).

For more information on the Depth Anything V2 model, please refer to the official repository.
