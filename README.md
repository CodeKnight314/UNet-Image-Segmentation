# UNet Semantic Image Segmentation for Aerial Imagery of Dubai

## Overview

This repository contains the implementation of UNet for Semantic Segmentation of Aerial Images, classifying land, water, buildings, roads, and vegetation. The code includes training and inference scripts to generate semantic maps of input images. This repository is built for educational purposes on the implementation of UNet and its application in real-world scenarios.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/CodeKnight314/UNet-Image-Segmentation.git
    ```

2. Create and activate a virtual environment (optional but recommended):
    ```bash
    python -m venv unet-env
    source unet-env/bin/activate
    ```

3. cd to project directory: 
    ```bash 
    cd UNet-Image-Segmentation/
    ```

4. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Preprocess data: 

1. Download Aerial Image Dataset: 
    ```bash
    kaggle datasets download -d humansintheloop/semantic-segmentation-of-aerial-imagery
    ```

2. Unzip the zip file: 
    ```bash
    unzip semantic-segmentation-of-aerial-imagery
    ```

3. run preprocess_data.py: 
    ```bash
    python3 preprocess_data.py --input_dir Semantic\ segmentation\ of\ aerial\ imagery/ --patch --patch_size 256 --stride 128
    ```

4. move classes.json into UNetData/

## Training:
Run training script:
    ```bash
    python UNet-Image-Segmentation/main.py --root_dir UNetData/ --output_dir UNetLogs/ --epochs 30 --lr 0.0001 --eta_min 0.000001
    ```