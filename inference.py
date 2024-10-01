from model import UNet
import argparse
from glob import glob
import os
from tqdm import tqdm
from torchvision import transforms as T
import torch 
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np

def inference(model : torch.nn.Module, input_dir : str, output_dir : str): 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = model.to(device)
    
    img_dir = glob(os.path.join(input_dir, "test/images/*.png"))
    mask_dir = glob(os.path.join(input_dir, "test/masks/*.png"))
    
    idx_to_color_dict = {
        0: [155, 155, 155],      # Unlabeled
        1: [226, 169, 41],      # Water
        2: [132, 41, 246],      # Land
        3: [110, 193, 228],      # Road
        4: [60, 16, 152],       # Building
        5: [254, 221, 58],      # Vegetation
    }
    
    transform = T.Compose([
        T.Resize((512,512)),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
    ])
    
    iteration = 0
    model.eval()
    for img, mask in tqdm(zip(img_dir, mask_dir)): 
        img_tensor = transform(Image.open(img).convert("RGB")).unsqueeze(0).to(device)
        
        H, W = img_tensor.shape[2], img_tensor.shape[3]

        with torch.no_grad(): 
            prediction = model(img_tensor)
        prediction = torch.argmax(prediction, dim=1).squeeze(0).cpu().numpy()
        predicted_mask_np = np.zeros((H, W, 3), dtype=np.uint8)
        for class_idx, color in idx_to_color_dict.items():
            predicted_mask_np[prediction == class_idx] = color
        
        img_np = np.array(Image.open(img).convert("RGB"))
        mask_np = np.array(Image.open(mask).convert("RGB"))
        
        fig, ax = plt.subplots(1, 3, figsize=(10, 5))
        
        ax[0].imshow(img_np)
        ax[0].set_title("Original Image")
        ax[0].axis('off')    
        
        ax[1].imshow(predicted_mask_np)
        ax[1].set_title("Predicted Mask")
        ax[1].axis('off')
        
        ax[2].imshow(mask_np)
        ax[2].set_title("Ground Truth Mask")    
        ax[2].axis('off')
        
        plt.savefig(f"{output_dir}/inference_{str(iteration).zfill(7)}.png")
        plt.close(fig)
        iteration+=1
    
    print("Inference Job Complete")

if __name__ == "__main__": 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--model_dict", type=str, required=True, help="Directory to UNet weights")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory to test images") 
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to output visualized images")
    parser.add_argument("--num_classes", type=int, required=True, help="number of classes for predicting")
    args = parser.parse_args() 
    
    model = UNet(num_classes=args.num_classes)
    model.load_state_dict(torch.load(args.model_dict, weights_only=True))
    
    inference(model, args.input_dir, args.output_dir)