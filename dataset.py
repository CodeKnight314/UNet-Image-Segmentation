import torch 
import os
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
from glob import glob
from PIL import Image 
import random
import numpy as np

class SegementationDataset(Dataset): 
    def __init__(self, root_dir, mode, patch_size): 
        super().__init__() 
        
        self.patch_size = patch_size
        
        self.root_dir = os.path.join(root_dir, mode)
        self.img_dir = glob(os.path.join(self.root_dir, "images") + "/*")
        self.mask_dir = glob(os.path.join(self.root_dir, "masks") + "/*")
        
        self.img_transform = T.Compose([
            T.ColorJitter(.1,.1,.1,.1),
            T.GaussianBlur(3, sigma=(0.1, 2.0)), 
            T.ToTensor(), 
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.mask_dict = {
            "Unlabeled" : [61, 61, 61],
            "Water" : [31, 89, 76],
            "Land" : [96, 65, 14], 
            "Road" : [87, 35, 50],
            "Building" : [82, 1, 11], 
            "Vegetation" : [25, 46, 2],
        }
        
        self.mask_label = list(self.mask_dict.keys())
        
        self.mask_transform = T.Compose([
            T.ToTensor()
        ])
    
    def rgb_to_label(self, mask):
        mask_np = np.array(mask)
        class_mask = np.zeros((mask_np.shape[0], mask_np.shape[1]), dtype=np.int64)
        
        for rgb, class_index in self.rgb_to_class.items():
            match = np.all(mask_np == rgb, axis=-1)
            class_mask[match] = class_index
        
        return torch.tensor(class_mask, dtype=torch.long)
                   
    def __len__(self): 
        return len(self.img_dir)
    
    def __getitem__(self, index): 
        img = self.img_transform(Image.open(self.img_dir[index]).convert("RGB"))
        mask = Image.open(self.mask_dir[index])
        mask = self.rgb_to_label(mask)
        one_hot_mask = self.rgb_to_one_hot_label(mask)
        
        if random.random() < 0.5: 
            img = T.functional.hflip(img)
            mask = T.functional.hflip(mask)
            one_hot_mask = T.functional.hflip(one_hot_mask)
            
        if random.random() < 0.5: 
            img = T.functional.vflip(img)
            mask = T.functional.vflip(mask)
            one_hot_mask = T.functional.vflip(one_hot_mask)

        img_h, img_w = img.shape[1], img.shape[2]
        
        start_x, start_y = random.randint(0, img_w - self.patch_size), random.randint(0, img_h - self.patch_size)
        img = img[:, start_x:start_x + self.patch_size, start_y:start_y + self.patch_size]
        mask = mask[:, start_x:start_x + self.patch_size, start_y:start_y + self.patch_size]
        one_hot_mask = one_hot_mask[:, start_x:start_x + self.patch_size, start_y:start_y + self.patch_size]
        
        return img, mask, one_hot_mask

def load_dataset(root_dir : str, mode : str = "train", patch_size = 256, batch_size : int = 16):
    """
    Helper Function to load dataset for selected mode with selected patch size and batch size.
    """
    return DataLoader(SegementationDataset(root_dir=root_dir, mode=mode, patch_size=patch_size), batch_size=batch_size, shuffle=True, num_workers=os.cpu_count())
    

        
        