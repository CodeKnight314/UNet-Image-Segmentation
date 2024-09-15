import torch
import os
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
from glob import glob
from PIL import Image
import random
import numpy as np

class SegmentationDataset(Dataset):
    def __init__(self, root_dir: str, mode: str, patch_size: int) -> None:
        """
        Initializes the SegmentationDataset.

        Args:
            root_dir (str): The root directory containing 'train' and 'val' folders with 'images' and 'masks' subdirectories.
            mode (str): Either 'train', 'val' or 'test' to specify the dataset split.
            patch_size (int): The size of the square patches to extract from the images.
        """
        super().__init__()

        self.patch_size = patch_size

        self.root_dir = os.path.join(root_dir, mode)
        self.img_dir = sorted(glob(os.path.join(self.root_dir, "images", "*")))
        self.mask_dir = sorted(glob(os.path.join(self.root_dir, "masks", "*")))

        assert len(self.img_dir) == len(self.mask_dir), "Number of images and masks do not match"

        # Define image transformations
        self.img_transform = T.Compose([
            T.ColorJitter(0.1, 0.1, 0.1, 0.1),
            T.GaussianBlur(3, sigma=(0.1, 2.0)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Define label mappings (assuming RGB color masks)
        self.mask_dict = {
            "Unlabeled": [61, 61, 61],
            "Water": [31, 89, 76],
            "Land": [96, 65, 14],
            "Road": [87, 35, 50],
            "Building": [82, 1, 11],
            "Vegetation": [25, 46, 2],
        }

        # Map RGB values to class indices
        self.rgb_to_class = {}
        for idx, (class_name, rgb) in enumerate(self.mask_dict.items()):
            self.rgb_to_class[tuple(rgb)] = idx

        self.mask_transform = T.Compose([
            T.ToTensor()
        ])

    def rgb_to_label(self, mask: Image.Image) -> torch.Tensor:
        """
        Converts an RGB mask image to a label map where each pixel value corresponds to a class index.

        Args:
            mask (PIL.Image.Image): The input mask image in RGB format.

        Returns:
            torch.Tensor: A tensor of shape (H, W) containing class indices for each pixel.
        """
        mask_np = np.array(mask)
        class_mask = np.zeros((mask_np.shape[0], mask_np.shape[1]), dtype=np.int64)

        # Map each RGB pixel to a class index
        for rgb, class_index in self.rgb_to_class.items():
            match = np.all(mask_np == rgb, axis=-1)
            class_mask[match] = class_index

        return torch.tensor(class_mask, dtype=torch.long)

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.img_dir)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Gets a sample from the dataset at the given index.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the image tensor (C, H, W) and the mask tensor (H, W).
        """
        # Load image and mask
        img = Image.open(self.img_dir[index]).convert("RGB")
        mask = Image.open(self.mask_dir[index]).convert("RGB")

        # Apply image transformations
        img = self.img_transform(img)

        # Convert mask to class indices
        mask = self.rgb_to_label(mask)

        # Data augmentation (spatial transformations)
        if random.random() < 0.5:
            img = T.functional.hflip(img)
            mask = T.functional.hflip(mask)

        if random.random() < 0.5:
            img = T.functional.vflip(img)
            mask = T.functional.vflip(mask)

        img_h, img_w = img.shape[1], img.shape[2]

        # Ensure the image is large enough for the patch size
        if img_h < self.patch_size or img_w < self.patch_size:
            raise ValueError(f"Image size ({img_h}, {img_w}) is smaller than patch size {self.patch_size}")

        # Random cropping
        start_y = random.randint(0, img_h - self.patch_size)
        start_x = random.randint(0, img_w - self.patch_size)
        img = img[:, start_y:start_y + self.patch_size, start_x:start_x + self.patch_size]
        mask = mask[start_y:start_y + self.patch_size, start_x:start_x + self.patch_size]

        return img, mask

def load_dataset(root_dir: str, mode: str = "train", patch_size: int = 256, batch_size: int = 16) -> DataLoader:
    """
    Loads the segmentation dataset for the specified mode.

    Args:
        root_dir (str): The root directory containing the dataset.
        mode (str, optional): The dataset split to load ('train' or 'val'). Defaults to 'train'.
        patch_size (int, optional): The size of the patches to extract. Defaults to 256.
        batch_size (int, optional): The batch size for the DataLoader. Defaults to 16.

    Returns:
        DataLoader: A DataLoader for the specified dataset split.
    """
    dataset = SegmentationDataset(root_dir=root_dir, mode=mode, patch_size=patch_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count())