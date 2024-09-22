import os
from glob import glob
import argparse
import random
import shutil
from PIL import Image
from tqdm import tqdm

def generate_patches(image_path: str, mask_path: str, patch_size: int, stride: int) -> list[tuple[Image.Image, Image.Image]]:
    # Open the image and mask
    image = Image.open(image_path)
    mask = Image.open(mask_path)

    width, height = image.size

    patches = []

    # Loop over the image with the given stride
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            # Crop the patch from image and mask
            image_patch = image.crop((x, y, x + patch_size, y + patch_size))
            mask_patch = mask.crop((x, y, x + patch_size, y + patch_size))

            patches.append((image_patch, mask_patch))

    return patches

def copy_files(image_list: list[str], mask_list: list[str], split: str, counters: dict[str, int], output_dir: str, patch_flag: bool = False, patch_size: int = 256, stride: int = 256):
    for img, msk in zip(image_list, mask_list):
        if patch_flag:
            # Generate patches
            patches = generate_patches(img, msk, patch_size, stride)
            for img_patch, msk_patch in patches:
                new_filename = f"{counters[split]:06d}"
                counters[split] += 1

                img_patch.save(os.path.join(output_dir, split, "images", f"{new_filename}.png"))
                msk_patch.save(os.path.join(output_dir, split, "masks", f"{new_filename}.png"))
        else:
            new_filename = f"{counters[split]:06d}"
            counters[split] += 1

            shutil.copy(img, os.path.join(output_dir, split, "images", f"{new_filename}.png"))
            shutil.copy(msk, os.path.join(output_dir, split, "masks", f"{new_filename}.png"))

def main(root_dir: str, output_dir: str, train_threshold: float, validation_threshold: float, test_threshold: float, patch_flag: bool = False, patch_size: int = 256, stride: int = 256):
    data = glob(os.path.join(root_dir, "*"))
        
    for split in ["train", "val", "test"]: 
        for labels in ["images", "masks"]: 
            path = os.path.join(output_dir, split, labels)
            os.makedirs(path, exist_ok=True)

    counters = {"train": 0, "val": 0, "test": 0}

    for dir in tqdm(data):
        if os.path.isdir(dir): 
            images = glob(os.path.join(dir, "images/*"))
            masks = glob(os.path.join(dir, "masks/*"))

            assert len(images) == len(masks), f"Number of images and masks don't match in {dir}"

            combined = list(zip(images, masks))
            random.shuffle(combined)
            images, masks = zip(*combined)

            total = len(images)
            train_idx = int(train_threshold * total)
            val_idx = int(validation_threshold * total) + train_idx

            train_images, val_images, test_images = images[:train_idx], images[train_idx:val_idx], images[val_idx:]
            train_masks, val_masks, test_masks = masks[:train_idx], masks[train_idx:val_idx], masks[val_idx:]

            copy_files(train_images, train_masks, "train", counters, output_dir, patch_flag, patch_size, stride)
            copy_files(val_images, val_masks, "val", counters, output_dir, patch_flag, patch_size, stride)
            copy_files(test_images, test_masks, "test", counters, output_dir, patch_flag, patch_size, stride)

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Directory to segmentation dataset")
    parser.add_argument("--output_dir", type=str, default="UNetData", help="Directory to output organized dataset")
    parser.add_argument("--threshold", type=float, nargs=3, default=[0.7, 0.2, 0.1], help="Threshold values for splitting dataset into train, validation, and test sets")
    parser.add_argument("--patch", action='store_true', help="Enable patch generation")
    parser.add_argument("--patch_size", type=int, default=256, help="Size of the patches")
    parser.add_argument("--stride", type=int, default=256, help="Stride for patch generation")

    args = parser.parse_args()
        
    main(args.input_dir, args.output_dir, args.threshold[0], args.threshold[1], args.threshold[2],
         args.patch, args.patch_size, args.stride)
