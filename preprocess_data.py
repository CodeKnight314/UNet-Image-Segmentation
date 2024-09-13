import os
from glob import glob
import argparse
import random
import shutil

def main(root_dir, output_dir, train_threshold, validation_threshold, test_threshold):
    data = glob(os.path.join(root_dir, "*"))
    
    for split in ["train", "val", "test"]: 
        for labels in ["images", "masks"]: 
            path = os.path.join(output_dir, split, labels)
            os.makedirs(path, exist_ok=True)

    counters = {"train": 0, "val": 0, "test": 0}

    for dir in data:
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

            def copy_files(image_list, mask_list, split):
                for img, msk in zip(image_list, mask_list):
                    new_filename = f"{counters[split]:04d}"
                    counters[split] += 1  

                    shutil.copy(img, os.path.join(output_dir, split, "images", f"{new_filename}.png"))
                    shutil.copy(msk, os.path.join(output_dir, split, "masks", f"{new_filename}.png"))

            copy_files(train_images, train_masks, "train")
            copy_files(val_images, val_masks, "val")
            copy_files(test_images, test_masks, "test")

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Directory to segmentation dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to output organized dataset")
    parser.add_argument("--threshold", type=float, nargs=3, default=[0.7, 0.2, 0.1], help="Threshold values for splitting dataset into train, validation, and test sets")

    args = parser.parse_args()
    
    main(args.input_dir, args.output_dir, args.threshold[0], args.threshold[1], args.threshold[2])
