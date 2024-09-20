from model import UNet
from dataset import load_dataset, DataLoader
from utils.early_stop import EarlyStopMechanism
from utils.log_writer import LOGWRITER

import torch
import torch.nn as nn
import argparse
import json
import os
from torch import optim as opt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def class_size(file_path: str) -> int:
    """
    Reads the number of classes from a JSON file.

    Args:
        file_path: Path to the JSON file containing class information.

    Returns:
        The number of classes.
    """
    with open(file_path, 'r') as file:
        class_json = json.load(file)
    return len(class_json['classes'])

def denormalize_image(img: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Denormalizes an image given its mean and standard deviation.

    Args:
        img: The normalized image (numpy array).
        mean: Mean values for each channel.
        std: Standard deviation values for each channel.

    Returns:
        The denormalized image (numpy array).
    """
    img_denorm = img * std[:, None, None] + mean[:, None, None]
    return img_denorm

def Segmentation(model: nn.Module, 
                 optimizer: opt.Optimizer, 
                 scheduler: opt.lr_scheduler._LRScheduler, 
                 train_dl: DataLoader,
                 valid_dl: DataLoader, 
                 total_epochs: int, 
                 output_dir: str, 
                 device: str, 
                 logger: LOGWRITER, 
                 writer: SummaryWriter, 
                 es_mech: EarlyStopMechanism
                 ) -> None:
    """
    Trains and evaluates a segmentation model.

    Args:
        model: The segmentation model (e.g., UNet).
        optimizer: The optimizer for training.
        scheduler: The learning rate scheduler.
        train_dl: The training dataloader.
        valid_dl: The validation dataloader.
        total_epochs: The total number of epochs to train for.
        output_dir: The directory to save outputs (logs, weights, visualizations).
        device: The device to use for training (e.g., 'cuda' or 'cpu').
        logger: The logger for recording training information.
        writer: The TensorBoard SummaryWriter for logging metrics.
        es_mech: The early stopping mechanism.
    """
    criterion = nn.CrossEntropyLoss()
    logger.write("[INFO] CrossEntropyLoss function instantiated.")

    mean = [0.485, 0.456, 0.406]  
    std = [0.229, 0.224, 0.225]

    logger.write(f"[INFO] Total Epochs: {total_epochs}")
    logger.write(f"[INFO] Training Dataloader loaded with {len(train_dl)} batches.")
    logger.write(f"[INFO] Validation Dataloader loaded with {len(valid_dl)} batches.")

    for epoch in range(total_epochs):
        model.train()
        total_tr_loss = 0.0

        # Training Loop
        for i, data in enumerate(tqdm(train_dl, desc=f"[Training SegModel] [{epoch+1}/{total_epochs}]")):
            img, mask = data
            img, mask = img.to(device), mask.to(device)

            optimizer.zero_grad()
            prediction = model(img)

            loss = criterion(prediction, mask)
            loss.backward()
            optimizer.step()

            total_tr_loss += loss.item()

        model.eval()
        total_val_loss = 0.0
        
        # Validation Loop
        with torch.no_grad():
            for i, data in enumerate(tqdm(valid_dl, desc=f"[Validating SegModel] [{epoch+1}/{total_epochs}]")):
                img, mask = data
                img, mask = img.to(device), mask.to(device)

                prediction = model(img)

                loss = criterion(prediction, mask)
                total_val_loss += loss.item()

        scheduler.step()
        logger.write(f"[INFO] Scheduler step at epoch {epoch+1}.")

        avg_tr_loss = total_tr_loss / len(train_dl)
        avg_val_loss = total_val_loss / len(valid_dl)

        # Early stopping mechanism
        es_mech.step(model=model, metric=avg_val_loss)
        if es_mech.check():
            logger.write("[INFO] Early Stopping Mechanism Engaged. Training procedure ended early.")
            break

        # Logging results
        logger.log_results(epoch=epoch+1,
                           tr_loss=avg_tr_loss,
                           val_loss=avg_val_loss)

        writer.add_scalar('Loss/Train', avg_tr_loss, epoch+1)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch+1)

    print("[INFO] Segmentation Training Job complete")
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True, help="Directory to dataset for image segmentation")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to store outputs logs and weights")
    parser.add_argument("--epochs", type=int, default=100, help="Total epochs to train model with")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument("--eta_min", type=float, default=1e-6, help="Minimum learning rate for Cosine Decay Scheduler")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger = LOGWRITER(output_directory=args.output_dir, total_epochs=args.epochs)
    logger.write("[INFO] Logger instantiated.")

    train_dataset = load_dataset(root_dir=args.root_dir, mode="train", patch_size=256)
    logger.write(f"[INFO] Training dataset loaded with {len(train_dataset)} samples.")
    valid_dataset = load_dataset(root_dir=args.root_dir, mode="val", patch_size=384)
    logger.write(f"[INFO] Validation dataset loaded with {len(valid_dataset)} samples.")

    model = UNet(class_size(os.path.join(args.root_dir, "classes.json"))).to(device)
    logger.write("[INFO] UNet model instantiated.")

    optimizer = opt.AdamW(params=model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-3)
    logger.write("[INFO] Optimizer instantiated.")

    scheduler = opt.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs, eta_min=args.eta_min, verbose=True)
    logger.write("[INFO] Scheduler instantiated.")

    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "log_outputs"))
    logger.write("[INFO] TensorBoard SummaryWriter instantiated.")

    es_mech = EarlyStopMechanism(metric_threshold=0.015, mode='min', grace_threshold=5, save_path=os.path.join(args.output_dir, "saved_weights"))
    logger.write("[INFO] EarlyStopMechanism instantiated.")

    Segmentation(model=model,
                 optimizer=optimizer,
                 scheduler=scheduler,
                 train_dl=train_dataset,
                 valid_dl=valid_dataset,
                 total_epochs=args.epochs,
                 output_dir=args.output_dir,
                 device=device,
                 logger=logger,
                 writer=writer,
                 es_mech=es_mech)
