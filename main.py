from model import UNet 
from loss import DiceLoss, FocalLoss
from dataset import load_dataset
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

def class_size(file_path): 
    with open(file_path, 'r') as file: 
        class_json = json.load(file)
    return len(class_json['classes'])

def Segmentation(model, optimizer, scheduler, train_dl, valid_dl, total_epochs, output_dir, device): 
    criterion_diceLoss = DiceLoss(smoothing=1e-6)
    criterion_focalLoss = FocalLoss(gamma=2, reduction="mean")
    
    writer = SummaryWriter(log_dir=os.path.join(output_dir, "log_outputs"))
    es_mech = EarlyStopMechanism(metric_threshold=0.015, mode='min', grace_threshold=5, save_path=os.path.join(output_dir, "saved_weights"))
    logger = LOGWRITER(output_directory=output_dir, total_epochs=total_epochs)
    
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
            
            diceLoss_value = criterion_diceLoss(prediction, mask)
            focalLoss = criterion_focalLoss(prediction, mask)
            
            loss = diceLoss_value + focalLoss
            loss.backward() 
            optimizer.step()
            
            total_tr_loss += loss.item()
        
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(tqdm(valid_dl, desc=f"[Validating SegModel] [{epoch+1}/{total_epochs}]")):
                img, mask = data
                img, mask = img.to(device), mask.to(device) 
                
                prediction = model(img)
                
                diceLoss_value = criterion_diceLoss(prediction, mask)
                focalLoss = criterion_focalLoss(prediction, mask)
                
                loss = diceLoss_value + focalLoss
                total_val_loss += loss.item()

        scheduler.step()
        
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
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to store outpus logs and weights")
    parser.add_argument("--epochs", type=int, default=100, help="Total epochs to train model with")
    parser.add_argument("--lr", type=float, default=1e-4, help="lr rate for optimizer")
    parser.add_argument("--eta_min", type=float, default=1e-6, help="minimum lr for Cosine Decay Scheduler")
    
    args = parser.parse_args() 
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    train_dataset = load_dataset(root_dir=args.root_dir, mode="train", patch_size=256)
    valid_dataset = load_dataset(root_dir=args.root_dir, mode="val", patch_size=384)

    model = UNet(class_size(os.path.join(args.root_dir, "classes.json"))).to(device)
    
    optimizer = opt.AdamW(params=model.parameters, lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-3)
    scheduler = opt.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs, eta_min=args.eta_min, verbose=True)    

    Segmentation(model=model, 
                 optimizer=optimizer, 
                 scheduler=scheduler,
                 train_dl=train_dataset,
                 valid_dl=valid_dataset,
                 total_epochs=args.epochs,
                 output_dir=args.output_dir, 
                 device=device)
    
    

    