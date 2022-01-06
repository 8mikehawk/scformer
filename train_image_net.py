import models
from models.pvt.pretrain_test import mit_b2
from utils.load_imageNet import ImageNetLoader
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optmi
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from loguru import logger

device = 'cuda'

model = mit_b2()

model = model.to(device)

model.load_state_dict(torch.load("/data/segformer/scformer/train_package/imageNet_pretrain/train.pth"))

batch_size = 256
num_works = 8
lr = 1e-4

logger.add("/data/segformer/scformer/train_package/imageNet_pretrain/imageNet.log")

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optmi.AdamW(model.parameters(), lr=lr)

train_ds = ImageNetLoader("/data/imageNet/train_.txt", "/data/imageNet/val_.txt")

train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_works, shuffle=True)


val_ds = ImageNetLoader("/data/imageNet/train_.txt", "/data/imageNet/val_.txt", mode="val")
val_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_works, shuffle=True)

best_loss = [100000000]

logger.info(f"| start training .... |")
for epoch in range(100000000000000):
    loss = 0
    for idx, (img, label) in tqdm(enumerate(train_loader)):
        img = img.to(device)
        label = label.to(device)

        out = model(img)
        loss = criterion(out, label)
        loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
        if idx % 20 == 0 and idx != 0:
            # print(loss.item() / idx)
            torch.save(model.state_dict(), "/data/segformer/scformer/train_package/imageNet_pretrain/train.pth")
            logger.info(f"| Epoch {epoch} | batch {idx} | loss : {loss.item() / idx}|")
    # logger.critical(f"| Epoch {epoch} | batch {idx} | loss : {loss.item() / idx}|")
    # logger.critical(f"| Epoch {epoch} | batch {idx} | loss : {loss.item() / idx} |")
    # torch.save(model.state_dict(), "/data/segformer/scformer/train_package/imageNet_pretrain/train.pth")
    if (loss / idx) < min(best_loss):
        best_loss.append((loss / idx))
        logger.critical(f"| Epoch {epoch} | best training loss : {min(best_loss)} |")
        torch.save(model.state_dict(), "/data/segformer/scformer/train_package/imageNet_pretrain/train_best.pth")