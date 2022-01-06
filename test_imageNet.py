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
batch_size = 16
num_works = 8

model = mit_b2()

model = model.to(device)

checkpoint_path = "/data/segformer/scformer/train_package/imageNet_pretrain/train_best.pth"

if device != "cuda":
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
else:
    model.load_state_dict(torch.load(checkpoint_path))

val_ds = ImageNetLoader("/data/imageNet/train_.txt", "/data/imageNet/val_.txt", mode="val")
val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_works, shuffle=True)

progress = tqdm(enumerate(val_loader), total=val_ds.__len__())
with torch.no_grad():
    acc = 0
    for idx, (img, label) in enumerate(val_loader):
        img = img.to(device)
        label = label.to(device)
        out = model(img)
        out = F.softmax(out, dim=1)
        pred = torch.argmax(out, dim=1)
        acc += sum(pred == label)
        print(sum(pred == label))
        # if idx != 0:
        #     print(f"batch : {idx}, acc : {acc / idx}")
        #     progress.set_description("Accuracy: {:.4f}".format(acc / (idx * batch_size)))
        # progress.update(batch_size)

    # print(f"accuracy is : {acc / idx}")
