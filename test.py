import models
from utils import ISIC2018, Timer
from torch.utils.data import DataLoader
from tqdm import tqdm
from models import sct_b2
from utils.eval_semantic_segmentation import eval_semantic_segmentation
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optmi
import os
import pdb
from utils import Dice_score
import numpy as np
from tqdm import tqdm

####################
# dataset config
train_img_root = "/mnt/DATA-1/DATA-2/Feilong/scformer/data/ISIC2018/train/"
val_img_root = "/mnt/DATA-1/DATA-2/Feilong/scformer/data/ISIC2018/val/"
train_label_root = "/mnt/DATA-1/DATA-2/Feilong/scformer/data/ISIC2018/train_labels/"
val_label_root = "/mnt/DATA-1/DATA-2/Feilong/scformer/data/ISIC2018/val_labels/"
crop_size = (512, 512)
batch_size = 8
num_workers = 8

# gpu config
device = 'cuda'

# model config
class_num = 2
model = sct_b2(class_num=class_num)
model = model.to(device)

# load pretrained 
model.load_state_dict(torch.load("/mnt/DATA-1/DATA-2/Feilong/scformer/models/checkpoints/val_best.pth"))
model = model.to(device)
val_ds = ISIC2018(train_img_root, val_img_root, train_label_root, val_label_root, crop_size, mode='val')
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

dice_cal = Dice_score()

dice__= []
with torch.no_grad():
    for idx, (img, label) in tqdm(enumerate(val_loader)):
        img = img.to(device)
        label = label.to(device)
        x = model(img)
        pred = F.softmax(x, dim=1)
        m_dice = dice_cal.cal_dice(pred, label)
        dice__.append(m_dice)
        # break
        # pred = pred.reshape((batch_size, class_num, crop_size[0]*crop_size[1]))
        # pred = torch.argmax(pred, dim=1)
        # pred = torch.where(pred == 1, 255, 0)
        # pred = pred.reshape((batch_size, crop_size[0], crop_size[1]))
        # miou = iou_mean(pred.cpu().numpy(), label.cpu().numpy(), n_classes=1)
        # miou = dice_coeff(pred.cpu().numpy(), label.cpu().numpy())
        # miou_.append(miou)
print(np.mean(dice__))
