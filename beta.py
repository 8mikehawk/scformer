import models
from utils import ISIC2018, cal_dice
from torch.utils.data import DataLoader
from tqdm import tqdm
from models import sct_b1
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optmi
import os
from loguru import logger
import numpy as np


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
model = sct_b1(class_num=class_num)
model = model.to(device)

# load pretrained 
# model.load_state_dict(torch.load("models/checkpoints/train_best.pth"))
# model.load_state_dict(torch.load("/mnt/DATA-1/DATA-2/Feilong/myself/cnn_model/best_train_miou_Seg.pth"))

# training config
lr = 1e-4
max_epoch = 10000
checkpoint_save_path = "./models/checkpoints/"

# logger
logger.add("models/checkpoints/sct_b1.log")
####################


# load datasets
train_ds = ISIC2018(train_img_root, val_img_root, train_label_root, val_label_root, crop_size, mode='train')
val_ds = ISIC2018(train_img_root, val_img_root, train_label_root, val_label_root, crop_size, mode='val')

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

criterion = nn.NLLLoss().to(device)
optimizer = optmi.AdamW(model.parameters(), lr=lr)

# best = [0]
# best_val_dice = [0]

# training 
logger.info("Start training ...")
# for epoch in tqdm(range(max_epoch)):
#     train_dice_score = []
#     eval_dice_score = []
    
#     model = model.train()
#     for idx, (img, label) in tqdm(enumerate(train_loader)):
    
#         img = img.to(device)
#         label = label.to(device)
#         out = model(img)
#         out_ = F.softmax(out, dim=1)
#         out = F.log_softmax(out, dim=1)

#         loss = criterion(out, label)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         train_dice_score.append(cal_dice(out_, label))
#     if max(best) <= (np.mean(train_dice_score)):
#         best.append(np.mean(train_dice_score))
#         torch.save(model.state_dict(), os.path.join(checkpoint_save_path, "train_best.pth"))

#     with torch.no_grad():
#         net = model
#         for idx, (img, label) in enumerate(val_loader):
#             img = img.to(device)
#             label = label.to(device)
#             out = model(img)
#             out = F.softmax(out, dim=1)
#             eval_dice_score.append(cal_dice(out, label))
#         print(np.mean(eval_dice_score))    
#         if max(best_val_dice) <= (np.mean(eval_dice_score)):
#             best_val_dice.append(np.mean(eval_dice_score))
#             torch.save(model.state_dict(), os.path.join(checkpoint_save_path, "val_best.pth"))
#             logger.critical(f"| epoch: {epoch} | best val mDice : {max(best_val_dice)} |") 
#     logger.info(f"| epoch {epoch} | training mDice : {np.mean(train_dice_score)} | val mDice : {np.mean(eval_dice_score)} |")