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
from loguru import logger


####################
# dataset config
train_img_root = "/mnt/DATA-1/DATA-2/Feilong/scformer/data/ISIC2018/train/"
val_img_root = "/mnt/DATA-1/DATA-2/Feilong/scformer/data/ISIC2018/val/"
train_label_root = "/mnt/DATA-1/DATA-2/Feilong/scformer/data/ISIC2018/train_labels/"
val_label_root = "/mnt/DATA-1/DATA-2/Feilong/scformer/data/ISIC2018/val_labels/"
class_dict_path = "data/ISIC2018/class_dict.csv"
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
model.load_state_dict(torch.load("models/checkpoints/train_best.pth"))

# training config
lr = 1e-4
max_epoch = 10000
checkpoint_save_path = "./models/checkpoints/"

# logger
logger.add("models/checkpoints/sct_b2.log")
# timmer
timer = Timer()
####################


# load datasets
train_ds = ISIC2018([train_img_root, train_label_root], crop_size, class_dict_path=class_dict_path)
val_ds = ISIC2018([val_img_root, val_label_root], crop_size, class_dict_path=class_dict_path)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

criterion = nn.NLLLoss().to(device)
optimizer = optmi.AdamW(model.parameters(), lr=lr)

best_val = [0]
best = [0]
best_val_dice = [0]

# training 
logger.info("Start training ...")
for epoch in tqdm(range(max_epoch)):
    train_loss = 0
    train_acc = 0
    train_miou = 0
    train_class_acc = 0
    model = model.train()
    for idx, (img, label) in tqdm(enumerate(train_loader)):
    
        img = img.to(device)
        label = label.to(device)
        out = model(img)
        out = F.log_softmax(out, dim=1)

        loss = criterion(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        pre_label = out.max(dim=1)[1].data.cpu().numpy()
             
        pre_label = [i for i in pre_label]
        true_label = label.data.cpu().numpy()
        true_label = [i for i in true_label]
        
        eval_metrix = eval_semantic_segmentation(pre_label, true_label, class_num)
        train_acc += eval_metrix['mean_class_accuracy']
        train_miou += eval_metrix['miou']
        train_class_acc += eval_metrix['class_accuracy']    

    metric_description = '|Train Acc|: {:.5f}|Train Mean IU|: {:.5f}\n|Train_class_acc|:{:}'.format(
        train_acc / len(train_loader),
        train_miou / len(train_loader),
        train_class_acc / len(train_loader))
    if max(best) <= train_miou / len(train_loader):
        best.append(train_miou / len(train_loader))
        torch.save(model.state_dict(), os.path.join(checkpoint_save_path, "train_best.pth"))

    # evaluation ---
    with torch.no_grad():
        net = model
        eval_loss = 0
        eval_acc = 0
        eval_miou = 0
        for idx, (img, label) in enumerate(val_loader):
            img = img.to(device)
            label = label.to(device)

            out = net(img)
            out = F.log_softmax(out, dim=1)
            loss = criterion(out, label)
            eval_loss = loss.item() + eval_loss
            pre_label = out.max(dim=1)[1].data.cpu().numpy()
            pre_label = [i for i in pre_label]

            true_label = label.data.cpu().numpy()
            true_label = [i for i in true_label]

            eval_metrics = eval_semantic_segmentation(pre_label, true_label, class_num)
            eval_acc = eval_metrics['mean_class_accuracy'] + eval_acc
            eval_miou = eval_metrics['miou'] + eval_miou

        if max(best_val) <= eval_miou / len(val_loader):
            best_val.append(eval_miou / len(val_loader))
            torch.save(model.state_dict(), os.path.join(checkpoint_save_path, "val_best.pth"))
            logger.critical(f"| epoch: {epoch} | best val mIou : {max(best_val)} |")
    remain_time = timer.get_remain_time(epoch, max_epoch)
    logger.info(f"| remain time: {remain_time['hour']} hour, {remain_time['min']} min, {remain_time['second']} second | epoch {epoch} | training mIou : {train_miou / len(train_loader)} | val mIou : {eval_miou / len(val_loader)} |")