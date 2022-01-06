from models import build
from loguru import logger
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optmi
import torch.nn.functional as F
from utils.tools import mean_dice
from utils import DataBuilder, Datareader, CustomDataSet
from utils.loss import *
from torch.utils.data import DataLoader
from torchvision.transforms import Compose 
from torchvision import transforms
import torch
import os
import sys
import numpy as np
import yaml

np.seterr(divide='ignore',invalid='ignore')


f = open(sys.argv[1])
config = yaml.safe_load(f)

evl_epoch = config['training']['evl_epoch']


# 定义模型
device = config['training']['device']
model = build(model_name=config['model']['model_name'], class_num=config['dataset']['class_num'])
model.to(device)

train_img_root = config['dataset']['train_img_root']
train_label_root = config['dataset']['train_label_root']
crop_size = (
    config['dataset']['crop_size']['w'],
    config['dataset']['crop_size']['h']
)
batch_size = config['dataset']['batch_size']
num_workers = config['dataset']['num_workers']
checkpoint_save_path = config['other']['checkpoint_save_path']

# training
max_epoch = config['training']['max_epoch']
lr = float(
    config['training']['lr']
)

train_ds = CustomDataSet(train_img_root, train_label_root, crop_size)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

# optimizer
# criterion = nn.NLLLoss().to(device)
criterion = DiceLoss().to(device)
optimizer = optmi.AdamW(model.parameters(), lr=lr)

# logger
print(config['other']['logger_path'])
logger.add(config['other']['logger_path'])

# start training
logger.info(f"| start training .... | current model {config['model']['model_name']} |")
best_val_dice = [0]

for epoch in tqdm(range(max_epoch)):
    train_loss = 0
    for idx, (img, label) in tqdm(enumerate(train_loader)):
        model = model.train()
        img = img.to(device)
        label = label.to(device)
        out = model(img)
        out = F.log_softmax(out, dim=1)
        
        loss = criterion(out, label)

        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("train epoch done")
    logger.info(f"| epoch : {epoch} | training done | loss: {train_loss / (idx + 1)} |")

    if epoch >= evl_epoch:

        val_cvc_300 = 0
        val_cvc_clinicDB = 0
        val_cvc_colonDB = 0
        val_etis = 0
        val_Kvasir = 0
        print("evaluating cvc-300")
        # cvc
        val_ds = Datareader(config['dataset']['test_CVC-300_img'], config['dataset']['test_CVC-300_label'], crop_size)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        # evaluate cvc
        val_dice = 0
        with torch.no_grad():
            for idx, (img, label) in tqdm(enumerate(val_loader)):
                img = img.to(device)
                label = label.to(device)
                x = model(img)
                pred = F.softmax(x, dim=1)
                # print(pred.shape, img.shape)
                pre_label = pred.max(dim=1)[1].data.cpu().numpy()
                pre_label = [i for i in pre_label]
                true_label = label.data.cpu().numpy()
                true_label = [i for i in true_label]
                all_acc, acc, dice = mean_dice(pre_label, true_label, num_classes = config['dataset']['class_num'], ignore_index = None)
                val_dice = dice[1] + val_dice
            val_cvc_300 = val_dice/(idx+1)

        print("evaluating CVC-ClinicDB")
        # CVC-ClinicDB
        val_ds = Datareader(config['dataset']['test_CVC-ClinicDB_img'], config['dataset']['test_CVC-ClinicDB_label'], crop_size)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        # evaluate CVC-ClinicDB
        val_dice = 0
        with torch.no_grad():
            for idx, (img, label) in tqdm(enumerate(val_loader)):
                img = img.to(device)
                label = label.to(device)
                x = model(img)
                pred = F.softmax(x, dim=1)
                # print(pred.shape, img.shape)
                pre_label = pred.max(dim=1)[1].data.cpu().numpy()
                pre_label = [i for i in pre_label]
                true_label = label.data.cpu().numpy()
                true_label = [i for i in true_label]
                all_acc, acc, dice = mean_dice(pre_label, true_label, num_classes = config['dataset']['class_num'], ignore_index = None)
                val_dice = dice[1] + val_dice
            val_cvc_clinicDB = val_dice/(idx+1)

        print("CVC-ColonDB")
        # CVC-ColonDB
        val_ds = Datareader(config['dataset']['test_CVC-ColonDB_img'], config['dataset']['test_CVC-ColonDB_label'], crop_size)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        # evaluate CVC-ColonDB
        val_dice = 0
        with torch.no_grad():
            for idx, (img, label) in tqdm(enumerate(val_loader)):
                img = img.to(device)
                label = label.to(device)
                x = model(img)
                pred = F.softmax(x, dim=1)
                # print(pred.shape, img.shape)
                pre_label = pred.max(dim=1)[1].data.cpu().numpy()
                pre_label = [i for i in pre_label]
                true_label = label.data.cpu().numpy()
                true_label = [i for i in true_label]
                all_acc, acc, dice = mean_dice(pre_label, true_label, num_classes = config['dataset']['class_num'], ignore_index = None)
                val_dice = dice[1] + val_dice
            val_cvc_colonDB = val_dice/(idx+1)

        print("evaluating ETIS-LaribPolypDB")

        # ETIS-LaribPolypDB
        val_ds = Datareader(config['dataset']['test_ETIS-LaribPolypDB_img'], config['dataset']['test_ETIS-LaribPolypDB_label'], crop_size)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        # evaluate ETIS-LaribPolypDB
        val_dice = 0
        with torch.no_grad():
            for idx, (img, label) in tqdm(enumerate(val_loader)):
                img = img.to(device)
                label = label.to(device)
                x = model(img)
                pred = F.softmax(x, dim=1)
                # print(pred.shape, img.shape)
                pre_label = pred.max(dim=1)[1].data.cpu().numpy()
                pre_label = [i for i in pre_label]
                true_label = label.data.cpu().numpy()
                true_label = [i for i in true_label]
                all_acc, acc, dice = mean_dice(pre_label, true_label, num_classes = config['dataset']['class_num'], ignore_index = None)
                val_dice = dice[1] + val_dice
            val_etis = val_dice/(idx+1)

        print("evaluating Kvasir")
        # Kvasir
        val_ds = Datareader(config['dataset']['test_Kvasir_img'], config['dataset']['test_Kvasir_label'], crop_size)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        # evaluate Kvasir
        val_dice = 0
        with torch.no_grad():
            for idx, (img, label) in tqdm(enumerate(val_loader)):
                img = img.to(device)
                label = label.to(device)
                x = model(img)
                pred = F.softmax(x, dim=1)
                # print(pred.shape, img.shape)
                pre_label = pred.max(dim=1)[1].data.cpu().numpy()
                pre_label = [i for i in pre_label]
                true_label = label.data.cpu().numpy()
                true_label = [i for i in true_label]
                all_acc, acc, dice = mean_dice(pre_label, true_label, num_classes = config['dataset']['class_num'], ignore_index = None)
                val_dice = dice[1] + val_dice
            val_Kvasir = val_dice/(idx+1)

        mean_total = (val_cvc_300 + val_cvc_clinicDB + val_cvc_colonDB + val_etis + val_Kvasir) / 5
        # mean_total = 0.2 * val_cvc_300 + 0.2 * val_cvc_clinicDB + 0.2 * val_cvc_colonDB + 0.2 * val_etis + val_Kvasir
        # logger.info(f"| epoch : {epoch} | CVC-300 : {val_cvc_300} | CVC-ClinicDB : {val_cvc_clinicDB} | CVC-ColonDB : {val_cvc_colonDB} | ETIS-LaribPolypDB : {val_etis} | Kvasir : {val_Kvasir} |")
        if max(best_val_dice) <=  mean_total:
            best_val_dice.append(mean_total)
            print('best_val_dice_score :{:}'.format(max(best_val_dice)))
            logger.info(f"| epoch : {epoch} | CVC-300 : {val_cvc_300} | CVC-ClinicDB : {val_cvc_clinicDB} | CVC-ColonDB : {val_cvc_colonDB} | ETIS-LaribPolypDB : {val_etis} | Kvasir : {val_Kvasir} |")
            logger.critical(f"| epoch : {epoch} | best_val_dice_score : {max(best_val_dice)} |")
            torch.save(model.state_dict(), os.path.join(checkpoint_save_path, "val_best.pth"))
        else:
            logger.info(f"| epoch : {epoch} | val done |")