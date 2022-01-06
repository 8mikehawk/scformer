import io
from models import build
from loguru import logger
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optmi
import torch.nn.functional as F
from utils.tools import mean_dice, mean_iou
from utils import DataBuilder, Datareader
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from torchvision import transforms
from torchvision.utils import save_image
import torch
import os
import sys
import numpy as np
import yaml

np.seterr(divide='ignore', invalid='ignore')

data_set_name = [
    'cvc-300',
    'CVC-ClinicDB',
    'CVC-ColonDB',
    'ETIS-LaribPolypDB',
    'Kvasir'
]

f = open(sys.argv[1])
num_of_dataset = 1
config = yaml.safe_load(f)

device = config['training']['device']
model = build(model_name=config['model']['model_name'], class_num=config['dataset']['class_num'])
if device == "cpu":
    model.load_state_dict(torch.load(config['test']['checkpoint_save_path']), map_location=torch.device('cpu'))
else:
    model.load_state_dict(torch.load(config['test']['checkpoint_save_path']))
model = model.to(device)

train_img_root = config['dataset']['train_img_root']
train_label_root = config['dataset']['train_label_root']
crop_size = (
    config['dataset']['crop_size']['w'],
    config['dataset']['crop_size']['h']
)

# batch size !!!!
batch_size = 8
num_workers = config['dataset']['num_workers']
checkpoint_save_path = config['other']['checkpoint_save_path']

# training
max_epoch = config['training']['max_epoch']
lr = float(
    config['training']['lr']
)

# cvc-300
if int(num_of_dataset) == 1:
    print(f" predicting {data_set_name[0]}")
    val_ds = Datareader(config['dataset']['test_CVC-300_img'], config['dataset']['test_CVC-300_label'], crop_size)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_cvc_300 = 0
    miou_cvc_300 = 0
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
            all_acc, acc, dice = mean_dice(pre_label, true_label, num_classes=config['dataset']['class_num'],
                                           ignore_index=None)
            all_acc, acc, iou = mean_iou(pre_label, true_label, num_classes=config['dataset']['class_num'],
                                           ignore_index=None)
            val_cvc_300 = dice[1] + val_cvc_300
            miou_cvc_300 += iou[1]
        val_cvc_300 = val_cvc_300 / (idx + 1)
        miou_cvc_300 = miou_cvc_300 / (idx + 1)

# CVC-ClinicDB
if int(num_of_dataset) == 1:
    print(f" predicting {data_set_name[1]} ")
    val_ds = Datareader(config['dataset']['test_CVC-ClinicDB_img'], config['dataset']['test_CVC-ClinicDB_label'],
                        crop_size)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_CVC_ClinicDB = 0
    miou_CVC_ClinicDB = 0
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
            all_acc, acc, dice = mean_dice(pre_label, true_label, num_classes=config['dataset']['class_num'],
                                           ignore_index=None)
            all_acc, acc, iou = mean_iou(pre_label, true_label, num_classes=config['dataset']['class_num'],
                                           ignore_index=None)                                           
            val_CVC_ClinicDB = dice[1] + val_CVC_ClinicDB
            miou_CVC_ClinicDB += iou[1]
        val_CVC_ClinicDB = val_CVC_ClinicDB / (idx + 1)
        miou_CVC_ClinicDB = miou_CVC_ClinicDB / (idx + 1)

# CVC-ColonDB
if int(num_of_dataset) == 1:
    print(f" predicting {data_set_name[2]} ")
    val_ds = Datareader(config['dataset']['test_CVC-ColonDB_img'], config['dataset']['test_CVC-ColonDB_label'],
                        crop_size)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_CVC_ColonDB = 0
    miou_CVC_ColonDB = 0
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
            all_acc, acc, dice = mean_dice(pre_label, true_label, num_classes=config['dataset']['class_num'],
                                           ignore_index=None)
            all_acc, acc, iou = mean_iou(pre_label, true_label, num_classes=config['dataset']['class_num'],
                                           ignore_index=None)                                              
            val_CVC_ColonDB = dice[1] + val_CVC_ColonDB
            miou_CVC_ColonDB += iou[1] 
        val_CVC_ColonDB = val_CVC_ColonDB / (idx + 1)
        miou_CVC_ColonDB = miou_CVC_ColonDB / (idx + 1)

# ETIS-LaribPolypDB
if int(num_of_dataset) == 1:
    print(f" predicting {data_set_name[3]} ")
    val_ds = Datareader(config['dataset']['test_ETIS-LaribPolypDB_img'],
                        config['dataset']['test_ETIS-LaribPolypDB_label'], crop_size)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_ETIS_LaribPolypDB = 0
    miou_ETIS_LaribPolypDB = 0
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
            all_acc, acc, dice = mean_dice(pre_label, true_label, num_classes=config['dataset']['class_num'],
                                           ignore_index=None)
            all_acc, acc, iou = mean_iou(pre_label, true_label, num_classes=config['dataset']['class_num'],
                                           ignore_index=None)                                              
            val_ETIS_LaribPolypDB = dice[1] + val_ETIS_LaribPolypDB
            miou_ETIS_LaribPolypDB += iou[1]
        val_ETIS_LaribPolypDB = val_ETIS_LaribPolypDB / (idx + 1)
        miou_ETIS_LaribPolypDB = miou_ETIS_LaribPolypDB / (idx + 1)

# Kvasir
if int(num_of_dataset) == 1:
    print(f" predicting {data_set_name[4]} ")
    val_ds = Datareader(config['dataset']['test_Kvasir_img'], config['dataset']['test_Kvasir_label'], crop_size)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_Kvasir = 0
    miou_Kvasir = 0
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
            all_acc, acc, dice = mean_dice(pre_label, true_label, num_classes=config['dataset']['class_num'],
                                           ignore_index=None)
            all_acc, acc, iou = mean_iou(pre_label, true_label, num_classes=config['dataset']['class_num'],
                                           ignore_index=None)                                             
            val_Kvasir = dice[1] + val_Kvasir
            miou_Kvasir += iou[1]
        val_Kvasir = val_Kvasir / (idx + 1)
        miou_Kvasir = miou_Kvasir / (idx + 1)

print(f"| dice score | cvc-300 : {val_cvc_300} | val_CVC_ClinicDB : {val_CVC_ClinicDB} | val_CVC_ColonDB : {val_CVC_ColonDB} | val_ETIS_LaribPolypDB : {val_ETIS_LaribPolypDB} | val_Kvasir : {val_Kvasir} |")
print(f"| miou | cvc-300 : {miou_cvc_300} | val_CVC_ClinicDB : {miou_CVC_ClinicDB} | val_CVC_ColonDB : {miou_CVC_ColonDB} | val_ETIS_LaribPolypDB : {miou_ETIS_LaribPolypDB} | val_Kvasir : {miou_Kvasir} |")
