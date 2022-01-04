from models import build
from loguru import logger
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optmi
import torch.nn.functional as F
from utils.tools import mean_dice
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


np.seterr(divide='ignore',invalid='ignore')

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
    print(f" predicting {data_set_name[0]} ")
    val_ds = Datareader(config['dataset']['test_CVC-300_img'], config['dataset']['test_CVC-300_label'], crop_size)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    with torch.no_grad():
        for idx, (img, label) in tqdm(enumerate(val_loader)):
            img = img.to(device)
            label = label.to(device)
            x = model(img)
            pred = F.softmax(x, dim=1)
            pred = torch.argmax(pred, dim=1)
            pred = torch.where(pred == 1, 255, 0)

            # convert pred into 3-dim statr
            pred_ = torch.cat((pred, pred), dim=1)
            pred_ = torch.cat((pred_, pred), dim=1)
            pred_ = pred_.reshape((8, 3, 352, 352))
            # convert pred into 3-dim end

            # convert label into 3-dim statr
            label_ = torch.cat((label, label), dim=1)
            label_ = torch.cat((label_, label), dim=1)
            label_ = label_.reshape((8, 3, 352, 352))
            # convert label into 3-dim end        

            break
        out = torch.cat((pred_.float(), label_.float()), dim=0)
        out = torch.cat((out, img), dim=0)
        # out = out.unsqueeze(1)
        save_image(out, "./predict_results/cvc-300.png")

# CVC-ClinicDB
if int(num_of_dataset) == 1:
    print(f" predicting {data_set_name[1]} ")
    val_ds = Datareader(config['dataset']['test_CVC-ClinicDB_img'], config['dataset']['test_CVC-ClinicDB_label'], crop_size)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    with torch.no_grad():
        for idx, (img, label) in tqdm(enumerate(val_loader)):
            img = img.to(device)
            label = label.to(device)
            x = model(img)
            pred = F.softmax(x, dim=1)
            pred = torch.argmax(pred, dim=1)
            pred = torch.where(pred == 1, 255, 0)

            # convert pred into 3-dim statr
            pred_ = torch.cat((pred, pred), dim=1)
            pred_ = torch.cat((pred_, pred), dim=1)
            pred_ = pred_.reshape((8, 3, 352, 352))
            # convert pred into 3-dim end

            # convert label into 3-dim statr
            label_ = torch.cat((label, label), dim=1)
            label_ = torch.cat((label_, label), dim=1)
            label_ = label_.reshape((8, 3, 352, 352))
            # convert label into 3-dim end        

            break
        out = torch.cat((pred_.float(), label_.float()), dim=0)
        out = torch.cat((out, img), dim=0)
        # out = out.unsqueeze(1)
        save_image(out, "./predict_results/CVC-ClinicDB.png")

# CVC-ColonDB
if int(num_of_dataset) == 1:
    print(f" predicting {data_set_name[2]} ")
    val_ds = Datareader(config['dataset']['test_CVC-ColonDB_img'], config['dataset']['test_CVC-ColonDB_label'], crop_size)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)


    with torch.no_grad():
        for idx, (img, label) in tqdm(enumerate(val_loader)):
            img = img.to(device)
            label = label.to(device)
            x = model(img)
            pred = F.softmax(x, dim=1)
            pred = torch.argmax(pred, dim=1)
            pred = torch.where(pred == 1, 255, 0)

            # convert pred into 3-dim statr
            pred_ = torch.cat((pred, pred), dim=1)
            pred_ = torch.cat((pred_, pred), dim=1)
            pred_ = pred_.reshape((8, 3, 352, 352))
            # convert pred into 3-dim end

            # convert label into 3-dim statr
            label_ = torch.cat((label, label), dim=1)
            label_ = torch.cat((label_, label), dim=1)
            label_ = label_.reshape((8, 3, 352, 352))
            # convert label into 3-dim end        

            break
        out = torch.cat((pred_.float(), label_.float()), dim=0)
        out = torch.cat((out, img), dim=0)
        # out = out.unsqueeze(1)
        save_image(out, "./predict_results/CVC-ColonDB.png")

# ETIS-LaribPolypDB
if int(num_of_dataset) == 1:
    print(f" predicting {data_set_name[3]} ")
    val_ds = Datareader(config['dataset']['test_ETIS-LaribPolypDB_img'], config['dataset']['test_ETIS-LaribPolypDB_label'], crop_size)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    with torch.no_grad():
        for idx, (img, label) in tqdm(enumerate(val_loader)):
            img = img.to(device)
            label = label.to(device)
            x = model(img)
            pred = F.softmax(x, dim=1)
            pred = torch.argmax(pred, dim=1)
            pred = torch.where(pred == 1, 255, 0)

            # convert pred into 3-dim statr
            pred_ = torch.cat((pred, pred), dim=1)
            pred_ = torch.cat((pred_, pred), dim=1)
            pred_ = pred_.reshape((8, 3, 352, 352))
            # convert pred into 3-dim end

            # convert label into 3-dim statr
            label_ = torch.cat((label, label), dim=1)
            label_ = torch.cat((label_, label), dim=1)
            label_ = label_.reshape((8, 3, 352, 352))
            # convert label into 3-dim end        

            break
        out = torch.cat((pred_.float(), label_.float()), dim=0)
        out = torch.cat((out, img), dim=0)
        # out = out.unsqueeze(1)
        save_image(out, "./predict_results/ETIS-LaribPolypDB.png")

# Kvasir
if int(num_of_dataset) == 1:
    print(f" predicting {data_set_name[4]} ")
    val_ds = Datareader(config['dataset']['test_Kvasir_img'], config['dataset']['test_Kvasir_label'], crop_size)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    with torch.no_grad():
        for idx, (img, label) in tqdm(enumerate(val_loader)):
            img = img.to(device)
            label = label.to(device)
            x = model(img)
            pred = F.softmax(x, dim=1)
            pred = torch.argmax(pred, dim=1)
            pred = torch.where(pred == 1, 255, 0)

            # convert pred into 3-dim statr
            pred_ = torch.cat((pred, pred), dim=1)
            pred_ = torch.cat((pred_, pred), dim=1)
            pred_ = pred_.reshape((8, 3, 352, 352))
            # convert pred into 3-dim end

            # convert label into 3-dim statr
            label_ = torch.cat((label, label), dim=1)
            label_ = torch.cat((label_, label), dim=1)
            label_ = label_.reshape((8, 3, 352, 352))
            # convert label into 3-dim end        

            break
        out = torch.cat((pred_.float(), label_.float()), dim=0)
        out = torch.cat((out, img), dim=0)
        # out = out.unsqueeze(1)
        save_image(out, "./predict_results/Kvasir.png")