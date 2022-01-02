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
num_of_dataset = sys.argv[2]
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
if int(num_of_dataset) == 0:
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
            break
        out = torch.cat((pred.float(), label.float()), dim=0)
        out = out.unsqueeze(1)
        save_image(out, "result.png")

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
            print(pred.shape)
            break
        out = torch.cat((pred.float(), label.float()), dim=0)
        out = out.unsqueeze(1)
        print(out.shape)
        save_image(out, "result.png")

# CVC-ColonDB
if int(num_of_dataset) == 2:
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
            print(pred.shape)
            break
        out = torch.cat((pred.float(), label.float()), dim=0)
        out = out.unsqueeze(1)
        print(out.shape)
        save_image(out, "result.png")

# ETIS-LaribPolypDB
if int(num_of_dataset) == 3:
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
            print(pred.shape)
            break
        out = torch.cat((pred.float(), label.float()), dim=0)
        out = out.unsqueeze(1)
        print(out.shape)
        save_image(out, "result.png")

# Kvasir
if int(num_of_dataset) == 4:
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
            print(pred.shape)
            break
        out = torch.cat((pred.float(), label.float()), dim=0)
        out = out.unsqueeze(1)
        print(out.shape)
        save_image(out, "result.png")



# # CVC-ClinicDB
# val_ds = Datareader(config['dataset']['test_CVC-ClinicDB_img'], config['dataset']['test_CVC-ClinicDB_label'], crop_size)
# val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
# # evaluate CVC-ClinicDB
# val_dice = 0
# with torch.no_grad():
#     for idx, (img, label) in tqdm(enumerate(val_loader)):
#         img = img.to(device)
#         label = label.to(device)
#         x = model(img)
#         pred = F.softmax(x, dim=1)
#         # print(pred.shape, img.shape)
#         pre_label = pred.max(dim=1)[1].data.cpu().numpy()
#         pre_label = [i for i in pre_label]
#         true_label = label.data.cpu().numpy()
#         true_label = [i for i in true_label]
#         all_acc, acc, dice = mean_dice(pre_label, true_label, num_classes = config['dataset']['class_num'], ignore_index = None)
#         val_dice = dice + val_dice
#     val_cvc_clinicDB = val_dice.mean()/(idx+1)
# print("CVC-ColonDB")
# # CVC-ColonDB
# val_ds = Datareader(config['dataset']['test_CVC-ColonDB_img'], config['dataset']['test_CVC-ColonDB_label'], crop_size)
# val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
# # evaluate CVC-ColonDB
# val_dice = 0
# with torch.no_grad():
#     for idx, (img, label) in tqdm(enumerate(val_loader)):
#         img = img.to(device)
#         label = label.to(device)
#         x = model(img)
#         pred = F.softmax(x, dim=1)
#         # print(pred.shape, img.shape)
#         pre_label = pred.max(dim=1)[1].data.cpu().numpy()
#         pre_label = [i for i in pre_label]
#         true_label = label.data.cpu().numpy()
#         true_label = [i for i in true_label]
#         all_acc, acc, dice = mean_dice(pre_label, true_label, num_classes = config['dataset']['class_num'], ignore_index = None)
#         val_dice = dice + val_dice
#     val_cvc_colonDB = val_dice.mean()/(idx+1)
# print("evaluating ETIS-LaribPolypDB")
# # ETIS-LaribPolypDB
# val_ds = Datareader(config['dataset']['test_ETIS-LaribPolypDB_img'], config['dataset']['test_ETIS-LaribPolypDB_label'], crop_size)
# val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
# # evaluate ETIS-LaribPolypDB
# val_dice = 0
# with torch.no_grad():
#     for idx, (img, label) in tqdm(enumerate(val_loader)):
#         img = img.to(device)
#         label = label.to(device)
#         x = model(img)
#         pred = F.softmax(x, dim=1)
#         # print(pred.shape, img.shape)
#         pre_label = pred.max(dim=1)[1].data.cpu().numpy()
#         pre_label = [i for i in pre_label]
#         true_label = label.data.cpu().numpy()
#         true_label = [i for i in true_label]
#         all_acc, acc, dice = mean_dice(pre_label, true_label, num_classes = config['dataset']['class_num'], ignore_index = None)
#         val_dice = dice + val_dice
#     val_etis = val_dice.mean()/(idx+1)
# print("evaluating Kvasir")
# # Kvasir
# val_ds = Datareader(config['dataset']['test_Kvasir_img'], config['dataset']['test_Kvasir_label'], crop_size)
# val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
# # evaluate Kvasir
# val_dice = 0
# with torch.no_grad():
#     for idx, (img, label) in tqdm(enumerate(val_loader)):
#         img = img.to(device)
#         label = label.to(device)
#         x = model(img)
#         pred = F.softmax(x, dim=1)
#         # print(pred.shape, img.shape)
#         pre_label = pred.max(dim=1)[1].data.cpu().numpy()
#         pre_label = [i for i in pre_label]
#         true_label = label.data.cpu().numpy()
#         true_label = [i for i in true_label]
#         all_acc, acc, dice = mean_dice(pre_label, true_label, num_classes = config['dataset']['class_num'], ignore_index = None)
#         val_dice = dice + val_dice
#     val_Kvasir = val_dice.mean()/(idx+1)

# print(save_tensor[0].shape)