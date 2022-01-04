import configparser
import io
from models import build
from loguru import logger
from utils.tools import mean_iou
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optmi
import torch.nn.functional as F
from utils.tools import mean_dice
from utils import DataBuilder
from torch.utils.data import DataLoader
import torch
import os
import sys
import yaml

f = open(sys.argv[1])
config = yaml.safe_load(f)

# # config
# train_img_root = cf.get("dataset", "train_img_root")
# test_img_root = cf.get("dataset", "test_img_root")
# train_label_root = cf.get("dataset", "train_label_root")
# test_label_root = cf.get("dataset", "test_label_root")
# crop_size = (int(cf.get("dataset", "crop_size_1")), int(cf.get("dataset", "crop_size_2")))
# batch_size = int(cf.get("dataset", "batch_size"))
# num_workers = int(cf.get("dataset", "num_workers"))

train_img_root = config['dataset']['train_img_root']
test_img_root = config['dataset']['test_img_root']
train_label_root = config['dataset']['train_label_root']
test_label_root = config['dataset']['test_label_root']
crop_size = (
    config['dataset']['crop_size']['w'],
    config['dataset']['crop_size']['h']
)
batch_size = config['dataset']['batch_size']
num_workers = config['dataset']['num_workers']
checkpoint_save_path = config['other']['checkpoint_save_path']


# 定义模型
# model = build_model(cf.get("model", "name"), cf.get("dataset", "class_num"))
device = config['training']['device']
model = build(model_name=config['model']['model_name'])
if device == "cpu":
    model.load_state_dict(torch.load(config['test']['checkpoint_save_path']), map_location=torch.device('cpu'))
else:
    model.load_state_dict(torch.load(config['test']['checkpoint_save_path']))
model = model.to(device)


# 加载测试集
test_ds = DataBuilder(train_img_root, test_img_root, train_label_root, test_label_root, crop_size, mode='val')
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

test_dice = 0
test_iou = 0
print("testing ....")
with torch.no_grad():
    for idx, (img, label) in tqdm(enumerate(test_loader)):
        img = img.to(device)
        label = label.to(device)
        x = model(img)
        pred = F.softmax(x, dim=1)
        pre_label = pred.max(dim=1)[1].data.cpu().numpy()
        pre_label = [i for i in pre_label]
        true_label = label.data.cpu().numpy()
        true_label = [i for i in true_label]
        # dice
        all_acc, acc, dice = mean_dice(pre_label, true_label, num_classes = config['dataset']['class_num'], ignore_index = None)
        # iou
        all_acc_, acc_, iou = mean_iou(pre_label, true_label, num_classes = config['dataset']['class_num'], ignore_index = None)
        test_dice = dice + test_dice
        test_iou = iou + test_iou
    epoch_iou = test_iou.mean()/(idx+1)    
    epoch_dice = test_dice.mean()/(idx+1)

print('| test_dice_score :{:} | test_iou : {:} |'.format(epoch_dice, epoch_iou))   

# test_iou = 0
# with torch.no_grad():
#     for idx, (img, label) in tqdm(enumerate(test_loader)):
#         img = img.to(device)
#         label = label.to(device)
#         x = model(img)
#         pred = F.softmax(x, dim=1)
#         pre_label = pred.max(dim=1)[1].data.cpu().numpy()
#         pre_label = [i for i in pre_label]
#         true_label = label.data.cpu().numpy()
#         true_label = [i for i in true_label]
#         all_acc, acc, iou = mean_iou(pre_label, true_label, num_classes = config['dataset']['class_num'], ignore_index = None)
#         test_iou = iou + test_iou
#         progress.update(1)
#     epoch_iou = test_iou.mean()/(idx+1)

# print('test_iou :{:}'.format(epoch_iou))   