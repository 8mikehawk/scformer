import configparser
import models
from models import sct_b1
from loguru import logger
from utils.tools import build_dataset, build_model, ISIC2018
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optmi
import torch.nn.functional as F
from utils.tools import mean_dice
from torch.utils.data import DataLoader
import torch
import os

cf = configparser.ConfigParser()

cf.read("/mnt/DATA-1/DATA-2/Feilong/scformer/train_package/sct_b1_isic2018/test.conf")

# config
train_img_root = cf.get("dataset", "train_img_root")
test_img_root = cf.get("dataset", "test_img_root")
train_label_root = cf.get("dataset", "train_label_root")
test_label_root = cf.get("dataset", "test_label_root")
crop_size = (int(cf.get("dataset", "crop_size_1")), int(cf.get("dataset", "crop_size_2")))
batch_size = int(cf.get("dataset", "batch_size"))
num_workers = int(cf.get("dataset", "num_workers"))


# 定义模型
# model = build_model(cf.get("model", "name"), cf.get("dataset", "class_num"))
device = "cuda"
model = sct_b1(class_num=int(cf.get("dataset", "class_num")))
if device == "cpu":
    model.load_state_dict(torch.load(cf.get("model", "pretrained_model_path"), map_location=torch.device('cpu')))
else:
    model.load_state_dict(torch.load(cf.get("model", "pretrained_model_path")))
model = model.to(device)

# 加载测试集
test_ds = ISIC2018(train_img_root, test_img_root, train_label_root, test_label_root, crop_size, mode='val')
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

progress = tqdm(enumerate(test_loader), desc="testing ...", total=len(test_loader))
test_dice = 0
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
        all_acc, acc, dice = mean_dice(pre_label, true_label, num_classes = 2, ignore_index = None)
        test_dice = dice + test_dice
        progress.update(1)
    epoch_dice = test_dice.mean()/(idx+1)

print('test_dice_score :{:}'.format(epoch_dice))   