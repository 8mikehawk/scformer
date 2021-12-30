import configparser
from models import sct_b1_pixel, build
from loguru import logger
from utils.tools import ISIC2018, Colorize
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
from torchvision.utils import save_image
import yaml
from PIL import Image
import numpy as np

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
class_num = config['dataset']['class_num']


# 定义模型
# model = build_model(cf.get("model", "name"), cf.get("dataset", "class_num"))
device = config['training']['device']
model = build(model_name=config['model']['model_name'], class_num=config['dataset']['class_num'])
if device == "cpu":
    model.load_state_dict(torch.load(config['test']['checkpoint_save_path']), map_location=torch.device('cpu'))
else:
    model.load_state_dict(torch.load(config['test']['checkpoint_save_path']))
model = model.to(device)


# 加载测试集
test_ds = DataBuilder(train_img_root, test_img_root, train_label_root, test_label_root, crop_size, mode='val')
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

with torch.no_grad():
    for idx, (x, y) in tqdm(enumerate(test_loader)):
        x = x.to(device)
        y = y.to(device)
        x = model(x)
        pred = F.softmax(x, dim=1)
        # result = np.argmax(pred.detach().cpu().numpy(), axis=1)

        pred = pred.reshape((pred.shape[0], class_num, crop_size[0]*crop_size[1]))
        pred = torch.argmax(pred, dim=1)
        pred = torch.where(pred == 1, 255, 0)
        pred = pred.reshape((pred.shape[0], crop_size[0], crop_size[1]))
        break
    out = torch.cat((pred.float(), y.float()), dim=0)
    out = out.unsqueeze(1)
    save_image(out, "result.png")
