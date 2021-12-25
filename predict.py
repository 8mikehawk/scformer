# from models import sct_b2
# import torch
# from utils import ISIC2018
# import numpy as np
# import matplotlib.pyplot as plt
# import torch.nn.functional as F
# from torchvision.utils import save_image
# import os
# from torch.utils.data import DataLoader
# from tqdm import tqdm

# #################################
# train_img_root = "/mnt/DATA-1/DATA-2/Feilong/scformer/data/ISIC2018/train/"
# val_img_root = "/mnt/DATA-1/DATA-2/Feilong/scformer/data/ISIC2018/val/"
# train_label_root = "/mnt/DATA-1/DATA-2/Feilong/scformer/data/ISIC2018/train_labels/"
# val_label_root = "/mnt/DATA-1/DATA-2/Feilong/scformer/data/ISIC2018/val_labels/"

# class_num = 2
# model = sct_b2(class_num=class_num)
# model = model.to('cpu')
# crop_size = (512, 512)

# save_path = "predict_results/"
# batch_size = 16
# num_workers = 16
# #################################

# model.load_state_dict(torch.load("models/checkpoints/val_best.pth", map_location=torch.device('cpu')))

# val_ds = ISIC2018(train_img_root, val_img_root, train_label_root, val_label_root, crop_size, mode='val')

# val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

# for idx, (x, y) in tqdm(enumerate(val_loader)):
#     x = model(x)
#     pred = F.softmax(x, dim=1)
#     pred = pred.reshape((batch_size, class_num, crop_size[0]*crop_size[0]))
#     pred = torch.argmax(pred, dim=1)
#     pred = torch.where(pred == 1, 255, 0)
#     pred = pred.reshape((batch_size, crop_size[0], crop_size[1]))
#     break
# out = torch.cat((pred.float(), y.float()), dim=0)
# out = out.unsqueeze(1)
# save_image(out, os.path.join(save_path, f"result.png"))

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
from torchvision.utils import save_image

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
device = "cpu"
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
with torch.no_grad():
    for idx, (x, y) in tqdm(enumerate(test_loader)):
        x = model(x)
        pred = F.softmax(x, dim=1)
        pred = pred.reshape((batch_size, int(cf.get("dataset", "class_num")), crop_size[0]*crop_size[0]))
        pred = torch.argmax(pred, dim=1)
        pred = torch.where(pred == 1, 255, 0)
        pred = pred.reshape((batch_size, crop_size[0], crop_size[1]))
        break
    out = torch.cat((pred.float(), y.float()), dim=0)
    out = out.unsqueeze(1)
    save_image(out, os.path.join(cf.get("predict", "save_path"), "result.png"))