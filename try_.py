# import torch
# from models import build
# # from loguru import logger
# # from tqdm import tqdm
# # import torch.nn as nn
# # import torch.optim as optmi
# # import torch.nn.functional as F
# # from utils.tools import mean_dice, mean_iou
# # from utils import DataBuilder, Datareader, CustomDataSet
# # from utils.loss import *
# # from torch.utils.data import DataLoader
# # from torchvision.transforms import Compose 
# # from torchvision import transforms
# # import torch
# # import os
# # import sys
# # import numpy as np
# # import yaml


# # 定义模型
# device = "cuda"
# model = build("srm")
# model.to(device)

# pth_path = "/data/segformer/scformer/pretrain/mit_b2.pth"

# check_point = torch.load(pth_path)
# model = check_point['backbone']
# model.load_state_dict(check_point['model_state_dict'])


# def load_checkpoint(filepath):
#     checkpoint = torch.load(filepath)
#     model = checkpoint['model']  # 提取网络结构
#     model.load_state_dict(checkpoint['model_state_dict'])  # 加载网络权重参数
#     optimizer = TheOptimizerClass()
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # 加载优化器参数
    
#     for parameter in model.parameters():
#         parameter.requires_grad = False
#     model.eval()
    
#     return model
    
# model = load_checkpoint('checkpoint.pkl')