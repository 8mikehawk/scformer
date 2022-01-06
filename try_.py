from os import tcgetpgrp
from torch._C import layout
from models import build
# from utils import DataBuilder, Datareader, CustomDataSet
import numpy as np
import torch

np.seterr(divide='ignore',invalid='ignore')


# f = open(sys.argv[1])
# config = yaml.safe_load(f)

device = "cpu"

def load_pretrained(pth_path, model):
    pretrained_dict = torch.load(pth_path)
    model_dict = model.state_dict()

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    return model

# 定义模型
model = build("scale_model")
model.to(device)

pretrained_dict = torch.load("/data/segformer/scformer/train_package/imageNet_pretrain/train_best.pth")
model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)