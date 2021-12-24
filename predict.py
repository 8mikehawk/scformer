from models import sct_b2
import torch
from utils import ISIC2018
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.utils import save_image
import os
from torch.utils.data import DataLoader
from tqdm import tqdm

#################################
val_img_root = "/mnt/DATA-1/DATA-2/Feilong/scformer/data/ISIC2018/val/"
val_label_root = "/mnt/DATA-1/DATA-2/Feilong/scformer/data/ISIC2018/val_labels/"
class_dict_path = "data/ISIC2018/class_dict.csv"

class_num = 2
model = sct_b2(class_num=class_num)
model = model.to('cpu')
crop_size = (512, 512)

save_path = "predict_results/"
batch_size = 4
num_workers = 4
#################################

model.load_state_dict(torch.load("models/checkpoints/val_best.pth", map_location=torch.device('cpu')))

val_ds = ISIC2018([val_img_root, val_label_root], crop_size, class_dict_path=class_dict_path)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

for idx, (x, y) in tqdm(enumerate(val_loader)):
    x = model(x)
    pred = F.softmax(x, dim=1)
    pred = pred.reshape((batch_size, class_num, crop_size[0]*crop_size[0]))
    pred = torch.argmax(pred, dim=1)
    pred = torch.where(pred == 1, 0, 255)
    pred = pred.reshape((batch_size, crop_size[0], crop_size[1]))
    break
out = torch.cat((pred.float(), y.float()), dim=0)
out = out.unsqueeze(1)
save_image(out, os.path.join(save_path, f"result.png"))