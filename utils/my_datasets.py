import pandas as pd
import os
import torch as t
import numpy as np
import torchvision.transforms.functional as ff
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from cv2 import cv2
import torch


class ISIC2018(Dataset):
    def __init__(self, train_img_root, val_img_root, train_label_root, val_label_root, crop_size, mode='train'):
        self.train_img_files = self.read_file(train_img_root)
        self.val_img_files = self.read_file(val_img_root)
        self.train_label_files = self.read_file(train_label_root)
        self.val_label_files = self.read_file(val_label_root)
        self.mode = mode
        self.crop_size = crop_size

    def __getitem__(self, index):
        if self.mode == 'train':
            img = Image.open(self.train_img_files[index])
            label = Image.open(self.train_label_files[index])

            img = img.resize((self.crop_size[0], self.crop_size[1]))
            label = label.resize((self.crop_size[0], self.crop_size[1]))

            img = np.array(img) / 255
            label = np.array(label)

            img = torch.as_tensor(img)
            label = torch.as_tensor(label)

            img = img.permute(2, 0, 1)

            return img.float(), label.long()

        if self.mode == 'val':
            img = Image.open(self.val_img_files[index])
            label = Image.open(self.val_label_files[index])

            img = img.resize((self.crop_size[0], self.crop_size[1]))
            label = label.resize((self.crop_size[0], self.crop_size[1]))

            img = np.array(img) / 255
            label = np.array(label)

            img = torch.as_tensor(img)
            label = torch.as_tensor(label)

            img = img.permute(2, 0, 1)

            return img.float(), label.long()       

    def __len__(self):
        if self.mode == 'train':
            total_img = len(self.train_img_files)
            return total_img
        if self.mode == 'val':
            total_img = len(self.val_img_files)
            return total_img

    def read_file(self, path):
        files_list = os.listdir(path)
        file_path_list = [os.path.join(path, img) for img in files_list]
        file_path_list.sort()
        return file_path_list   