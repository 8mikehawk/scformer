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
# from utilis.augmentations import Compose, ColorJitter, RandomHorizontalFlip, RandomCrop, RandomScale


class LabelProcessor:
    def __init__(self, file_path):

        self.colormap = self.read_color_map(file_path)

        self.cm2lbl = self.encode_label_pix(self.colormap)

    @staticmethod
    def read_color_map(file_path):
        pd_label_color = pd.read_csv(file_path, sep=',')
        colormap = []
        for i in range(len(pd_label_color.index)):
            tmp = pd_label_color.iloc[i]
            color = [tmp['r'], tmp['g'], tmp['b']]
            colormap.append(color)
        return colormap

    @staticmethod
    def encode_label_pix(colormap):
        cm2lbl = np.zeros(256 ** 3)
        for i, cm in enumerate(colormap):
            cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i
        return cm2lbl

    def encode_label_img(self, img):
        data = np.array(img, dtype='int32')
        idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
        return np.array(self.cm2lbl[idx], dtype='int64')


class ISIC2018(Dataset):
    def __init__(self, file_path=[], crop_size=None, mode='train', class_dict_path=None):
        if len(file_path) != 2:
            raise ValueError("同时需要图片和标签文件夹的路径，图片路径在前")
        self.img_path = file_path[0]
        self.label_path = file_path[1]
        self.imgs = self.read_file(self.img_path)
        self.labels = self.read_file(self.label_path)
        self.crop_size = crop_size
        self.class_dict_path = class_dict_path

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        img = Image.open(img).convert('RGB')
        label = Image.open(label).convert('RGB')
        img, label = self.center_crop(img, label, self.crop_size)
        img, label = self.img_transform(img, label)
        # sample = {'image': img, 'label': label}
        # return sample
        return img, label

    def __len__(self):
        return len(self.imgs)

    def read_file(self, path):
        files_list = os.listdir(path)
        file_path_list = [os.path.join(path, img) for img in files_list]
        file_path_list.sort()
        return file_path_list

    def center_crop(self, data, label, crop_size):
        data = ff.resize(data, crop_size)
        label = ff.resize(label, crop_size)
        return data, label

    def img_transform(self, img, label):

        label_processor = LabelProcessor(self.class_dict_path)

        label = np.array(label)
        img = np.array(img)
        label = Image.fromarray(label.astype('uint8'))
        transform_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        img = transform_img(img)
        label = label_processor.encode_label_img(label)
        label = t.from_numpy(label)
        return img, label