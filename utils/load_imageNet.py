from torch.utils.data import Dataset
import albumentations as A
from torchvision import transforms
from PIL import Image
import numpy as np
import torch

class ImageNetLoader(Dataset):
    def __init__(self, train_txt, val_txt, mode='train'):
        super().__init__()
        self.trian_txt = train_txt
        self.val_txt = val_txt

        self.mode = mode

        self.train_img = []
        self.train_label = []

        self.val_img = []
        self.val_label = []

        self.init_path()

    
    def __getitem__(self, index):
        if self.mode == "train":
            img = Image.open(self.train_img[index])
            label = self.train_label[index]

            img = img.resize((224, 224))

            img = np.array(img) 
            
            if len(img.shape) != 3:
                img = np.repeat(img, 3)
                img = img.reshape((224, 224, 3))
            if img.shape[2] == 4:
                img = img[:,:,:3]

            img = torch.as_tensor(img)
            label = torch.as_tensor(label)

            img = img.permute(2, 0, 1)

            img = self.transformation(img)

            return img.float(), label.long()

        if self.mode == "val":
            img = Image.open(self.val_img[index])
            label = self.val_label[index]

            img = img.resize((224, 224))

            img = np.array(img)
            
            if len(img.shape) != 3:
                img = np.repeat(img, 3)
                img = img.reshape((224, 224, 3))
            if img.shape[2] == 4:
                img = img[:,:,:3]

            img = torch.as_tensor(img)
            label = torch.as_tensor(label)

            img = img.permute(2, 0, 1)

            img = self.transformation(img)

            return img.float(), label.long()

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_img)
        if self.mode == 'val':
            return len(self.val_img)

    def init_path(self):
        """
        return numpy.array
        """
        with open(self.trian_txt, 'r') as fr:
            img_path = fr.readlines()
            for i in range(len(img_path)):
                self.train_img.append(img_path[i].split(' ')[0][:-1])
                self.train_label.append(int(img_path[i].split(' ')[1]))


        with open(self.val_txt, 'r') as fr:
            val_path = fr.readlines()
            for i in range(len(val_path)):
                self.val_img.append(val_path[i].split(' ')[0][:-1])
                self.val_label.append(int(val_path[i].split(' ')[1]))            

        self.train_img = np.array(self.train_img)
        self.train_label = np.array(self.train_label)
        self.val_label = np.array(self.val_label)
        self.val_img = np.array(self.val_img)
    
    def transformation(self, img):
        train_transformer = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(45),
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        img = train_transformer(img)

        return img
        
    
if __name__ == '__main__':
    imageNet = ImageNetLoader("/data/imageNet/train_.txt", "/data/imageNet/val_.txt")
    img, label = imageNet[2000] 