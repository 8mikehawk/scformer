import torch

from my_dataset import CustomisedDataSet
from torchvision import transforms
from torchvision.utils import save_image
import albumentations as A
import cv2
import matplotlib.pyplot as plt

import numpy as np

#     def img_transform(self, img, label):
#         label = np.array(label)
#         img = np.array(img)
# #        img = Image.fromarray(img.astype('uint8'))
#         transform = A.Compose([A.VerticalFlip(p=0.5), A.HorizontalFlip(p=0.5),
#                             A.RandomScale(scale_limit=(0.75, 1.25), interpolation=cv2.INTER_CUBIC, p=0.5),
#                             A.ShiftScaleRotate(rotate_limit=(0,359), p=0.5),
# #                            A.PadIfNeeded(p=1, min_height=352, min_width=352),
#                             ])(image=img, mask=label)
                            
#         img = transform['image']
#         label = transform['mask']
        
#         return img, label


img_path = "/data/segformer/scformer/data/polyp/TrainDataset/images"
label_path = "/data/segformer/scformer/data/polyp/TrainDataset/masks"

data_ds = CustomisedDataSet(img_path, label_path, transformation=True)

# img, label = data_ds[np.random.randint(0, 200)]
img, label = data_ds[20]
# label = label.repeat(3,1,1) * 255

fig = plt.figure(figsize=(5.12, 10.24))
fig.add_subplot(1,2,1)
plt.imshow(img)
fig.add_subplot(1,2,2)
plt.imshow(label)
plt.show()