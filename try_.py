from PIL import Image
import numpy as np
import torch


img_path = "/data/segformer/scformer/data/polyp/TrainDataset/images/1.png"
labe_path = "/data/segformer/scformer/data/polyp/TrainDataset/masks/1.png"


data = np.load("/data/segformer/scformer/pred.npz")['arr_0']