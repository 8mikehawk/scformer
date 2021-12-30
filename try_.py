
import numpy as np
import torch
import matplotlib.pyplot as plt

class Colorize:
    def __init__(self, n):
        self.cmap = self.colormap(256)
        self.cmap[n] = self.cmap[-1]
        self.cmap = torch.from_numpy(self.cmap[:n])#array->tensor

    def colormap(self, n):
        cmap=np.zeros([n, 3]).astype(np.uint8)
        cmap[0,:] = np.array([ 0,  0,  0])
        cmap[1,:] = np.array([244, 35,232])
        cmap[2,:] = np.array([ 70, 70, 70])
        cmap[3,:] = np.array([ 102,102,156])
        cmap[4,:] = np.array([ 190,153,153])
        # cmap[5,:] = np.array([ 153,153,153])
    
        # cmap[6,:] = np.array([ 250,170, 30])
        # cmap[7,:] = np.array([ 220,220,  0])
        # cmap[8,:] = np.array([ 107,142, 35])
        # cmap[9,:] = np.array([ 152,251,152])
        # cmap[10,:] = np.array([ 70,130,180])
    
        # cmap[11,:] = np.array([ 220, 20, 60])
        # cmap[12,:] = np.array([ 119, 11, 32])
        # cmap[13,:] = np.array([ 0,  0,142])
        # cmap[14,:] = np.array([  0,  0, 70])
        # cmap[15,:] = np.array([  0, 60,100])
    
        # cmap[16,:] = np.array([  0, 80,100])
        # cmap[17,:] = np.array([  0,  0,230])
        # cmap[18,:] = np.array([ 255,  0,  0])
        
        return cmap
 
    def __call__(self, gray_image):
        size = gray_image.size()#这里就是上文的output
        color_image = torch.ByteTensor(3, size[0], size[1]).fill_(0)
 
        for label in range(0, len(self.cmap)):
            mask = gray_image == label
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image

data = np.load("/data/segformer/scformer/pred.npz")['arr_0']
data = torch.as_tensor(data)

colorize = Colorize(4)
for i in range(32):
    if torch.max(data[i, :, :]) != 0:
        print(i)