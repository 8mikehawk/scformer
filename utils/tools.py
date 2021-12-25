import time
import torch
import numpy as np


class Timer():
    def __init__(self):
        self.last_time = time.time()
        self.remain_hour = None
        self.remain_min = None
        self.remain_second = None

    def get_remain_time(self, idx, max_epoch):
        remain_time=(time.time()-self.last_time)*(max_epoch-idx-1)
        self.last_time=time.time()

        self.remain_hour = int(remain_time / 3600)
        self.remain_min = int((remain_time - self.remain_hour * 3600) / 60)
        self.remain_second = int(remain_time - (3600 * self.remain_hour) - (60 * self.remain_min))

        # print(f"hour {self.remain_hour}, min {self.remain_min}, second {self.remain_second}")

        return {"hour": self.remain_hour, "min": self.remain_min, "second": self.remain_second}


def iou_mean(pred, target, n_classes = 1):
#n_classes ：the number of classes in your dataset,not including background
# for mask and ground-truth label, not probability map
    ious = []
    iousSum = 0
    pred = torch.from_numpy(pred)
    pred = pred.view(-1)
    target = np.array(target)
    target = torch.from_numpy(target)
    target = target.view(-1)    
    # Ignore IoU for background class ("0")
    for cls in range(1, n_classes+1):  # This goes from 1:n_classes-1 -> class "0" is ignored
      pred_inds = pred == cls
      target_inds = target == cls
      intersection = (pred_inds[target_inds]).long().sum().data.cpu().item()  # Cast to long to prevent overflows
      union = pred_inds.long().sum().data.cpu().item() + target_inds.long().sum().data.cpu().item() - intersection
      if union == 0:
        ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
      else:
        ious.append(float(intersection) / float(max(union, 1)))
        iousSum += float(intersection) / float(max(union, 1))
    return iousSum/n_classes

def dice_coeff(pred, target):
    smooth = 1e-5
    pred = np.array(pred)
    target = np.array(target)
    
    pred = torch.as_tensor(pred)
    target = torch.as_tensor(target)
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()
    res = (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)
    return res