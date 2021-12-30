from segformer import sct_b1
from thop import profile
import torch


model = sct_b1()
input = torch.randn(1, 3, 512, 512)
macs, params = profile(model, inputs=(input, ))
print('macs:',macs)
print('params:',params)