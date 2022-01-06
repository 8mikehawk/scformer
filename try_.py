import numpy as np
import matplotlib.pyplot as plt


pre_label = np.load("/data/segformer/scformer/pre_label.npz")['arr_0']
true_label = np.load("/data/segformer/scformer/true_label.npz")['arr_0']