from .sct import sct_b1
from .sct_dw import sct_b1 as sct_dw_b1
from .sct_pixel import sct_b1 as sct_pixel_b1
from .sct_dw_pixel import sct_b1 as sct_dw_pixel_b1
from .segformer import sct_b1 as b1
from .segformer_b2 import sct_b2 as segformer_b2
from .sct import sct_b4


def build(model_name, class_num=2):
    if model_name == "sct_b1":
        model = sct_b1(class_num=class_num)
        return model

    if model_name == "sct_dw_b1":
        model = sct_dw_b1(class_num=class_num)
        print(model)
        return model

    if model_name == "sct_pixel_b1":
        model = sct_pixel_b1(class_num=class_num)
        return model

    if model_name == "sct_dw_pixel_b1":
        model = sct_dw_pixel_b1(class_num=class_num)
        return model

    if model_name == "segformer":
        model = b1(class_num=class_num)
        return model

    if model_name == "segformer_b2":
        model = segformer_b2(class_num=class_num)
        return model        

    if model_name == "sct_b4":
        model = sct_b4(class_num=class_num)
        return model            

