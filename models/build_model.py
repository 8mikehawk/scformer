from .segformer import sct_b2 as segformer

from .srm import sct_b2 as srm
from .srm_fully_conv import sct_b2 as srm_fully_conv
from .srm_b2_pixel import sct_b2 as srm_b2_pixel
from .srm_b2_skip import sct_b2 as srm_b2_skip
from .scale_model import sct_b2 as scale_model
from .scale_model2 import sct_b2 as scale_model2
from .feilong_test import sct_b2 as feilong_test

from .srm_pretrain import mit_b2 as srm_pretrain

def build(model_name, class_num=2):

    if model_name == "segformer":
        model = segformer(class_num=class_num)
        return model
        
    if model_name == "srm":
        model = srm(class_num=class_num)
        return model
        
    if model_name == "srm_fully_conv":
        model = srm_fully_conv(class_num=class_num)
        return model


    if model_name == "srm_b2_pixel":
        model = srm_b2_pixel(class_num=class_num)
        return model
        
    if model_name == "srm_b2_skip":
        model = srm_b2_skip(class_num=class_num)
        return model

    if model_name == "feilong_test":
        model = feilong_test(class_num=class_num)
        return model
        
    if model_name == "scale_model":
        model = scale_model(class_num=class_num)
        return model     
                           
    if model_name == "scale_model2":
        model = scale_model2(class_num=class_num)
        return model         
    
    if model_name == "srm_pretrain":
        model = srm_pretrain(class_num=class_num)
        return model             