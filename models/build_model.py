from .segformer_b2_pretrained_original import sct_b2 as segformer_b2_pretrained_original


from .srm import sct_b2 as srm
from .srm_fully_conv import sct_b2 as srm_fully_conv
from .srm_b2_pixel import sct_b2 as srm_b2_pixel
from .srm_b2_skip import sct_b2 as srm_b2_skip

from .feilong_test import sct_b2 as feilong_test

def build(model_name, class_num=2):

    if model_name == "segformer_b2_pretrained_original":
        model = segformer_b2_pretrained_original(class_num=class_num)
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
                