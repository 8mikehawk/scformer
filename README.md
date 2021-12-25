# scformer (not finished)

# usage
1. in folder **train_package**, add your training folder for example **sct_b1_isic2018**.
2. add `test.conf` and `train.conf` as the configurations. It looks like
   ```
   [dataset]
    name = ISIC2018
    train_img_root   = "your data path"
    val_img_root     = "your data path"

    train_label_root = "your data path"
    val_label_root   = "your data path"

    class_num = 2 

    crop_size = (512, 512)
    batch_size = 8
    num_workers = 8

    [model]
    # model config
    name = sct_b2

    [schedule]
    # training config
    device = cuda
    lr = 1e-4
    max_epoch = 10000
    checkpoint_save_path = /mnt/DATA-1/DATA-2/Feilong/scformer/train_package/sct_b1_isic2018/

    [logger]
    path = /mnt/DATA-1/DATA-2/Feilong/scformer/train_package/sct_b1_isic2018/sct_b1_isic.log
   ```
3. add a new class on `/utils/my_dataset.py` if you want to train on a new dataset. Then, import this new class on the `train.py` like `from utils.tools import ISIC2018`
4. 

1. 创建文件夹
2. 改train.ocnf test.config
3. 改train中数据集的类
4. 确定train中model
5. 改train中train_ds 那部分