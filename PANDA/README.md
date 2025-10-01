# Code and datasets 
Cross Position Aggregation Network for Few-shot Strip Steel Surface Defect Segmentation

## Dataset:

You can contact ours by email to get the dataset-decompression-password（songkc@me.neu.edu.cn）

## Config：

  Before training you need to modify the *.YAML file path in the config folder, such as:
  ```js
  data_root: '../FSSD-12/'
  train_list: './data_list/train/fold0_defect.txt'
  val_list: './data_list/val/fold0_defect.txt'
  ```
## Training：

+ You can directly use our dataset for training.
+ Download the ImageNet pretrained [**backbones**](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155122171_link_cuhk_edu_hk/EQEY0JxITwVHisdVzusEqNUBNsf1CT8MsALdahUhaHrhlw?e=4%3a2o3XTL&at=9) and put them into the `initmodel` directory.
+ you can use your dataset for training, this process requires regenerating the "data_list" file.

# Related Repositories

+ **TGRNet**: https://github.com/bbbbby-99/TGRNet-Surface-Defect-Segmentation
+ **PFENEt**: https://github.com/Jia-Research-Lab/PFENet

