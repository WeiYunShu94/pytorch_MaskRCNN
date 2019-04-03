# Faster RCNN and Mask RCNN in PyTorch1.0
这个项目是基于facebook的mask RCNN in PyTorch1.0基础上改进，实现特定bowl的检测和
分割。[源代码链接](https://github.com/facebookresearch/maskrcnn-benchmark)和
[论文链接](cn.arxiv.org/pdf/1703.06870v3)（这里的代码和论文中的代码并不是同一个）


## Installation
安装和配置环境在[install.md](install.md)中有详细说明

## demo
盘子检测的demo如下：[bowl_detect()](demo/bowl_detect.py)
```python
# 将参数放入网络中
cfg.merge_from_file("../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml")
# 让所有的CfgNode和所有他的子节点保持不变
cfg.freeze()
# 这里的confidence_threshold是置信度，淘汰置信度低于0.7的框
# min_image_size是图像的进入网络时压缩的大小，为较小边，采用coco数据集来训练的，训练时采用的是800，这时效果最好
coco_demo = COCODemo(cfg, confidence_threshold=0.7, min_image_size=800)
# 读取图像
img = cv2.imread('2.jpg')
# 返回图像中标记的图像以及每个盘子的img_mask
composite, masks = coco_demo.run_on_opencv_image(img)
img_masks, img_maskRGBs = coco_demo.get_masks(img, masks)
if img_masks != 0:
    cv2.imshow("composite", composite)
    cv2.imshow("img_mask", img_masks[0] * 255)
    cv2.imshow("img_maskRGB", img_maskRGBs[0])
    cv2.waitKey(0)
else:
    print("no bowl detect! ")
```

## 参数与模型比较
在[MODEL_ZOO.md](MODEL_ZOO.md)能够发现预训练模型，以及与Detectron和mmdetection的比较，
我们这里选用的是`mask_rcnn_R-50-FPN`

## Inference in a few lines
这里提供一个更有效的`COCODemo`类，用预训练简化书写inference piplines：
```python
from maskrcnn_benchmark.config import cfg

config_file = "../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"

# 用参数文件更新参数选项
cfg.merge_from_file(config_file)
# 手工书写一些选项
cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
)
# load image and then run prediction
image = ...
predictions = coco_demo.run_on_opencv_image(image)
```

## 在COCO 数据集上训练


