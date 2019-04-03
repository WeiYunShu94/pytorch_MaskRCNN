"""
 输入：img
 输出：img_mask：[num,label,w,h]
      img_maskRGB: [num,label,w,h,3]
      若没有图片，则num = 0
"""
import cv2
import torch
import numpy as np
import time
from torchvision import transforms as T
from maskrcnn_benchmark.config import cfg

from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker


class COCODemo(object):
    CATEGORIES = [
        "__background",
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",   # 46
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]
    def __init__(
        self,
        cfg,
        confidence_threshold=0.5,
        show_mask_heatmaps=False,
        masks_per_dim=2,
        min_image_size=124,
    ):
        self.cfg = cfg.clone()
        self.model = build_detection_model(cfg)
        self.model.eval()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.model.to(self.device)
        self.min_image_size = min_image_size

        checkpointer = DetectronCheckpointer(cfg, self.model)
        _ = checkpointer.load(cfg.MODEL.WEIGHT)

        self.transforms = self.build_transform()

        mask_threshold = -1 if show_mask_heatmaps else 0.5
        self.masker = Masker(threshold=mask_threshold, padding=1)

        # used to make colors for each class
        self.palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])

        self.cpu_device = torch.device("cpu")
        self.confidence_threshold = confidence_threshold

    def build_transform(self):
        """
        将原图进行变换，输入到网络的入口
        """
        cfg = self.cfg
        #
        if cfg.INPUT.TO_BGR255:
            to_bgr_transform = T.Lambda(lambda x: x * 255)
        else:
            to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

        normalize_transform = T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        )
        # 这里要将原图进行裁减、转变成Tensor类型，并且进行标准化
        transform = T.Compose(
            [
                T.ToPILImage(),                     # 将图像转换成PILImage格式
                T.Resize(self.min_image_size),      # 将原图压缩
                T.ToTensor(),                       # 转换成Tensor()格式，能够直接放入网络中
                to_bgr_transform,                   # 将图像转换成BGR形式，0~255
                normalize_transform,                # 将图像正交化，方便后期的处理
            ]
        )
        return transform

    def run_on_opencv_image(self, image):
        # 得到所有的预测值
        predictions = self.compute_prediction(image)
        # 通过一个阈值进行判定，淘汰低于这个阈值分数的
        top_predictions = self.select_top_predictions(predictions)
        # type(image) 是 numpy
        result = image.copy()
        # 将mask的边缘覆盖到原图上
        result = self.overlay_mask(result, top_predictions)
        # 提取出mask图，这里的mask肯能有1个，可能没有，也可能有多个
        mask = top_predictions.get_field("mask").numpy()
        return result,mask

    def compute_prediction(self, original_image):
        # apply pre-processing to image
        # 对原图进行预处理，得到
        image = self.transforms(original_image)
        # convert to an ImageList, padded so that it is divisible by
        # cfg.DATALOADER.SIZE_DIVISIBILITY
        image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)
        # compute predictions
        with torch.no_grad():
            predictions = self.model(image_list)
        predictions = [o.to(self.cpu_device) for o in predictions]

        # 这里每次只检测一幅图像
        prediction = predictions[0]

        # reshape prediction (a BoxList) into the original image size
        height, width = original_image.shape[:-1]
        prediction = prediction.resize((width, height))

        if prediction.has_field("mask"):
            # if we have masks, paste the masks in the right position
            # in the image, as defined by the bounding boxes
            masks = prediction.get_field("mask")
            masks = self.masker(masks, prediction)
            prediction.add_field("mask", masks)
        return prediction

    def select_top_predictions(self, predictions):
        """
            这里挑选出来的全部都是盘子
        """
        labels = predictions.get_field("labels")
        # keep = torch.nonzero(labels == 1).squeeze(1)
        keep = torch.nonzero(labels == 46).squeeze(1)
        predictions = predictions[keep]
        scores = predictions.get_field("scores")
        keep = torch.nonzero(scores > self.confidence_threshold).squeeze(1)
        predictions = predictions[keep]
        scores = predictions.get_field("scores")
        _, idx = scores.sort(0, descending=True)
        return predictions[idx]

    def compute_colors_for_labels(self, labels):
        """
        Simple function that adds fixed colors depending on the class
        """
        colors = labels[:, None] * self.palette
        colors = (colors % 255).numpy().astype("uint8")
        return colors

    def overlay_mask(self, image, predictions):
        masks = predictions.get_field("mask").numpy()
        labels = predictions.get_field("labels")
        colors = self.compute_colors_for_labels(labels).tolist()
        labels = [self.CATEGORIES[i] for i in labels]
        for mask, label, color in zip(masks,labels, colors):
            thresh = mask[0, :, :, None]
            _, contours, hierarchy = cv2.findContours(
                thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            image = cv2.drawContours(image, contours, -1, color, 3)
        return image

    def get_masks(self,img,masks):
        if masks.shape[0] == 0:
            return 0,0
        img_masks = []
        for i in range(masks.shape[0]):
            img_masks.append(masks[i][0])
        img_maskRGBs = []
        for i in range(masks.shape[0]):
            image = np.zeros(img.shape, dtype=np.uint8)
            image[:, :, 0] = np.multiply(img[:, :, 0], img_masks[i])
            image[:, :, 1] = np.multiply(img[:, :, 1], img_masks[i])
            image[:, :, 2] = np.multiply(img[:, :, 2], img_masks[i])
            img_maskRGBs.append(image)
        return img_masks,img_maskRGBs


def main():
    # 将参数放入网络中
    cfg.merge_from_file("../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml")
    # 让所有的CfgNode和所有他的子节点保持不变
    cfg.freeze()
    # 这里的confidence_threshold是置信度，淘汰置信度低于0.7的框
    # min_image_size是图像的进入网络时压缩的大小，为较小边，采用coco数据集来训练的，训练时采用的是800，这时效果最好
    coco_demo = COCODemo(cfg, confidence_threshold=0.2, min_image_size=1280)
    # 读取图像
    img = cv2.imread('pic/atr_bowl2.jpg')
    img = cv2.resize(img,(1000,800))
    # 返回图像中标记的图像以及每个盘子的img_mask
    composite, masks = coco_demo.run_on_opencv_image(img)
    img_masks, img_maskRGBs = coco_demo.get_masks(img, masks)
    if img_masks != 0:
        cv2.imshow("composite", composite)
        # cv2.imwrite("composite.jpg",composite)
        # cv2.imshow("img_mask", img_masks[0] * 255)
        # cv2.imshow("img_maskRGB", img_maskRGBs[0])
        cv2.waitKey(0)
    else:
        print("no bowl detect! ")

def webcam():
    # 将参数放入网络中
    cfg.merge_from_file("../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml")
    # 让所有的CfgNode和所有他的子节点保持不变
    cfg.freeze()
    # 这里的confidence_threshold是置信度，淘汰置信度低于0.7的框
    # min_image_size是图像的进入网络时压缩的大小，为较小边，采用coco数据集来训练的，训练时采用的是800，这时效果最好
    coco_demo = COCODemo(cfg, confidence_threshold=0.2, min_image_size=800)
    cap = cv2.VideoCapture(0)
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        composite, masks = coco_demo.run_on_opencv_image(frame)
        print("Time: {:.2f} s / img".format(time.time() - start_time))
        cv2.imshow('frame', composite)  # 一个窗口用以显示原视频
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    # cam = cv2.VideoCapture("/home/atr/WMJ/MaskRCNN/maskrcnn-benchmark/demo_wmj/192.168.153.208.mp4")
    # while True:
    #     # start_time = time.time()
    #     ret_val, img = cam.read()
    #     # composite, masks = coco_demo.run_on_opencv_image(img)
    #     # print("Time: {:.2f} s / img".format(time.time() - start_time))
    #     cv2.imshow("COCO detections", img)
    #     if cv2.waitKey(1) == 27:
    #         break  # esc to quit
    # cv2.destroyAllWindows()

# def main1():
#     import cv2
#     cap = cv2.VideoCapture(0)
#     k = 0
#     while k != 27:  # esc
#         ret, img = cap.read(0)
#     cv2.imshow(‘233’, img)
#     k = cv2.waitKey(20) & 0xff
#
#     print( ‘begin
#     to
#     record
#     images…’ )
#
#     for ii in range(1000):
#         ret, img = cap.read(0)
#     cv2.imshow(‘233’, img)
#     cv2.imwrite(‘imaged % 04
#     d.jpg’ % (ii), img)
#     cv2.waitKey(20)

def cam():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)  # 一个窗口用以显示原视频
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    # webcam()
    # cam()