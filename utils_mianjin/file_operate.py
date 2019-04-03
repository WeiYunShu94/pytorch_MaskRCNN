import numpy as np
import os
import shutil


def dir_sort():
    # 将文件下的所有文件按照从小到大的顺序重新命名
    root = "/home/atr/WMJ/deep_sort_pytorch-master/crop/atr10190305data_release3/merge/ch04"
    dirs = [os.path.join(root, dirs) for dirs in os.listdir(root)]
    dirs = sorted(dirs, key=lambda x: int(x))
    for i, dir in enumerate(dirs):
        #     print(i,img_name)
        os.rename(os.path.join(root, dir), os.path.join(root, str(i)))

if __name__=="__main__":
    dir_sort()