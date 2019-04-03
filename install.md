# Installation

## operating system enviroment:
- ubuntu16.04

## Requirements:
- PyTorch1.0
- CUDA 9.0
- CUDNN 7.0
- GCC > 4.9.3
### Option 0: 安装ubuntu16.04
### Option 1: 安装CUDA和CUDNN
```bash
# 查看cuda和cudnn的版本号
# CUDA：
cat /usr/local/cuda/version.txt
# CUDNN：
cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2
```
### Option 2: 安装python及相应的虚拟环境
```bash
# step1: 安装anaconda3以及pycharm
# pycharm是最常用的编译环境，这个根据自己习惯而定
# anaconda3中的canda在环境管理方面能够节省很多麻烦，anaconda3中自带的是python3版本

# 创建一个合适的python环境
conda create --name maskrcnn_benchmark # 创建虚拟环境
source activate maskrcnn_benchmark     # 进入虚拟环境 


```


