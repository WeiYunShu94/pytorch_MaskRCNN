# structures文件说明

## ImageList
在PyTorch中，输入网络的第一个尺寸通常代表了这一批次的尺寸，因此，所有相同批次的所有元素都有相同的
height/width。为了支持同一批次不同尺寸和不同比例的图片，我们创造了`ImageList`这个类。图像被0
填充，如此就有相同大小的尺寸。在图像填充之前的原始尺寸则存储在`image_sizes`中，batched tensor
则存储在`tensors`当中。我们提供了一个方便的函数to_image_list，能够接受不同的输入类型，包括一系列
tensors,并且返回一个`ImageList`的对象。
```python
from maskrnn_benchmark.structures.image_list import to_image_list

images = [torch.rand(3, 100, 200), torch.rand(3, 150, 170)]
batched_images = to_image_list(images)

# it is also possible to make the final batched image be a multiple of a number
batched_images_32 = to_image_list(images, size_divisible=32)
```

## BoxList
对于一张图片，`BoxList`类包含一系列的bounding boxes，也包含了图片的(width,height) tuple。
它也包含一系列方法，让bounding boxes产生几何变化，如裁减，尺寸变化和翻转。这个类包含了两个不同的输入格式：
- xyxy,即，x1,y1,x2,y2
- xywh,即，x1,y1,w,h.
另外，每个BoxList也能包含一些其他信息，如，labels,visibility,probability scores等
这里有个例子，展现从一些列坐标中产生BoxList:
```python
from maskrcnn_benchmark.structures.bounding_box import BoxList, FLIP_LEFT_RIGHT

width = 100
height = 200
boxes = [
  [0, 10, 50, 50],
  [50, 20, 90, 60],
  [10, 10, 50, 50]
]
# create a BoxList with 3 boxes
bbox = BoxList(boxes, image_size=(width, height), mode='xyxy')

# perform some box transformations, has similar API as PIL.Image
bbox_scaled = bbox.resize((width * 2, height * 3))
bbox_flipped = bbox.transpose(FLIP_LEFT_RIGHT)

# add labels for each bbox
labels = torch.tensor([0, 10, 1])
bbox.add_field('labels', labels)

# bbox also support a few operations, like indexing
# here, selects boxes 0 and 2
bbox_subset = bbox[[0, 2]]
```