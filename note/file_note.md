###文件的精简
### 删除
需要直接交界给小益，要删除其中不必要的成分
`cocoapi`：这里主要是用于训练时候才使用cocoapi
`vision`：vision当中放有一些与本项目没有之间关系的文件，可以直接删除，然后根据自己的需要再重新写
`.idea`：不太明白这个文件的用处，删除之后，没有影响
其中`maskrcnn-benchmark/maskrcnn_benchmark`是最主要的部分，其他的可以删除，删除如下：
`.github`: 在此没有用处
`build`:在这里用处不大
`docker`：搬运工人，在这里用处不大
`maskrcnn_benchmark.egg-info`：在此用处不大
`tests`：在此用处不大
### 保留
`config`：这里存放cfg参数
其中maskrcnn_benchmark中的东西都必须保留：
`csrc`：？？
`data`：数据的处理
`engine`：？？
`layers`：存放每一层的参数
`modeling`：存放模型，backbone是骨架部分
`solver`：这里存放的是反向传播和学习率的调整
`structures`：bounding_box，image_list，boxlist_ops，segmentation，这四个函数
`utils`：常用的一些工具

### 添加
`utils_mianjin`：自己常用到的一些函数
`torchvision`：图像的处理，包括一些数据的加载，常用的模型以及一些常用的变化
`yacs`：cfg调用的函数，可以更加方便的实现，提交效率

### 说明文档
`install.md`：环境的安装配置
`readme.md`：如何使用这个环境

