实现yolov1目标检测算法，主要为了熟悉算法，学习使用。

包括完整的voc数据集读取，标签制作，自定义训练，网络和损失函数编写，输出后处理（NMS）代码

将原论文的类googlenet卷积替换为vgg16在imagenet上的预训练权重，在voc2012train+val上训练验证，没有测试集，没有做数据增强。

![alt](test.jpg)