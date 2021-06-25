"""
    网络模型的搭建
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import voc, coco
import os


class SSD(nn.Module):
    """
    调用SSD实例获得(loc,conf,priors)
    初始化参数:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: 列表,vgg函数的返回值，包含了conv1_2,conv2_2,conv3_3,conv4_3,conv5_3,conv6和conv_7
        extras: 列表，add_extras函数返回值，包含了conv8_2,conv9_2,conv10_2,conv11_2
        head: 元组，包含了由multibox函数生成的loc_layers, conf_layers
              loc_layers,conf_layers是长度为6的列表
    """

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(SSD, self).__init__()
        self.phase = phase # test or train
        self.size = size
        self.num_classes = num_classes
        self.cfg = (coco, voc)[num_classes == 21] # 挑出voc配置
        self.priorbox = PriorBox(self.cfg)
        # self.priors = Variable(self.priorbox.forward(), volatile=True) # 此老版本语句已经失效
        with torch.no_grad():
            self.priors = Variable(self.priorbox.forward)

        # === SSD的骨干网络 ===
        # vgg函数返回一个包括conv1_2到con7的列表
        self.vgg_layers = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extra_layers = nn.ModuleList(extras) # 包括conv8_2到con11_2
        # === SSD的骨干网络 ===

        # === SSD的位置偏移量矩阵和置信度生成层 ===
        self.loc_layers = nn.ModuleList(head[0])
        self.conf_layers = nn.ModuleList(head[1])
        # === SSD的位置偏移量矩阵和置信度生成层 ===

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1) # 不知道干嘛的
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45) # # 不知道干嘛的

    def forward(self, x):
        """

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list() # 存储6个特征图层
        loc = list() # 存储对特征图层进行位置偏移量预测卷积后的矩阵
        conf = list() # 存储对特征图层进行置信度预测卷积后的矩阵

        # 此循环一直到con4_3(特征图层1)后面的relu层
        # 将conv4_3生成的特征图添加到source列表中
        for k in range(23):
            x = self.vgg_layers[k](x)
        s = self.L2Norm(x) # 为什么在这里需要L2Norm？
        sources.append(s)

        # 此循环一直到conv7(特征图层2)后面的relu层
        # 将conv7生成的特征图添加到source列表中
        for k in range(23, len(self.vgg_layers)): # len(self.vgg)=35
            x = self.vgg_layers[k](x)
        sources.append(x)

        # 将con8_2,conv9_2,conv10_2,conv11_2生成的特征图添加到source列表中
        for k, v in enumerate(self.extra_layers): # # len(self.extras)=8
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # x:6个特征图层
        # l:对特征图层要进行的卷积操作，预测位置
        # c:对特征图层要进行的卷积操作，预测置信度
        # 此循环将各个特征图中的定位和分类预测结果append进列表中
        for (x, l, c) in zip(sources, self.loc_layers, self.conf_layers):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        # loc:存储了6个位置偏移量矩阵
        # conf:存储了6个置信度矩阵
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], dim=1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], dim=1)

        if self.phase == "test":
        # 如果是测试阶段需要对定位和分类的预测结果进行后处理得到最终的预测框
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,self.num_classes)),                # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
        # 如果是训练阶段则直接输出定位和分类预测结果以计算损失函数
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# cfg参数为conv1_2、conv2_2、conv3_3、conv4_3、conv5_3的卷积out_channels:
# [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
#             512, 512, 512]
# i:输入图片的channel
# 定义了conv1_2、conv2_2、conv3_3、conv4_3、conv5_3,且conv6和conv7就是vgg16中的全连接层
def vgg(cfg, i, batch_norm=False):
    layers = []
    # global in_channels = i 可以将conv_7的输出channel直接传递给add_extras函数
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers # layers为list len(layers)=15


# convn_x：n为第n个卷积块层，x为此卷积块有多少个单卷积层
# vgg函数中只定义至conv7，add_extras定义conv8_2，conv9_2，conv10_2，conv11_2
# out_channels cfg:[256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256], 1024
# layers =     [
#               Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1)),
# 用于特征提取   Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
#               Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1)),
# 用于特征提取   Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
#               Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1)),
# 用于特征提取   Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1)),
#               Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1)),
# 用于特征提取   Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1))
#              ]
def add_extras(cfg, i, batch_norm=False):
    layers = []
    in_channels = i # 接conv7，故i应该等于1024
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           # add_extras函数中使用stride=2来减少分辨率而不是像vgg函数中使用池化
                           # conv10_2、conv11_2不需要使用stride=2(用公式计算以下就知道了)
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


def multibox(vgg, extra_layers, cfg, num_classes):
    """
    predictor multibox：为SSD网络提供loc,conf预测层
    # vgg:列表，包含了到conv7的base网络
    # extra_layers:列表，包含了新添加的extras层
    # cfg:要做特征提取的6个层中每个网格单元对应的default box个数,cfg = [4, 6, 6, 6, 4, 4]
    # num_classes:总的类别数（20+背景类=21）
    """
    loc_layers = []
    conf_layers = []
    vgg_source = [21, -2] # 21：conv4_3中最后一个3x3x512 -2：conv7

    # 此循环从vgg函数返回的列表中挑出out_channel为512，1024的特征图层1与特征图层2
    # 并以out_channel来生成[loc1,loc2],[conf1,conf2]
    for k, v in enumerate(vgg_source): # 目的是提取这些特征图层的out_channes
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]

    # 此循环从add_extras函数返回的列表中挑出out_channel为512,256,256,256的特征图层3、特征图层4、特征图层5、特征图层6
    # 目的是提取这些特征图层的out_channes
    for k, v in enumerate(extra_layers[1::2], start=2): # start表示k从几开始计数
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)


# vgg函数中的cfg参数
base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}
# add_extra函数中的cfg参数
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}
# multibox函数中的cfg参数
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [],
}


def build_ssd(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return
    # multibox函数返回值：vgg, extra_layers, (loc_layers, conf_layers)
    base_, extras_, head_ = multibox(vgg(base[str(size)], 3),
                                     add_extras(extras[str(size)], 1024),
                                     mbox[str(size)], num_classes)
    return SSD(phase, size, base_, extras_, head_, num_classes)
