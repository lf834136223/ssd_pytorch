from .config import HOME
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np


# xml文件解析
if sys.version_info[0] == 2: # 如果python版本为2
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

# note: if you used our download scripts, this should be right
VOC_ROOT = osp.join(HOME, "data/VOCdevkit/")


class VOCAnnotationTransform(object):
    """将xml文件按照此方法转换
    Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        # VOC_CLASSES是包含所有类别str的元组
        # zip函数构造了一个字典{'aeroplane':0, 'bicycle':1, 'bird':2, 'boat':3, ...}
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
            width,height:根据这两个参数决定是取bouding box的绝对值还是相对比例
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        # 参数target是通过xml.etree.ElementTree.parse().getroot()得到的xml树
        # 遍历的obj，就是背景框的信息
        # 例如	<object>
        # 		<name>aeroplane</name>
        # 		<pose>Frontal</pose>
        # 		<truncated>0</truncated>
        # 		<difficult>0</difficult>
        # 		<bndbox>
        # 			<xmin>104</xmin>
        # 			<ymin>78</ymin>
        # 			<xmax>375</xmax>
        # 			<ymax>183</ymax>
        # 		</bndbox>
        # 	</object>
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip() # name字符串为检测的类别之一，比如飞机
            bbox = obj.find('bndbox')

            positions = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            # 此循环将obj的四个坐标(绝对值或者相对比例)添加到bndbox列表中
            for i, position in enumerate(positions):
                # 取出xml文件中bndbox的xmin,ymin,xmax,ymax的具体数值 ==> cur_pt
                current_position = int(bbox.find(position).text) - 1
                # 对xmin和xmax除以宽，对ymin和ymax除以高
                # 如果传进来的width和height为图片的宽高，那么下面这句就是求bouding box的相对比例
                # 如果传进来的width和height为1，那么下面这句就是求bouding box的绝对值
                current_position = current_position / width if i % 2 == 0 else current_position / height
                bndbox.append(current_position)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]

        return res  # [[xmin, ymin, xmax, ymax, label_index], ... ]


class VOCDetection(data.Dataset):
    """
    Arguments:
        root (string):
            VOC格式的数据集路径.例如~/data/VOCdevkit/
        image_sets (string):
            哪些数据集？(例如： 'train', 'val', 'test')
        transform (callable, optional):
            对图片需要采取的处理方法
        target_transform (callable, optional):
            对标签采取的处理方法，默认为VOCAnnotationTransform，
            VOCAnnotationTransform方法通过xml文件，
            返回原图的tensor，和target [[xmin,ymin,xmax,ymax,label_id],...]
        dataset_name (string, optional):
            which dataset to load
    """

    def __init__(self, root,
                 transform=None,
                 image_sets=[('2007', 'trainval'),('2012', 'trainval')],  # 默认是使用trainval
                 target_transform=VOCAnnotationTransform(),
                 dataset_name='VOC2012'):
        root = root.replace('\\', '/')
        self.root = root
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()

        year, name = image_sets[1] # 使用VOC2012数据集
        rootpath = osp.join(self.root, 'VOC' + year)
        rootpath = rootpath.replace('\\', '/') # 源码的目录使用/，在windows系统里得到的是\\
        # 读取Imagesets/main/name.txt中的内容
        for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
            # 默认是trainval集，trainval集里每一行都是类似2008_000002
            # 但是比如train_val集里每一行都是2008_000002 -1
            # 那么在pull_item函数里的target = ET.parse(self._annopath % img_id).getroot()就会出错，需要注意
            self.ids.append((rootpath, line.strip())) # ids列表中元素为元组，每个元组存储了要训练的图片的路径以及名字
        # 如果既想使用VOC2007又想使用VOC2012，使用这一段代码
        # for (year, name) in image_sets:
        #     rootpath = osp.join(self.root, 'VOC' + year)
        #     for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
        #         self.ids.append((rootpath, line.strip()))

    def __getitem__(self, index):
        # 实例化后可以使用obj[index]
        # pull_item(index) ==> return torch.from_numpy(img).permute(2, 0, 1), target, height, width
        im, gt, h, w = self.pull_item(index)

        # __getiitem__方法会返回某一张图片的的tensor，
        # 以及此图片的ground truth ==> [[xmin,ymin,xmax,ymax,label_id],...]
        return im, gt

    def __len__(self):
        # 实际上就是选定的训练集or测试集...的长度
        return len(self.ids)

    def pull_item(self, index):
        # ids列表中元素为元组，每个元组存储了要训练的图片的路径以及名字
        img_id = self.ids[index]

        # getroot()方法将xml看作成一个类似树的对象，可以迭代出每一个子节点，子节点又可以有子节点，可以用[]方法，比如target[0][1][1].tag 或 .text 等等
        # self._annopath % img_id = ~/data/VOCdevkit/VOC2012/Annotations/name.xml，注意了，这里的%是占位符
        target = ET.parse(self._annopath % img_id).getroot()
        # self._annopath % img_id = ~/data/VOCdevkit/VOC2012/JPEGImages/name.jpg
        img = cv2.imread(self._imgpath % img_id) # opencv使用bgr格式读取图片
        height, width, channels = img.shape

        if self.target_transform is not None:
            # VOCAnnotationTransform类__call__函数返回 res = [[xmin, ymin, xmax, ymax, label_id], ... ]
            target = self.target_transform(target, width, height)

        # target_transform根本就不会为None
        # 所以下面target = self.target_transform(target, width, height)=[[xmin, ymin, xmax, ymax, label_id], ... ]
        if self.transform is not None:
            target = np.array(target)
            # img.shape >> (h,w,c)
            # boxes.shape >> [[xmin,ymin,xmax,ymax],
            #                 [xmin,ymin,xmax,ymax]]
            # labels.shape >> [label_id,...]行向量
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # opencv的bgr格式变成rgb格式，还可以直接这么变啊？
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1) 这是源码里写的
            # np.hstack((arg1,arg2)),将两个张量水平组合起来
            # 例如： t1 = np.array([[1,2],[3,4]])    t2 = np.array([[5,6],[7,8]])
            #        t = np.hstack((t1,t2))  t = array([[1, 2, 5, 6],[3, 4, 7, 8]])
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        # 返回值：
        # 原图tensor 与numpy的(h,w,c)格式不同，pytorch的为(c,h,w)
        # 标签traget [[xmin,ymin,xmax,ymax,label_id],...]
        # 原图的高height
        # 原图的宽width
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width
        # return torch.from_numpy(img), target, height, width

    def pull_image(self, index):
        '''
            读取指定index的图片的numpy.ndarray格式矩阵
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''
            返回:
            1、该图片的名字，比如2007_000027
            2、一张图片中所有目标obj的bouding box坐标值和类别数字(绝对值)列表：[[xmin, ymin, xmax, ymax, label_id], ... ]

        '''
        # ids = [('~/data/VOCdevkit/VOC2012','2007_000027'),...]
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        # width和height为1，找到VocAnnotationTransform的__call__函数就知道是返回bouding box的绝对坐标值
        gt = self.target_transform(target=anno, width=1, height=1)
        return img_id[1], gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: 不要将这个函数使用到self.__getitem__()里, as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)
