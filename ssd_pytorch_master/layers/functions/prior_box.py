from __future__ import division
from math import sqrt as sqrt
from itertools import product as product
import torch




class PriorBox(object):
    """
    PriorBox类用来生成一开始所需要的先验框
    参数：
    min_dim = 300
    	"输入图最短边的尺寸"

    feature_maps = [38, 19, 10, 5, 3, 1]
    steps = [8, 16, 32, 64, 100, 300]
    	"共有6个特征图：
    	由于steps由于网络结构所以为固定，所以作者应该是由300/steps[k]得到feature_maps"

    min_sizes = [30, 60, 111, 162, 213, 264]
    max_sizes = [60, 111, 162, 213, 264, 315]
    这两个size根据SK=Smin+[(Smax-Smin)/m-1]*(k-1) k从1~m计算而来

    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    	"各层除1以外的aspect_ratios，可以看出是各不相同的，
    	这样每层特征图的每个像素点分别有[4,6,6,6,4,4]个default boxes
    	这个可以根据自己的场景适当调整"

    """
    def __init__(self, cfg): # cfg应该是config.py文件中的
        super(PriorBox, self).__init__()
        self.image_size = cfg['min_dim']
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps'] # 特征图的尺寸
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        self.version = cfg['name'] # 数据格式的名字 voc or coco
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    @property
    def forward(self):
        mean = []
        # 返回所有先验框的中心坐标比例与宽高比例(相对原图): [8732,4]
        for k, f in enumerate(self.feature_maps): # feature_maps = [38, 19, 10, 5, 3, 1]
            # product函数生成i,j =  [(0,0),(0,1)...(0,f-1)
            #                       (1,0),(1,1)...(1,f-1)
            #                        ...................
            #                        .                 .
            #                        ...................
            #                       (f-1,0),(f-1,1)...(f-1,f-1)]
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k] # steps = [8, 16, 32, 64, 100, 300]
                # cx,cy为每个像素点的中心比例(相对原图)
                cx = (j + 0.5) / f_k # 注意了这都是比例，而不是实际的坐标
                cy = (i + 0.5) / f_k

                # aspect_ratio: 1
                # s_k就是公式中的sk
                # rel_size:min_size
                s_k = self.min_sizes[k]/self.image_size
                mean += [cx, cy, s_k, s_k]

                #min_sizes = [30, 60, 111, 162, 213, 264]     这两个尺寸都是相对于输入图像300而言
                #max_sizes = [60, 111, 162, 213, 264, 315]    在特征图上应该除以300做映射

                # aspect_ratio: 1
                # s_k_prime就是公式中的sk'
                # s_k_prime: sqrt(s_k * s_(k+1))
                # rel_size: sqrt(min_size*max_size)
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # 除开1剩下的高宽比
                # rel size: min_size
                #aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)

        # a = torch.randn(4)
        # ==> tensor([-1.7120,  0.1734, -0.0478, -0.0922])
        # torch.clamp(a, min=-0.5, max=0.5)
        # ==> tensor([-0.5000,  0.1734, -0.0478, -0.0922])
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
