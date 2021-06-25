# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data import coco as cfg
from ..box_utils import match, log_sum_exp


class MultiBoxLoss(nn.Module):
    """
        正样本: 1、对每个GT box匹配得到的IoU最大的那些prior box，算做正样本
                2、对每个prior box而言，只要它与任何一个GTbox的IoU值大于threshold，则算正样本

        负样本：对除开正样本以外的样本计算置信度损失，选取置信度损失最高的，
               按照负样本:正样本为3:1的ratio选取负样本

    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 use_gpu=True):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos # 负/正样本比例
        self.neg_overlap = neg_overlap
        self.variance = cfg['variance']

    def forward(self, predictions, targets):
        """
        重点难点是挑选出正负样本
        predictions:元组,包含loc,conf,priors
            loc.shape = [batch_size,8732,4] SSD网络预测出来的坐标偏移量
            conf.shape = [batch_size,8732,num_classes+1] SSD网络预测出来的类别分数
            priors.shape = [8732,4] 这里的4是一个default box的中心点的坐标，宽高坐标(但都是相对于原图的比例)
        targets:(list)  [[xmin,ymin,xmax,ymax,label_id],...]
                targets列表中有batch_size个gt Tensor
                len(targets) = batch_size
        """

        # loc_data.shape = [batch_size, 8732, 4]
        # conf_data.shape = [batch_size, 8732, 21]
        # priors.shape = [8732,4]   x,y,w,h
        loc_data, conf_data, priors = predictions
        batch_size = loc_data.size(0)
        # prior = priors[:loc_data.size(1), :] # 为什么需要这一句？？完全没有变化
        # print(priors==prior)
        num_priors = (priors.size(0)) # num_priors = 8732
        num_classes = self.num_classes # num_classes = 21

        # loc_t：与batch中所有图片的dt box相匹配的gt box的经过encode的回归系数
        loc_t = torch.Tensor(batch_size, num_priors, 4)
        # loc_t：与batch中所有图片的dt box相匹配的gt box的label
        conf_t = torch.LongTensor(batch_size, num_priors)

        # 经过此循环
        # loc_t.shape = [batch_size,num_priors,4] 其中4代表了每一个priors的边框回归系数
        # conf_t.shape = [batch_size,num_priors] 第二个维度代表了每一个priors的label
        for idx in range(batch_size):
            truths = targets[idx][:, :-1].data  # 一张图片中所有gt box的坐标 [num_objects,4]
            labels = targets[idx][:, -1].data  # 一张图片中所有gt box的label_id [num_object,1]
            defaults = priors.data  # default box的坐标(与原图相对比例)

            # threshold:阈值
            # match函数本身没有返回值
            match(self.threshold, truths, defaults, self.variance, labels,
                  loc_t, conf_t, idx)
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()

        # wrap targets
        # loc_t是计算得来的priors移动到其对应的gt box所需要的平移和缩放系数
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)

        '''
            只计算正样本的位置损失
        '''
        # pos.dtype = bool true表示正样本
        # pos.shape = [batch_size,8732]
        pos = conf_t > 0  # mask
        num_pos = pos.sum(dim=1, keepdim=True) # True=1,False=0

        # 位置损失 (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data) # pos.dim()=2 使用loc_data的shape扩充pos

        loc_p = loc_data[pos_idx].view(-1, 4)  # 取出loc_data中正样本的坐标 这是网络预测出来的
        loc_t = loc_t[pos_idx].view(-1, 4)  # 取出loc_t中正样本的坐标 这是encode函数计算出来的边框回归系数
        # 所以说SSD网络在位置上需要学习的参数就是priors的平移与缩放系数
        # 位置损失只针对正样本
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        '''
            进行负样本的挖掘，并根据这些正负样本计算类别分数损失
            负样本时如何挖掘的:
            1、conf_t的shape为[batch_size,num_priors,1],具体值代表了batch_size张图片所有priors所匹配的dt box对应的label，
               且conf_t中大部分值应该为0(对应背景标签),将conf_t reshape成[batch_size*num_priors,1]
            2、conf_data的shape为[batch_size,num_priors,num_classes+1]，具体值代表了所有priors预测所有类别的得分
               reshape成[batch_size*num_priors,num_classes+1]
            3、利用conf_t(具体值为标签可以利用到conf_data上作为索引取出priors匹配的dt box的标签的得分)，根据logsoftmax公式计算
            4、根据logsoftmax计算得到的值(均为负值)进行取反后越大(仅在负样本中比较)说明网络将该priors预测为背景的可能性越小(以后看不懂就画log函数图像理解)
               而负样本是要将priors预测为背景类别的，就越不可以接受。故降序排列后取前Top-k个作为负样本
        '''

        # 这里使用的conf_data是网络预测出来的置信度
        # batch_conf.shape = [batch_size*num_priors,21]
        batch_conf = conf_data.view(-1, self.num_classes)

        # log_sum_exp(batch_conf) = torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max
        # log_sum_exp(batch_conf).shape = [batch_size*num_priors,1]
        # conf_t.view(-1, 1).shape = [batch_size*num_priors,1]
        loss_c = -(batch_conf.gather(dim=1, index=conf_t.view(-1, 1)) - log_sum_exp(batch_conf))

        # shape = [batch_size,num_priors] 每一个具体值代表了一个prior所匹配标签的置信度损失(-logsoftmax，越大表示预测为背景的可能性小)
        loss_c = loss_c.view(batch_size, -1)
        loss_c[pos] = 0  # 将正样本在loss_c中过滤掉 pos = conf_t > 0

        # 两次sort排序，能够得到idx_rank：每个元素在降序排列中的位置idx_rank 暂时不理解为什么两次排序可以达到这个效果
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)  # idx_rank.shape = [batch_size,num_priors]

        # 正样本数 shape = [batch_size,1]
        num_pos = pos.long().sum(1, keepdim=True)
        # 负样本数 shape = [batch_size,1]
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1) # 这里的max设为8731,就是保险一下，正样本数目一般很少
        # num_neg.shape = [batch_size,1]          idx_rank.shape = [batch_size,num_priors]
        # 抽取前num_neg个负样本
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # pos_idx.shape = [batch_size,num_priors,21]
        # pos_idx第三个维度的值都true
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        # pos_idx.shape = [batch_size,num_priors,21]
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)

        ''' 核心目的，根据正负样本的idx在conf_t(网络预测)和conf_data(实际)提取正负样本的label
            维度十分混乱,所以在这里把理解下面两句时用到的变量列举出来
            conf_t.shape = [batch_size,num_priors,1] 保存了batch张图片所有priors的实际label
            loc_t.shape = [batch_size,num_priors,4] 保存了batch张图片所有priors的预测回归系数
            conf_data.shape = [batch_size,num_priors,21] 保存了batch张图片所有priors的预测label
            
            loss_c.shape = [batch_size,num_priors] 每一行代表一张图片的priors对应dt box的类别的置信度损失(不记得就去看上面)
            
            idx_rank.shape = [batch_size,num_priors] 每一行代表loss_c按降序排列后，所有元素在loss_c的原索引
            
            pos.shape = [batch_size,num_priors] 每一行都为bool值,True代表该prior为正样本，用来在conf_t上索引
            neg.shape = [batch_size,num_priors] 每一行都为bool值,True代表该prior为负样本，用来在conf_t上索引
            
            pos_idx.shape = [batch_size,num_priors,21] 第三维度的21个值同为True或False，用来在conf_data上索引
            neg_idx.shape = [batch_size,num_priors,21] 第三维度的21个值同为True或False，用来在conf_data上索引
            
            conf_p.shape = [batch_size*(num_pos+num_neg),21] batch_size张图片所有priors的正负样本的类别得分
            targets_weighted.shape = [batch_size*(num_pos+num_neg)] batch_size张图片所有priors的正负样本的实际label
        '''
        # torch.gt(a,b) 逐个比较a,b的元素，在某个元素a>b，则返回1，a<b，则返回0
        # Tensor1[Tensor2]这种索引方式会把取出来的值放在一个维度为1的Tensor里
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]

        #
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        N = num_pos.data.sum()  # N为正样本个数
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c
