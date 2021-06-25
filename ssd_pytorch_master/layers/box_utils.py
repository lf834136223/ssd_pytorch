# -*- coding: utf-8 -*-
import torch


def point_form(boxes):
    """
        生成的defult box的格式是(cx,cy,width,height)
        point_form函数把default box的格式改成(xmin,ymin,xmax,ymax)
        这样就和gt box的格式一样了
    Args:
        boxes: (tensor) 格式为[cx,cy,width,height](相对比例)default boxes
    """
    # boxes.shape = [8732,4]
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,         # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), dim=1)  # xmax, ymax


def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, 2:] + boxes[:, :2])/2,  # cx, cy
                     boxes[:, 2:] - boxes[:, :2], 1)  # w, h


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)  # num_objects
    B = box_b.size(0)  # num_priors 8732
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(dt_box, gt_box):
    """Compute the jaccard overlap(交并比) of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) gt box Shape: [num_objects,4]
        box_b: (tensor) dt box Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(dt_box, gt_box)  # 两个box交集
    # area.shape =
    area_a = ((dt_box[:, 2]-dt_box[:, 0]) *
              (dt_box[:, 3]-dt_box[:, 1])).unsqueeze(1).expand_as(inter)
    area_b = ((gt_box[:, 2]-gt_box[:, 0]) *
              (gt_box[:, 3]-gt_box[:, 1])).unsqueeze(0).expand_as(inter)
    union = area_a + area_b - inter  # 并集
    return inter / union  # IoU


def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    """
    输入:
        threshold:(int) 阈值
        truths:(Tensor) gt boxes  shape = [num_objects,4]
        priors:(Tensor) dt boxes, shape = [num_objects,4] 存储的是相对比例
        variances:(list) dt boxes回归时需要用到的参数
        labels:(Tensor) gt boxes的类别标签, [num_objects,1].
        loc_t: (Tensor) 存储匹配后default boxes的回归系数 [batch, 8732, 4]
        conf_t: (Tensor) 存储匹配后各default boxes的真实类别标记 [batch, 8732]
        idx: (int) 当前图片在batch中的索引
    返回:
        函数本身不返回值，但它会把匹配框的位置和置信信息存储在loc_t, conf_t两个tensor中。
    目的:
        最终的目标是给每个default_box匹配到一个gt box
    """

    # overlaps.shape = [truths.shape[0],priors.shape[0]]
    # 每一个gt box与和每一个dt box的IoU
    overlaps = jaccard(
        truths, # gt box [batch_size,4]
        point_form(priors) # dt box转成(xmin,ymin,xmax,ymax)格式
    )

    '''
        Tensor.max()函数在指定维度上得到该维度的最大值，并得到该最大值的下标，以Tensor返回,可以用两个变量接收这两个Tensor
        eg:
            t1 = torch.Tensor([[1,3,5,7],[5,2,4,8]])
            a,b = t1.max(1,keepdim=True) >> a = Tensor([7,8]),b = Tensor([3,3])
            c,d = t1.max(0,keepdim=True) >> c = Tensor([5,3,5,8]),d = Tensor([1,0,0,1])
    '''
    # 与背景框匹配度最高的先验框 这部分一定是正样本
    # best_prior_overlap.shape and best_prior_idx.shape= [1,num_objects]
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    # 与先验框匹配度最高的背景框
    # best_truth_overlap.shape and best_truth_idx.shape = [1,num_priors]
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)

    # squeeze_ 只有维度的值等于1才会被裁剪
    best_truth_idx.squeeze_(0)  # shape由[1,num_priors] >> [num_priors]
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)  # shape由[num_objects,1] >> [num_objects]
    best_prior_overlap.squeeze_(1)

    # 确保最好的prior boxes不会因为阈值太低而被删除掉
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)
    # 保证每一个ground truth 匹配它的都是具有最大IOU的prior
    for j in range(best_prior_idx.size(0)):  # 0,1,...,num_objects
        best_truth_idx[best_prior_idx[j]] = j # j在这里的含义其实就是第j个gt box

    # truths.shape = [num_objects, 4]
    # matches.Shape = [num_priors,4]
    matches = truths[best_truth_idx]
    # loc.shape = [num_priors,4] 其中4包含了priors中cx,cy,w,h的回归系数
    loc = encode(matches, priors, variances)

    # conf.Shape = [num_priors]
    # conf为8732个priors的labels
    # 真实标签从0到sum_classes，所以需要加1，把0标签给背景类别
    conf = labels[best_truth_idx] + 1
    # 与truth的IoU小于threshold的priors的label置为背景类别0
    conf[best_truth_overlap < threshold] = 0

    loc_t[idx] = loc    # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior


def encode(matched, priors, variances):
    """
        找到了所有default box对应的GT box的四个角的坐标(也就是传入的matched参数)，就可以开始进行边框偏移(平移加缩放)的计算，
        在SSD中此处有一个技巧，假设已知priors的位置p=(pcx,pcy,pw,ph)，
        以及它对应的GT box的位置m=(mcx,mcy,mw,mh)，通常的边框偏移是按照如下方式计算的
        # d_cx = (mcx-pcx)/(pw*v[0])   平移系数
        # d_cy = (mcy-pcy)/(ph*v[0])   平移系数
        # d_w = (log(mw/pw))/v[1]      缩放系数
        # d_h = (log(mh/ph))/v[1]      缩放系数
    Args:
        matched: (tensor) xmin,ymin,xmax,ymax格式 是比例而不是绝对值
            Shape: [num_priors, 4].
        priors: (tensor) cx,cy,w,h格式 是比例而不是绝对值
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes [0.1,0.2]
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """

    """priors框的回归参数"""
    d_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
    # encode variance
    d_cxcy /= (variances[0] * priors[:, 2:])
    # match wh / prior wh
    d_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    d_wh = torch.log(d_wh) / variances[1]
    # return target for smooth_l1_loss
    # 返回值的shape = [num_priors,4],其中4包含了priors中cx,cy,w,h的回归系数
    return torch.cat([d_cxcy, d_wh], 1)


# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    # torch.log以e为底
    # tor.exp是e的指数函数
    # torch.sum(torch.exp(x-x_max), 1, keepdim=True)保持维度不变，在第一个维度上取指数函数的和
    #return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max
    res = torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max
    return res


"""去除重复检测框"""
def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) 存储一个图片的所有预测框, Shape: [num_priors,4].
        scores: (tensor) 置信度。如果为多分类则需要将nms函数套在一个循环内, Shape:[num_priors].
        overlap: (float) iou的阈值
        top_k: (int) 选取置信度前top_k个框再进行nms.
    Return:
        nms后剩余预测框的索引.
    """

    # 函数new(): 构建一个有相同数据类型的tensor
    # .zero_()：元素全置为0
    # .long(): 元素置为int64  例：[2.1,3.4,5.99] ==> [2,3,5]
    keep = scores.new(scores.size(0)).zero_().long() # keep用来保存留下来的box的索引 [num_positive]
    # tensor的元素个数
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0] # 所有box的第一列 x1.shape=[1,num_priors]
    y1 = boxes[:, 1] # 这种索引方法是pytorch独有的，与原生Python不一样
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1) # 乘法，并行化计算所有框的面积
    # Tensor.sort(dim)：dim未指定时候按最后一个维度升序排序，返回一个(values,indices)也是Tensor格式
    v, idx = scores.sort(0)
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # 从后选取top_k个索引
    xx1 = boxes.new() # boxes.shape = [num_priors,4]
    yy1 = boxes.new() # 注意：new方法在不指定维度是创建一个维度为0的空Tensor
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    # len(idx) = top_k
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        keep[count] = i  # 将置信度最大的下标添加到keep keep.shape = [num_priors]
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # 左闭右开，将idx最后一个值移除
        # torch.index_select(input, dim, index, out=None)
        # input (Tensor) – 输入张量
        # dim (int) – 索引的轴
        # index (LongTensor) – 一维张量，在dim维度上的索引
        # out (Tensor, optional) – 目标张量
        # 在x1,x2,x3,x4中挑出idx个值
        torch.index_select(x1, 0, idx, out=xx1)  # load bboxes of next highest vals
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)

        # 计算当前最大置信框与其他剩余框的交集
        # i为当前idx中score值最大的索引
        # 这一段代码以后看不懂了就画图
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1 # w,h存储了除最大置信度的top_k-1个框与最大置信度框交并的宽，高值
        h = yy2 - yy1

        # 修正下w,h中可能会有的负值
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        # inter：其他box与最大置信度box的交并
        inter = w*h #  Tensor中对应元素相乘，torch.dot(tensor1,tensor2)：积的和 |
        # IoU = i / (area(a) + area(b) - i)
        # 将置信度最高的top_k-1个框的面积存储在rem_areas
        rem_areas = torch.index_select(area, 0, idx)  # area:所有box的面积
        union = (rem_areas - inter) + area[i] #  union:两个框的合集
        IoU = inter/union  # 将交并比存储在IoU中

        # idx中仅留下IoU <= overlap的值
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)] # 这种切片方式也是Pytorch独有
    return keep, count