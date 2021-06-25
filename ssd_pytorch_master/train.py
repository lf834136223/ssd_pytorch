from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')
parser.add_argument('--dataset_root', default=VOC_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=2, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=False, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--visdom', default=True, type=str2bool,
                    help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()


# 使用GPU
if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

# 创建保存权重文件夹
if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

def train():
    # 创建coco格式的dataset
    if args.dataset == 'COCO':
        if args.dataset_root == VOC_ROOT:
            if not os.path.exists(COCO_ROOT):
                parser.error('Must specify dataset_root if specifying dataset')
            print("WARNING: Using default COCO dataset_root because " +
                  "--dataset_root was not specified.")
            args.dataset_root = COCO_ROOT
        cfg = coco
        dataset = COCODetection(root=args.dataset_root,
                                transform=SSDAugmentation(cfg['min_dim'],
                                                          MEANS))

    # 创建voc格式的dataset
    elif args.dataset == 'VOC':
        if args.dataset_root == COCO_ROOT:
            parser.error('Must specify dataset if specifying dataset_root')
        cfg = voc # 配置信息

        # VOCDetection类的__getitem__()以元组形式返回(im,gt)
        # 其中im为图片的张量，gt = [[xmin,ymin,xmax,ymax,label_id],...]
        dataset = VOCDetection(root=args.dataset_root,
                               transform=SSDAugmentation(cfg['min_dim'],MEANS), # 对图片的处理
                               target_transform=VOCAnnotationTransform() # [[xmin, ymin, xmax, ymax, label_idx], ... ]
                              )

    # 关于DataLoader的自己的理解：
    # 应该是在DataLoader内部会获得batch_size个DataSet的__getitem__函数的返回值(Tensor img,gt)
    # 如果要对这batch_size个元组数据进行处理，就在collate_fn函数里进行处理(例子可以参考下面的这个)
    # 得到一个迭代器容器，每一个迭代里是batch_size张图片的(Image Tensors,list of targets)
    data_loader = data.DataLoader(dataset, batch_size=args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, pin_memory=True,
                                  collate_fn=detection_collate)

    # # 自己加的测试代码
    # for i,item in enumerate(iter(data_loader)):
    #     if i==0:
    #         print(item[1])
    #         break

    if args.visdom:
        import visdom
        global viz
        viz = visdom.Visdom()

    # 创建网络
    ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
    net = ssd_net

    # 多块GPU并行训练
    if args.cuda:
        net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True

    # 如果有完整的模型文件则加载整个模型
    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_weights(args.resume)
    # 没有则加载vgg的权重文件
    else:
        # args.save_folder, default='weights/'
        # args.basenet, default='vgg16_reducedfc.pth'
        vgg_weights = torch.load(args.save_folder + args.basenet)
        print('Loading base network...')
        ssd_net.vgg_layers.load_state_dict(vgg_weights)

    # 为什么有时候用的ssd_net，有时候用的是net？
    if args.cuda:
        net = net.cuda()

    # 除了vgg网络以外其他参数的初始化
    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extra_layers.apply(weights_init)
        ssd_net.loc_layers.apply(weights_init)
        ssd_net.conf_layers.apply(weights_init)

    # 优化器
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    # 损失函数
    criterion = MultiBoxLoss(num_classes=cfg['num_classes'],
                             overlap_thresh=0.5,
                             prior_for_matching=True,
                             bkg_label=0,
                             neg_mining=True,
                             neg_pos=3,
                             neg_overlap=0.5,
                             encode_target=False,
                             use_gpu=args.cuda)

    net.train()
    # loss counters
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    print('Loading the dataset...')

    epoch_size = len(dataset) // args.batch_size # 每一个epoch中有多少个batch
    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)

    step_index = 0

    # 训练可视化
    if args.visdom:
        vis_title = 'SSD.PyTorch on ' + dataset.name
        vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
        iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)

    # 构建循环开始训练
    batch_iterator = iter(data_loader) # 批次迭代器
    # default start_iter:0,接着以前的工作继续训练
    # default max iter:120000，在config.py文件中修改
    for iteration in range(args.start_iter, cfg['max_iter']): # epochs
        # 这一段代码不知道作用，只知道是画图
        if args.visdom and iteration != 0 and (iteration % epoch_size == 0):
            update_vis_plot(epoch, loc_loss, conf_loss, epoch_plot, None,
                            'append', epoch_size)
            # reset epoch loss counters
            loc_loss = 0
            conf_loss = 0
            epoch += 1

        # 'lr_steps': (80000, 100000, 120000)  每隔指定步数调整学习率
        if iteration in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

        # dataset的__getitem__得到返回值im,gt
        images, targets = next(batch_iterator)

        # 将图片以及标签送入cpu或者gpu
        if args.cuda:
            images = Variable(images.cuda())
            with torch.no_grad():
                targets = [Variable(ann.cuda()) for ann in targets]
        else:
            images = Variable(images)
            with torch.no_grad():
                targets = [Variable(ann) for ann in targets]

        # 前向传播
        t0 = time.time()
        # out是一个元组包含loc,conf,priors
        # loc.shape = [batch_size,8732,4] 预测出来的default box的偏移量
        # conf.shape = [batch_size,8732,num_classes+1] 预测出来的default box的类别
        # priors.shape = [8732,4] 这里的4是一个default box的中心点的坐标，宽高坐标(但都是相对于原图的比例)
        out = net(images)
        # 反向传播
        optimizer.zero_grad()
        # https://blog.csdn.net/goodxin_ie/article/details/89675756 讲损失函数的博客
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()
        t1 = time.time()
        #loc_loss += loss_l.data[0]  # 0.5以上的版本会报错
        #conf_loss += loss_c.data[0]
        loc_loss += loss_l.item()
        conf_loss += loss_c.item()

        # 每十个epoch打印一次训练信息
        if iteration % 10 == 0:
            print('timer: %.4f sec.' % (t1 - t0))
            # str():返回字符串本身
            # repr():返回对象的更多机器信息
            #print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.data[0]), end=' ')
            # .data返回的是一个tensor
            # 而.item()返回的是一个具体的数值。
            print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.item()), end=' ')

        # 更新训练可视化图表信息
        if args.visdom:
            update_vis_plot(iteration, loss_l.item(), loss_c.item(),
                            iter_plot, epoch_plot, 'append')

        # 每5000个epoch保存一个检查点
        save_iter_flag = 5000
        if iteration != 0 and iteration % save_iter_flag == 0:
            print('Saving state, iter:', iteration)
            torch.save(ssd_net.state_dict(), 'weights/ssd300_VOC_' +
                       repr(iteration) + '.pth')

    # 保存整个模型的权重文件
    torch.save(ssd_net.state_dict(),
               args.save_folder + '' + args.dataset + '.pth')


"""
    训练过程中更改学习率
"""
def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

"""
    xavier方法初始化卷积层参数
    偏置值置初始化为0
"""
def xavier(param):
    init.xavier_uniform(param)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


"""
    使用visdom库绘制训练信息
"""
def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )

def update_vis_plot(iteration, loc, conf, window1, window2, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration
    if iteration == 0:
        viz.line(
            X=torch.zeros((1, 3)).cpu(),
            Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
            win=window2,
            update=True
        )


if __name__ == '__main__':
    train()
