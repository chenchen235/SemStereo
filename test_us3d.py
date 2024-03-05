# from __future__ import print_function, division
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import time
from tensorboardX import SummaryWriter
from datasets import __datasets__
# from models import __models__, model_loss_train_attn_only, model_loss_train_freeze_attn, model_loss_train, model_loss_test
from models import __models__, model_loss_train, model_loss_test, model_label_loss
from utils import *
from torch.utils.data import DataLoader
import gc
# from apex import amp
import cv2

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='SemStereo: Semantic-Constrained Stereo Matching Network for Remote Sensing')

parser.add_argument('--model', default='CS2_Net', help='select a model structure', choices=__models__.keys())

parser.add_argument('--maxdisp', type=int, default=64, help='maximum disparity')
parser.add_argument('--num_classes', type=int, default=6, help='number of classes')
parser.add_argument('--attention_weights_only', default=False, type=str,  help='only train attention weights')

parser.add_argument('--dataset', default='us3d', help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', default="../data/us3d/JAX", help='data path')
parser.add_argument('--testlist',default='../data/us3d/JAX/test.txt', help='testing list')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--loadckpt', 
                    default='../checkpoints/lr/all/checkpoint_000047.ckpt',help='load the weights from a specific checkpoint')

parser.add_argument('--seg_if', default=True, type=str,  help='only train attention weights')
parser.add_argument('--stereo_if', default=True, type=str,  help='only train attention weights')

# parse arguments, set seeds
args = parser.parse_args()

# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
test_dataset = StereoDataset(args.datapath, args.testlist, False)
TestImgLoader = DataLoader(test_dataset, args.test_batch_size, shuffle=False, num_workers=16, drop_last=False)

# model, optimizer
model = __models__[args.model](args.maxdisp, args.attention_weights_only, args.seg_if, args.stereo_if, args.num_classes)

model = nn.DataParallel(model)
model.cuda()

# load parameters
print("loading model {}".format(args.loadckpt))
state_dict = torch.load(args.loadckpt)
model.load_state_dict(state_dict['model'])

def test():
    avg_test_scalars = AverageMeterDict()
    avg_test_scalars2 = AverageMeterDict2()
    
    for batch_idx, sample in enumerate(TestImgLoader):    
        start_time = time.time()
        # loss, scalar_outputs = test_sample(sample)
        loss, disp_loss, label_loss, scalar_outputs, scalar_outputs2 = test_sample(sample)
        avg_test_scalars.update(scalar_outputs)
        avg_test_scalars2.update(scalar_outputs2)
        del scalar_outputs
        print('Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(batch_idx,
                                                                    len(TestImgLoader), loss,
                                                                    time.time() - start_time))
        
    avg_test_scalars = avg_test_scalars.mean()
    avg_test_scalars2 = avg_test_scalars2.mean()
    
    print("avg_test_scalars", avg_test_scalars,"avg_test_scalars2", avg_test_scalars2)


nums = args.num_classes

# test one sample
@make_nograd_func
def test_sample(sample,compute_metrics=True):
    model.eval()
    metric = SegmentationMetric(nums-1)
    
    imgL, imgR, disp_gt,label_true = sample['left'], sample['right'], sample['disparity'], sample['label']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()
    label_true = label_true.cuda()
    mask = (disp_gt < args.maxdisp) & (disp_gt >= -args.maxdisp)
    masks = [mask]
    disp_ests, label_est = model(imgL, imgR)
    
    metric.addBatch(label_est, label_true)

    disp_gts = [disp_gt]
    
    # label_true_l = [label_true] # mutil 
    # label_est_l = [label_est]
    #label_loss = model_label_loss(label_est_l, label_true_l, nums, args.attention_weights_only)

    disp_loss = model_loss_test(disp_ests, disp_gts, masks)
    label_loss = model_label_loss(label_est, label_true, nums, args.attention_weights_only)
    
    loss = disp_loss + label_loss
    scalar_outputs = {"loss": loss, "disp_loss":disp_loss, "label_loss":label_loss}
    scalar_outputs2 = {}
    scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
    scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
    scalar_outputs["PA"] = [metric.pixelAccuracy()]
    scalar_outputs["MPA"] = [metric.meanPixelAccuracy()]
    scalar_outputs["mIoU"] = [metric.meanIntersectionOverUnion()]
    for i in range(nums - 1):
        scalar_outputs2["CPA" + str(i)] = [metric.classPixelAccuracy()[i]]
        scalar_outputs2["IoU" + str(i)] = [metric.IoU()[i]]
    return tensor2float(loss), tensor2float(disp_loss), tensor2float(label_loss), tensor2float(scalar_outputs), scalar_outputs2

if __name__ == '__main__':
    test()
