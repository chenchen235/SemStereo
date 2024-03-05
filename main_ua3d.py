# from __future__ import print_function, division
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5'
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
from models import __models__, model_loss_train, model_loss_test, model_label_loss, LRSC_loss
from utils import *
from torch.utils.data import DataLoader
import gc
import cv2
import sys
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


class Logger1(object):
    def __init__(self, filename='default.log', add_flag=True, stream=sys.stdout):
        self.terminal = stream
        print("filename:", filename)
        self.filename = filename
        self.add_flag = add_flag
        # self.log = open(filename, 'a+')

    def write(self, message):
        if self.add_flag:
            with open(self.filename, 'a+') as log:
                self.terminal.write(message)
                log.write(message)
        else:
            with open(self.filename, 'w') as log:
                self.terminal.write(message)
                log.write(message)

    def flush(self):
        pass

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='SemStereo: Semantic-Constrained Stereo Matching Network for Remote Sensing')
parser.add_argument('--model', default='SemStereo', help='select a model structure', choices=__models__.keys())

parser.add_argument('--maxdisp', type=int, default=64, help='maximum disparity')
parser.add_argument('--num_classes', type=int, default=6, help='number of classes')
parser.add_argument('--dataset', default='us3d', help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', default="../data/us3d/JAX", help='data path')
parser.add_argument('--trainlist', default='../data/us3d/JAX/train.txt', help='training list')
parser.add_argument('--testlist',default='../data/us3d/JAX/test.txt', help='testing list')
# parser.add_argument('--datapath', default="../data/us3d/OMA", help='data path')
# parser.add_argument('--trainlist', default='../data/us3d/OMA/train.txt', help='training list')
# parser.add_argument('--testlist',default='../data/us3d/OMA/test.txt', help='testing list')
parser.add_argument('--resume', action='store_true', help='continue training the model')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--summary_freq', type=int, default=50, help='the frequency of saving summary')
parser.add_argument('--save_freq', type=int, default=4, help='the frequency of saving checkpoint')

parser.add_argument('--attention_weights_only', default=False, type=str,  help='only train attention weights')
parser.add_argument('--seg_if', default=True, type=str,  help='only train attention weights')
parser.add_argument('--stereo_if', default=True, type=str,  help='only train attention weights')

parser.add_argument('--lr', type=float, default=0.001, help='base learning rate')
parser.add_argument('--batch_size', type=int, default=4, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=4, help='testing batch size')

parser.add_argument('--epochs', type=int, default=48, help='number of epochs to train')
parser.add_argument('--lrepochs',default="12,22,30,38,44:2", type=str, help='the epochs to decay lr: the downscale rate')
parser.add_argument('--loadckpt', default='../checkpoints/lr/only/checkpoint_000047.ckpt',help='load the weights from a specific checkpoint')
parser.add_argument('--logdir',default='../checkpoints/lr/all', help='the directory to save logs and checkpoints')
args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
os.makedirs(args.logdir, exist_ok=True)

# create summary logger
print("creating new summary file")
logger = SummaryWriter(args.logdir)

# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
train_dataset = StereoDataset(args.datapath, args.trainlist, True)
test_dataset = StereoDataset(args.datapath, args.testlist, False)
TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=16, drop_last=True)
TestImgLoader = DataLoader(test_dataset, args.test_batch_size, shuffle=False, num_workers=16, drop_last=False)

# model, optimizer
model = __models__[args.model](args.maxdisp, args.attention_weights_only, args.seg_if, args.stereo_if, args.num_classes)

model = nn.DataParallel(model)
model.cuda()
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
# load parameters
start_epoch = 0
if args.resume:
    # find all checkpoints file and sort according to epoch id
    all_saved_ckpts = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
    all_saved_ckpts = sorted(all_saved_ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # use the latest checkpoint file
    loadckpt = os.path.join(args.logdir, all_saved_ckpts[-1])
    print("loading the lastest model in logdir: {}".format(loadckpt))
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    start_epoch = state_dict['epoch'] + 1
elif args.loadckpt:
    # load the checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model_dict = model.state_dict()
    pre_dict = {k: v for k, v in state_dict['model'].items() if k in model_dict}
    model_dict.update(pre_dict) 
    model.load_state_dict(model_dict)
print("start at epoch {}".format(start_epoch))

nums = args.num_classes

def train():

    for epoch_idx in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch_idx, args.lr, args.lrepochs)

        # training
        for batch_idx, sample in enumerate(TrainImgLoader):
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            start_time = time.time()
            do_summary = global_step % (args.summary_freq * 1000) == 0
            loss, disp_loss, label_loss, label_r_loss, scalar_outputs = train_sample(sample, compute_metrics=do_summary)

            if do_summary:
                save_scalars(logger, 'train', scalar_outputs, global_step)
            del scalar_outputs

            print('Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, disp_loss = {:.3f}, label_loss = {:.3f}, label_r_loss = {:.3f}, time = {:.3f}'.format(epoch_idx, args.epochs,
                                                                                                   batch_idx,
                                                                                                   len(TrainImgLoader), loss, disp_loss, label_loss, label_r_loss, 
                                                                                                   time.time() - start_time))

        # saving checkpoints

        if (epoch_idx + 1) % args.save_freq == 0:
            checkpoint_data = {'epoch': epoch_idx, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            #id_epoch = (epoch_idx + 1) % 100
            torch.save(checkpoint_data, "{}/checkpoint_{:0>6}.ckpt".format(args.logdir, epoch_idx))
        gc.collect()

        # # testing
        avg_test_scalars = AverageMeterDict()
        avg_test_scalars2 = AverageMeterDict2()
        for batch_idx, sample in enumerate(TestImgLoader):

            global_step = len(TestImgLoader) * epoch_idx + batch_idx
            start_time = time.time()
            do_summary = global_step % (args.summary_freq) == 0
            # do_summary = global_step % 1 == 0
            loss, disp_loss, label_loss, scalar_outputs, scalar_outputs2, image_outputs = test_sample(sample, compute_metrics=do_summary)
            if do_summary:
               save_images(logger, 'test', image_outputs, global_step)
            # print(scalar_outputs,scalar_outputs2)
            avg_test_scalars.update(scalar_outputs)
            avg_test_scalars2.update(scalar_outputs2)
            del scalar_outputs,scalar_outputs2, image_outputs
            print('Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, disp_loss = {:.3f}, label_loss = {:.3f}, time = {:3f}'.format(epoch_idx, args.epochs,
                                                                                     batch_idx,
                                                                                     len(TestImgLoader), loss, disp_loss, label_loss,
                                                                                     time.time() - start_time))
        avg_test_scalars = avg_test_scalars.mean()
        avg_test_scalars2 = avg_test_scalars2.mean()
        save_scalars(logger, 'fulltest', avg_test_scalars, len(TrainImgLoader) * (epoch_idx + 1)) #,avg_test_scalars2
        save_scalars2(logger, 'fulltest2', avg_test_scalars2, len(TrainImgLoader) * (epoch_idx + 1)) #,avg_test_scalars2
        print("avg_test_scalars", avg_test_scalars,"avg_test_scalars2", avg_test_scalars2)
        gc.collect()


# train one sample
def train_sample(sample, compute_metrics=False):
    model.train()
    imgL, imgR, disp_gt, disp_gt_4, label_true = sample['left'], sample['right'], sample['disparity'], sample['disparity_4'], sample['label']

    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()
    disp_gt_4 = disp_gt_4.cuda()
    label_true = label_true.cuda()

    optimizer.zero_grad()

    disp_ests, label_est, label_est_r = model(imgL, imgR)
    mask = (disp_gt < args.maxdisp) & (disp_gt >= -args.maxdisp)
    mask_4 = (disp_gt_4 < args.maxdisp) & (disp_gt_4 >= -args.maxdisp)
    masks = [mask, mask_4, mask, mask_4]
    disp_gts = [disp_gt, disp_gt_4, disp_gt, disp_gt_4]

    lrsc_loss = LRSC_loss(label_est_r, disp_ests, label_true)
    disp_loss = model_loss_train(disp_ests, disp_gts, masks)
    label_loss = model_label_loss(label_est, label_true, nums, args.attention_weights_only)

    loss = disp_loss + label_loss + lrsc_loss

    disp_gt[disp_gt < -871.0] = 0

    scalar_outputs = {"loss": loss,"disp_loss": disp_loss, "label_loss": label_loss}
    
    if compute_metrics:
        with torch.no_grad():
            scalar_outputs["EPE"] = [EPE_metric(disp_ests[0], disp_gt, mask)]
            scalar_outputs["D1"] = [D1_metric(disp_ests[0], disp_gt, mask)]
            scalar_outputs["Thres1"] = [Thres_metric(disp_ests[0], disp_gt, mask, 1.0)]
            scalar_outputs["Thres2"] = [Thres_metric(disp_ests[0], disp_gt, mask, 2.0)]  
    loss.backward()
    optimizer.step()
    return tensor2float(loss), tensor2float(disp_loss), tensor2float(label_loss), tensor2float(lrsc_loss), tensor2float(scalar_outputs)

# test one sample
@make_nograd_func
def test_sample(sample, compute_metrics=True):
    model.eval()
    metric = SegmentationMetric(nums - 1)
    imgL, imgR, disp_gt, label = sample['left'], sample['right'], sample['disparity'], sample['label']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()
    label_true = label.cuda()

    mask = (disp_gt < args.maxdisp) & (disp_gt >= -args.maxdisp)
    
    disp_ests, label_est = model(imgL, imgR)
    metric.addBatch(label_est, label_true)

    disp_gts = [disp_gt]
    masks = [mask]


    disp_loss = model_loss_test(disp_ests, disp_gts, masks)
    label_loss = model_label_loss(label_est, label_true, nums, args.attention_weights_only)
    loss = disp_loss + label_loss

    disp_gt[disp_gt < -871.0] = 0
    
    scalar_outputs2 = {}
    scalar_outputs = {"loss": loss, "disp_loss":disp_loss, "label_loss":label_loss}
    image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL}#, 
    # image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL,"label_est_vis":label_est_vis, "label_mask_vis": label_mask_vis}
    scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
    scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
    scalar_outputs["PA"] = [metric.pixelAccuracy()]
    scalar_outputs["MPA"] = [metric.meanPixelAccuracy()]
    scalar_outputs["mIoU"] = [metric.meanIntersectionOverUnion()]
    for i in range(nums - 1):
        scalar_outputs2["CPA" + str(i)] = [metric.classPixelAccuracy()[i]]
        scalar_outputs2["IoU" + str(i)] = [metric.IoU()[i]]
    
    if compute_metrics:
        image_outputs["label"] = [feature_vis(F.one_hot(label_true.to(torch.int64), nums).permute(0, 3, 1, 2).float())]
        image_outputs["label_est"] = [feature_vis(label_est)]
        image_outputs["errormap"] = [disp_error_image_func.apply(disp_est, disp_gt) for disp_est in disp_ests]
    return tensor2float(loss),  tensor2float(disp_loss), tensor2float(label_loss), tensor2float(scalar_outputs), scalar_outputs2, image_outputs

if __name__ == '__main__':
    sys.stdout = Logger1(args.logdir+"/log.log", sys.stdout)
    train()