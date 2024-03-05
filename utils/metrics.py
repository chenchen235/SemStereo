import torch
import torch.nn.functional as F
from utils.experiment import make_nograd_func
from torch.autograd import Variable
from torch import Tensor
# from utils.metrics_mask import *
import numpy as np

def check_shape_for_metric_computation(*vars):
    assert isinstance(vars, tuple)
    for var in vars:
        assert len(var.size()) == 3
        assert var.size() == vars[0].size()

# a wrapper to compute metrics for each image individually
def compute_metric_for_each_image(metric_func):
    def wrapper(D_ests, D_gts, masks, *nargs):
        check_shape_for_metric_computation(D_ests, D_gts, masks)
        bn = D_gts.shape[0]  # batch size
        results = []  # a list to store results for each image
        # compute result one by one
        for idx in range(bn):
            # if tensor, then pick idx, else pass the same value
            cur_nargs = [x[idx] if isinstance(x, (Tensor, Variable)) else x for x in nargs]
            if masks[idx].float().mean() / (D_gts[idx] > 0).float().mean() < 0.1:
                print("masks[idx].float().mean() too small, skip")
            else:
                ret = metric_func(D_ests[idx], D_gts[idx], masks[idx], *cur_nargs)
                results.append(ret)
        if len(results) == 0:
            print("masks[idx].float().mean() too small for all images in this batch, return 0")
            return torch.tensor(0, dtype=torch.float32, device=D_gts.device)
        else:
            return torch.stack(results).mean()
    return wrapper

@make_nograd_func
@compute_metric_for_each_image
def D1_metric(D_est, D_gt, mask):
    D_est, D_gt = D_est[mask], D_gt[mask]
    E = torch.abs(D_gt - D_est)
    err_mask = (E > 3) & (E / D_gt.abs() > 0.05)
    return torch.mean(err_mask.float())

@make_nograd_func
@compute_metric_for_each_image
def Thres_metric(D_est, D_gt, mask, thres):
    assert isinstance(thres, (int, float))
    D_est, D_gt = D_est[mask], D_gt[mask]
    E = torch.abs(D_gt - D_est)
    err_mask = E > thres
    return torch.mean(err_mask.float())

# NOTE: please do not use this to build up training loss
@make_nograd_func
@compute_metric_for_each_image
def EPE_metric(D_est, D_gt, mask):
    D_est, D_gt = D_est[mask], D_gt[mask]
    return F.l1_loss(D_est, D_gt, size_average=True)



@make_nograd_func
@compute_metric_for_each_image
def D1_metric_mask(D_est, D_gt, mask, mask_img):
    # D_est, D_gt = D_est[(mask&mask_img)], D_gt[(mask&mask_img)]
    D_est, D_gt = D_est[mask_img], D_gt[mask_img]
    E = torch.abs(D_gt - D_est)
    err_mask = (E > 3) & (E / D_gt.abs() > 0.05)
    return torch.mean(err_mask.float())

@make_nograd_func
@compute_metric_for_each_image
def Thres_metric_mask(D_est, D_gt, mask, thres, mask_img):
    assert isinstance(thres, (int, float))
    # D_est, D_gt = D_est[(mask&mask_img)], D_gt[(mask&mask_img)]
    D_est, D_gt = D_est[mask_img], D_gt[mask_img]
    E = torch.abs(D_gt - D_est)
    err_mask = E > thres
    return torch.mean(err_mask.float())

# NOTE: please do not use this to build up training loss
@make_nograd_func
@compute_metric_for_each_image
def EPE_metric_mask(D_est, D_gt, mask, mask_img):
    # print((mask&mask_img).size(), D_est.size(), mask, mask_img)
    # D_est, D_gt = D_est[(mask&mask_img)], D_gt[(mask&mask_img)]
    D_est, D_gt = D_est[mask_img], D_gt[mask_img]
    return F.l1_loss(D_est, D_gt, size_average=True)

class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass 
        self.confusionMatrix = np.zeros((self.numClass,)*2)
        # self.numClass_3 = numClass - 1 
        # self.confusionMatrix_3 = np.zeros((self.numClass_3,)*2)
 
    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() /  self.confusionMatrix.sum()
        return acc
 
    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率
 
    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        # print(classAcc)
        meanAcc = np.nanmean(classAcc) # np.nanmean 求平均值，nan表示遇到Nan类型，其值取为0
        return meanAcc # 返回单个值，如：np.nanmean([0.90, 0.80, 0.96, nan, nan]) = (0.90 + 0.80 + 0.96） / 3 =  0.89
 
    def IoU(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix) # 取对角元素的值，返回列表
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(self.confusionMatrix) # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表 
        IoU = intersection / union  # 返回列表，其值为各个类别的IoU
        return IoU

    def meanIntersectionOverUnion(self): # MIOU
        mIoU = np.nanmean(self.IoU()) # 求各类别IoU的平均
        return mIoU
    
    # def miou_3(self):

    #     # # accumulate statistics for IOU-3
    #     # for cat in range(NUM_CATEGORIES):
    #     #     valid_disparity = (imgLabel == 3) | (abs(D_est[mask_img] - D_gt[mask_img]) < 3)
    #     #     tp3[cat] += ((imgLabel == cat) & (imgLabel == cat) & (imgLabel < NUM_CATEGORIES) & valid_disparity).sum()
    #     #     fp3[cat] += ((imgLabel == cat) & (imgLabel != cat) & (imgLabel < NUM_CATEGORIES)).sum()
    #     #     fn3[cat] += ((imgLabel != cat) & (imgLabel == cat) & (imgLabel < NUM_CATEGORIES)).sum()

    #     intersection = np.diag(self.confusionMatrix) + (np.sum(self.confusionMatrix[3], axis=1) + np.sum(self.confusionMatrix[3], axis=0)) - np.diag(self.confusionMatrix)[3] # 取对角元素的值，返回列表
    #     union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(self.confusionMatrix) # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表 
    #     IoU = intersection / union  # 返回列表，其值为各个类别的IoU
    #     mIoU_3 = np.nanmean(IoU) # 求各类别IoU的平均
    #     return mIoU_3

    def get_confusion_matrix(self, label, pred,  num_class = 5, ignore= None):
        """
        Calcute the confusion matrix by given label and pred
        """
        size = pred.size()
        output = pred.cpu().numpy().transpose(0, 2, 3, 1)
        seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
        seg_gt = np.asarray(label.cpu().numpy()[:, :size[-2], :size[-1]], dtype=np.int)

        if ignore:
            ignore_index = seg_gt != ignore
            seg_gt = seg_gt[ignore_index]
            seg_pred = seg_pred[ignore_index]

        index = (seg_gt * num_class + seg_pred).astype('int32').flatten()
        # print(np.max(index))
        label_count = np.bincount(index)
        confusion_matrix = np.zeros((num_class, num_class))

        for i_label in range(num_class):
            for i_pred in range(num_class):
                cur_index = i_label * num_class + i_pred
                if cur_index < len(label_count):
                    confusion_matrix[i_label, i_pred] = label_count[cur_index]
        
        return confusion_matrix

    # def get_confusion_matrix_3(self, label, pred,  num_class = 4, ignore= [3, 5]):
    #     """
    #     Calcute the confusion matrix by given label and pred
    #     """
    #     size = label.size()
    #     output = pred.cpu().numpy().transpose(0, 2, 3, 1)
    #     seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
    #     seg_gt = np.asarray(
    #     label.cpu().numpy()[:, :size[-2], :size[-1]], dtype=np.int)

    #     ignore_index = seg_gt != ignore
    #     seg_gt = seg_gt[ignore_index]
    #     seg_pred = seg_pred[ignore_index]

    #     index = (seg_gt * num_class + seg_pred).astype('int32')
    #     label_count = np.bincount(index)
    #     confusion_matrix = np.zeros((num_class, num_class))

    #     for i_label in range(num_class):
    #         for i_pred in range(num_class):
    #             cur_index = i_label * num_class + i_pred
    #             if cur_index < len(label_count):
    #                 confusion_matrix[i_label,
    #                                 i_pred] = label_count[cur_index]
    #     return confusion_matrix
 
    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU
 
 
    def addBatch(self, imgPredict, imgLabel): #, D_est, D_gt, mask_img
        # assert imgPredict.shape == imgLabel.shape
        # print(imgPredict.size(), imgLabel.size())
        self.confusionMatrix += self.get_confusion_matrix(imgLabel, imgPredict, num_class = self.numClass)
        # self.confusionMatrix_3 += self.get_confusion_matrix_3(imgLabel, imgPredict, D_est, D_gt, mask_imgnum_class = self.numClass_3)
 
    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))