from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
from torch.autograd.function import Function
import torch.nn.functional as F
import numpy as np

class Fusion(nn.Module):
    def __init__(self, inplanes, interplanes_x, interplanes_y, mid_channels):
        super(Fusion, self).__init__()
        self.conv_y = nn.Sequential(nn.Conv2d(interplanes_y, interplanes_x//2, kernel_size=1, padding=0, bias=True), nn.BatchNorm2d(interplanes_x//2))
        self.conv_x = nn.Sequential(nn.Conv2d(interplanes_x, interplanes_x//2, kernel_size=1, padding=0, bias=True), nn.BatchNorm2d(interplanes_x//2))
        self.conv = BasicConv(interplanes_x, interplanes_x, kernel_size=3, stride=1, padding=1)

    def forward(self, x, y):
        
        input_size = x.size()
        y = self.conv_y(y)
        x = self.conv_x(x)
        if (y.shape != x.shape):
            y = F.interpolate(y, (input_size(-2), input_size(-1)), mode='bilinear', align_corners=False)
        cat_ = torch.cat((x, y), 1)
        cat_ = self.conv(cat_)
        cat_ = torch.softmax(cat_, dim=1)
        x = x * (cat_ + 1)
        y = y * (cat_ + 1)
        return x, y

class segmenthead(nn.Module):

    def __init__(self, inplanes, interplanes, outplanes, scale_factor=None):
        super(segmenthead, self).__init__()
        self.conv1 = BasicConv(inplanes, interplanes, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(interplanes, outplanes, kernel_size=1, padding=0, bias=True)
        # self.dropout = nn.Dropout2d(p=0.1)
        self.scale_factor = scale_factor

    def forward(self, x):
        
        x = self.conv1(x)
        # x = self.dropout(x)
        out = self.conv2(x)

        if self.scale_factor is not None:  # 线性上采样
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            out = F.interpolate(out,
                        size=[height, width],
                        mode='bilinear', align_corners = False)
        return out

class DWConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DWConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        return x


class DWConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DWConv3d, self).__init__()
        self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels)
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        return x

class BasicConv(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, bn=True, relu=True, **kwargs):
        super(BasicConv, self).__init__()

        self.relu = relu
        self.use_bn = bn
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = nn.ReLU()(x)#, inplace=True)
            # x = nn.ReLU()(x) # Faster
        return x


class Conv2x(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, keep_concat=True, bn=True, relu=True, keep_dispc=False):
        super(Conv2x, self).__init__()
        self.concat = concat
        self.is_3d = is_3d 
        if deconv and is_3d: 
            kernel = (4, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3

        if deconv and is_3d and keep_dispc:
            kernel = (1, 4, 4)
            stride = (1, 2, 2)
            padding = (0, 1, 1)
            self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=kernel, stride=stride, padding=padding)
        else: # default
            self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=kernel, stride=2, padding=1)

        if self.concat: 
            mul = 2 if keep_concat else 1
            self.conv2 = BasicConv(out_channels*2, out_channels*mul, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)
            # self.conv2 = BasicConv(out_channels*2, out_channels*mul, False, is_3d, bn, relu, kernel_size=1, stride=1, padding=0)
        else:
            self.conv2 = BasicConv(out_channels, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)
            # self.conv2 = BasicConv(out_channels, out_channels, False, is_3d, bn, relu, kernel_size=1, stride=1, padding=0)


    def forward(self, x, rem): # x 2 rem
        x = self.conv1(x)
        if x.shape != rem.shape:
            x = F.interpolate(
                x,
                size=(rem.shape[-2], rem.shape[-1]),
                mode='bilinear') # nearest
        if self.concat:
            x = torch.cat((x, rem), 1)
        else: 
            x = x + rem
        x = self.conv2(x)
        return x


def disparity_regression(x, maxdisp):
    # print(x)
    assert len(x.shape) == 4
    disp_values = torch.arange(-maxdisp, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp * 2, 1, 1)
    # print(torch.sum(x * disp_values))
    return torch.sum(x * disp_values, 1, keepdim=False)


def build_concat_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 2 * C, maxdisp * 2, H, W])
    for i in range(-maxdisp, maxdisp):
        if i < 0:
            volume[:, :C, i + maxdisp, :, :i] = refimg_fea[:, :, :, :i]
            volume[:, C:, i + maxdisp, :, :i] = targetimg_fea[:, :, :, -i:]
        elif i > 0:
            volume[:, :C, i + maxdisp, :, i:] = refimg_fea[:, :, :, i:]
            volume[:, C:, i + maxdisp, :, i:] = targetimg_fea[:, :, :, :-i]
        else:
            volume[:, :C, i + maxdisp, :, :] = refimg_fea
            volume[:, C:, i + maxdisp, :, :] = targetimg_fea
    volume = volume.contiguous()
    return volume


def groupwise_correlation(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost

def build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp * 2, H, W])
    for i in range(-maxdisp, maxdisp): # 从负到正
        if i < 0:
            volume[:, :, i + maxdisp, :, :i] = groupwise_correlation(refimg_fea[:, :, :, :i], targetimg_fea[:, :, :, -i:],
                                                           num_groups)
        elif i > 0:
            volume[:, :, i + maxdisp, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i],
                                                           num_groups)
        else:
            volume[:, :, i + maxdisp, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume

def groupwise_correlation_norm(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    fea1 = fea1.view([B, num_groups, channels_per_group, H, W])
    fea2 = fea2.view([B, num_groups, channels_per_group, H, W])
    cost = ((fea1/(torch.norm(fea1, 2, 2, True)+1e-05)) * (fea2/(torch.norm(fea2, 2, 2, True)+1e-05))).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost


def build_gwc_volume_norm(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp * 2, H, W])
    # print(volume.size())
    for i in range(-maxdisp, maxdisp):
        if i < 0:
            volume[:, :, i + maxdisp, :, :i] = groupwise_correlation_norm(refimg_fea[:, :, :, :i], targetimg_fea[:, :, :, -i:],
                                                           num_groups)
        elif i > 0:
            volume[:, :, i + maxdisp, :, i:] = groupwise_correlation_norm(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i],
                                                           num_groups)
        else:
            volume[:, :, i + maxdisp, :, :] = groupwise_correlation_norm(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume

def norm_correlation(fea1, fea2):
    cost = torch.mean(((fea1/(torch.norm(fea1, 2, 1, True)+1e-05)) * (fea2/(torch.norm(fea2, 2, 1, True)+1e-05))), dim=1, keepdim=True)
    return cost

def build_norm_correlation_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 1, maxdisp * 2, H, W])
    for i in range(-maxdisp, maxdisp):
        if i < 0:
            volume[:, :, i + maxdisp, :, :i] = norm_correlation(refimg_fea[:, :, :, :i], targetimg_fea[:, :, :, -i:])
        elif i > 0:
            volume[:, :, i + maxdisp, :, i:] = norm_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i])
        else:
            volume[:, :, i +maxdisp, :, :] = norm_correlation(refimg_fea, targetimg_fea)
    volume = volume.contiguous()
    return volume

def disparity_variance(x, maxdisp, disparity):
    # the shape of disparity should be B,1,H,W, return is the variance of the cost volume [B,1,H,W]
    assert len(x.shape) == 4
    disp_values = torch.arange(-maxdisp, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp * 2, 1, 1)
    disp_values = (disp_values - disparity) ** 2
    return torch.sum(x * disp_values, 1, keepdim=True)

def SpatialTransformer_grid(x, y, disp_range_samples):

    bs, channels, height, width = y.size()
    ndisp = disp_range_samples.size()[1] # 

    mh, mw = torch.meshgrid([torch.arange(0, height, dtype=x.dtype, device=x.device),
                                 torch.arange(0, width, dtype=x.dtype, device=x.device)])  # (H *W)

    mh = mh.reshape(1, 1, height, width).repeat(bs, ndisp, 1, 1)
    mw = mw.reshape(1, 1, height, width).repeat(bs, ndisp, 1, 1)  # (B, D, H, W)

    cur_disp_coords_y = mh
    cur_disp_coords_x = mw - disp_range_samples

    coords_x = cur_disp_coords_x / ((width - 1.0) / 2.0) - 1.0  # trans to -1 - 1
    coords_y = cur_disp_coords_y / ((height - 1.0) / 2.0) - 1.0
    grid = torch.stack([coords_x, coords_y], dim=4) #(B, D, H, W, 2)

    y_warped = F.grid_sample(y, grid.view(bs, ndisp * height, width, 2), mode='bilinear',
                               padding_mode='zeros', align_corners=True).view(bs, channels, ndisp, height, width)  #(B, C, D, H, W)

    x_warped = x.unsqueeze(2).repeat(1, 1, ndisp, 1, 1) #(B, C, D, H, W)

    return y_warped, x_warped

class Propagation(nn.Module):
    def __init__(self):
        super(Propagation, self).__init__()
        self.replicationpad = nn.ReplicationPad2d(1) # 对称四个方向填充一行

    def forward(self, disparity_samples):

        one_hot_filter = torch.zeros(5, 1, 3, 3, device=disparity_samples.device).float()
        one_hot_filter[0, 0, 0, 0] = 1.0
        one_hot_filter[1, 0, 1, 1] = 1.0
        one_hot_filter[2, 0, 2, 2] = 1.0
        one_hot_filter[3, 0, 2, 0] = 1.0
        one_hot_filter[4, 0, 0, 2] = 1.0
        disparity_samples = self.replicationpad(disparity_samples)
        aggregated_disparity_samples = F.conv2d(disparity_samples,
                                                    one_hot_filter,padding=0)
                                                    
        return aggregated_disparity_samples

class Propagation2(nn.Module):
    def __init__(self):
        super(Propagation2, self).__init__()
        self.replicationpad = nn.ReplicationPad2d(2) # 对称四个方向填充一行

    def forward(self, disparity_samples):

        one_hot_filter = torch.zeros(9, 1, 5, 5, device=disparity_samples.device).float()
        one_hot_filter[0, 0, 0, 2] = 1.0
        one_hot_filter[1, 0, 2, 0] = 1.0
        one_hot_filter[2, 0, 2, 2] = 1.0
        one_hot_filter[3, 0, 2, 4] = 1.0
        one_hot_filter[4, 0, 4, 2] = 1.0
        one_hot_filter[5, 0, 1, 1] = 1.0
        one_hot_filter[6, 0, 3, 3] = 1.0
        one_hot_filter[7, 0, 1, 3] = 1.0
        one_hot_filter[8, 0, 3, 1] = 1.0

        disparity_samples = self.replicationpad(disparity_samples)
        aggregated_disparity_samples = F.conv2d(disparity_samples,
                                                    one_hot_filter,padding=0)
                                                    
        return aggregated_disparity_samples
        

class Propagation_prob2(nn.Module):
    def __init__(self):
        super(Propagation_prob2, self).__init__()
        self.replicationpad = nn.ReplicationPad3d((2, 2, 2, 2, 0, 0))

    def forward(self, prob_volume):
        one_hot_filter = torch.zeros(9, 1, 1, 5, 5, device=prob_volume.device).float()
        one_hot_filter[0, 0, 0, 0, 0] = 1.0
        one_hot_filter[1, 0, 0, 1, 1] = 1.0
        one_hot_filter[2, 0, 0, 2, 2] = 1.0
        one_hot_filter[3, 0, 0, 2, 0] = 1.0
        one_hot_filter[4, 0, 0, 0, 2] = 1.0
        one_hot_filter[0, 0, 0, 0, 2] = 1.0
        one_hot_filter[1, 0, 0, 2, 0] = 1.0
        one_hot_filter[2, 0, 0, 2, 2] = 1.0
        one_hot_filter[3, 0, 0, 2, 4] = 1.0
        one_hot_filter[4, 0, 0, 4, 2] = 1.0
        one_hot_filter[5, 0, 0, 1, 1] = 1.0
        one_hot_filter[6, 0, 0, 3, 3] = 1.0
        one_hot_filter[7, 0, 0, 1, 3] = 1.0
        one_hot_filter[8, 0, 0, 3, 1] = 1.0

        prob_volume = self.replicationpad(prob_volume)
        prob_volume_propa = F.conv3d(prob_volume, one_hot_filter,padding=0)

        return prob_volume_propa
        
class Propagation_prob(nn.Module):
    def __init__(self):
        super(Propagation_prob, self).__init__()
        self.replicationpad = nn.ReplicationPad3d((1, 1, 1, 1, 0, 0))

    def forward(self, prob_volume):
        one_hot_filter = torch.zeros(5, 1, 1, 3, 3, device=prob_volume.device).float()
        one_hot_filter[0, 0, 0, 0, 0] = 1.0
        one_hot_filter[1, 0, 0, 1, 1] = 1.0
        one_hot_filter[2, 0, 0, 2, 2] = 1.0
        one_hot_filter[3, 0, 0, 2, 0] = 1.0
        one_hot_filter[4, 0, 0, 0, 2] = 1.0

        prob_volume = self.replicationpad(prob_volume)
        prob_volume_propa = F.conv3d(prob_volume, one_hot_filter,padding=0)

        return prob_volume_propa

class ConvSelfAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvSelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # Compute queries, keys, and values
        query = self.query_conv(x)
        key = self.key_conv(x)
        value = self.value_conv(x)

        # Reshape the tensors for efficient matrix multiplication
        b, c, h, w = query.size()
        query = query.view(b, c, -1)
        key = key.view(b, c, -1)
        value = value.view(b, c, -1)

        # Compute self-attention scores
        attention = torch.bmm(query.permute(0, 2, 1), key)
        attention = torch.nn.functional.softmax(attention, dim=2)

        # Apply attention to values
        out = torch.bmm(attention, value)
        out = out.view(b, c, h, w)

        # Combine with the input using a learnable parameter gamma
        out = self.gamma * out + x

        return out

class SSR_upsample(nn.Module):
    def __init__(self, num_classes):
        super(SSR_upsample, self).__init__()
        self.num_classes = num_classes
        self.conv = nn.Sequential(nn.BatchNorm2d(1), nn.Conv2d(1, num_classes, kernel_size=3, padding = 1), nn.BatchNorm2d(num_classes))
        self.conv1 = nn.Sequential(nn.Conv2d(num_classes, num_classes, kernel_size=1, padding = 0), nn.BatchNorm2d(num_classes))
        self.conv2 = nn.Sequential(nn.Conv2d(num_classes, num_classes, kernel_size=1, padding = 0), nn.BatchNorm2d(num_classes))
        self.conv3 = nn.Conv2d(num_classes, 1, kernel_size=1, padding=0)

    def forward(self, depth_low, weights, pred_label):
        b, c, h, w = depth_low.shape
        pred_label = F.softmax(pred_label, dim=1)
        depth_ = F.interpolate(depth_low, (h*4,w*4),mode='bilinear').reshape(b,1,h*4,w*4)
        depth = self.conv(depth_)

        prob =torch.sigmoid(self.conv1(pred_label * weights))
        prob = torch.sigmoid(self.conv2(prob * weights))
        res = self.conv3(depth * prob)
        depth = depth_ + res
        return depth.squeeze(1)


def regression_topk(cost, disparity_samples, k):

    _, ind = cost.sort(1, True)
    pool_ind = ind[:, :k]
    cost = torch.gather(cost, 1, pool_ind)
    prob = F.softmax(cost, 1)
    disparity_samples = torch.gather(disparity_samples, 1, pool_ind)    
    pred = torch.sum(disparity_samples * prob, dim=1, keepdim=True)
    return pred
    


