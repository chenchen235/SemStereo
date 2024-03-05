from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from models.submodule import *
from models.submodule_other import *
import math
import gc
import time
import timm

class SubModule(nn.Module):
    def __init__(self):
        super(SubModule, self).__init__()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class Feature(SubModule):
    def __init__(self):
        super(Feature, self).__init__()

        model = timm.create_model('mobilevitv2_100', pretrained=True, features_only=True)

        self.conv_stem = model.stem # 100: 32  /50:16 
        
        self.block0 = torch.nn.Sequential(*model.stages_0) # 2  64  / 50 : 32
        self.block1 = torch.nn.Sequential(*model.stages_1) # 4   128  64
        self.block2 = torch.nn.Sequential(*model.stages_2) # 8   256  128
        self.block3 = torch.nn.Sequential(*model.stages_3) # 16  384  192
        self.block4 = torch.nn.Sequential(*model.stages_4) # 32  512  256    
        
    def forward(self, x):

        x = self.conv_stem(x)
        x2 = self.block0(x)
        x4 = self.block1(x2)
        x8 = self.block2(x4)
        x16 = self.block3(x8)
        x32 = self.block4(x16)

        return [x2, x4, x8, x16, x32]


class FeatUp(SubModule):
    def __init__(self):
        super(FeatUp, self).__init__()
        chans = [64, 128, 256, 384, 512]
        self.deconv32_16 = Conv2x(chans[4], chans[3], deconv=True, concat=True)
        self.deconv16_8 = Conv2x(chans[3]*2, chans[2], deconv=True, concat=True)
        self.deconv8_4 = Conv2x(chans[2]*2, chans[1], deconv=True, concat=True)
        self.deconv4_2 = Conv2x(chans[1]*2, chans[0], deconv=True, concat=True)

        self.weight_init()

    def forward(self, featL, featR=None):
        x2, x4, x8, x16, x32 = featL
        y2, y4, y8, y16, y32 = featR

        x16 = self.deconv32_16(x32, x16)
        y16 = self.deconv32_16(y32, y16)
        
        x8 = self.deconv16_8(x16, x8)
        y8 = self.deconv16_8(y16, y8)

        x4 = self.deconv8_4(x8, x4)
        y4 = self.deconv8_4(y8, y4)

        x2 = self.deconv4_2(x4, x2)
        y2 = self.deconv4_2(y4, y2)

        return [x2, x4, x8, x16, x32], [y2, y4, y8, y16, y32]


class channelAtt(SubModule):
    def __init__(self, cv_chan, im_chan):
        super(channelAtt, self).__init__()

        self.im_att = nn.Sequential(
            BasicConv(im_chan, im_chan//2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(im_chan//2, cv_chan, 1)) # 96 -> 48 -> 16

        self.weight_init()

    def forward(self, cv, im):

        channel_att = self.im_att(im).unsqueeze(2)
        cv = (torch.sigmoid(channel_att) )*cv
        return cv


class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(in_channels, in_channels * 2, 3, 2, 1),
                                   nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 2, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 4, 3, 2, 1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 4, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.attention_block = attention_block(channels_3d=in_channels * 4, num_heads=16, block=(4, 4, 4))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels * 2))

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels))

        self.redir1 = convbn_3d(in_channels, in_channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_3d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv4 = self.attention_block(conv4)
        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)
        return conv6

class hourglass2(nn.Module):
    def __init__(self, in_channels):
        super(hourglass2, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(in_channels, in_channels * 2, 3, 2, 1),
                                   nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 2, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 4, 3, 2, 1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 4, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.attention_block = attention_block(channels_3d=in_channels * 4, num_heads=16, block=(6, 4, 4))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels * 2))

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels))

        self.redir1 = convbn_3d(in_channels, in_channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_3d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv4 = self.attention_block(conv4)
        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)
        return conv6

class SemStereo(nn.Module):
    def __init__(self, maxdisp, att_weights_only, seg_if, stereo_if, num_classes):
        super(SemStereo, self).__init__()
        self.att_weights_only = att_weights_only
        self.seg_if = seg_if
        self.stereo_if = stereo_if
        self.maxdisp = maxdisp
        self.num_classes = num_classes
        
        self.feature = Feature()
        self.feature_up = FeatUp()

        self.chans = [128, 256, 512, 768, 512] # 2 4 8 16 32
        self.chans2 = [64, 128, 256, 384, 256]    

        if self.seg_if:
            self.head_l = segmenthead(inplanes = self.chans[0], interplanes = self.chans[0]//4, outplanes = num_classes, scale_factor=2)
            self.head_r = segmenthead(inplanes = self.chans[0], interplanes = self.chans[0]//4, outplanes = num_classes, scale_factor=2)

        if self.stereo_if:
            self.gamma = nn.Parameter(torch.zeros(1))
            self.beta = nn.Parameter(2*torch.ones(1))        

            self.spx2 =  nn.Sequential(nn.ConvTranspose2d(self.chans2[0] * 2, 6, kernel_size=4, stride=2, padding=1))
            self.spx4_2 = Conv2x(self.chans2[1]*2, self.chans2[0], True)
            self.spx8_4 = Conv2x(self.chans2[2]*2, self.chans2[1], True)
            self.spx16_8 = Conv2x(self.chans2[3]*2, self.chans2[2], True)
            self.spx32_16 = Conv2x(self.chans2[4], self.chans2[3], True)

            self.chal_0 = nn.Sequential(nn.Conv2d(self.chans[0], self.chans2[0], kernel_size=1, stride=1), nn.BatchNorm2d(self.chans2[0]))
            self.chal_1 = nn.Sequential(nn.Conv2d(self.chans[1], self.chans2[1], kernel_size=1, stride=1), nn.BatchNorm2d(self.chans2[1]))
            self.chal_2 = nn.Sequential(nn.Conv2d(self.chans[2], self.chans2[2], kernel_size=1, stride=1), nn.BatchNorm2d(self.chans2[2]))
            self.chal_3 = nn.Sequential(nn.Conv2d(self.chans[3], self.chans2[3], kernel_size=1, stride=1), nn.BatchNorm2d(self.chans2[3]))
            self.chal_4 = nn.Sequential(nn.Conv2d(self.chans[4], self.chans2[4], kernel_size=1, stride=1), nn.BatchNorm2d(self.chans2[4]))

            self.patch = nn.Conv3d(self.chans2[2]//8, self.chans2[2]//8, kernel_size=(1,3,3), stride=1, dilation=1, groups=self.chans2[2]//8, padding=(0,1,1), bias=False)

            self.concat_feature = nn.Sequential(
                                BasicConv(self.chans2[1], self.chans2[1]//2, kernel_size=3, stride=1, padding=1),
                                nn.Conv2d(self.chans2[1]//2, self.chans2[1]//4, 3, 1, 1, bias=False))

            self.corr_feature_att_8 = channelAtt(self.chans2[1]//4, self.chans2[2])
            self.concat_feature_att_4 = channelAtt(self.chans2[1]//4, self.chans2[1])
            self.hourglass_att = hourglass(32)
            self.classif_att_ = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))
            self.hourglass = hourglass2(32)
            self.classif = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

            self.concat_stem = BasicConv(self.chans2[1]//2, self.chans2[1]//4, is_3d=True, kernel_size=3, stride=1, padding=1)
            self.propagation = Propagation()
            self.propagation_prob = Propagation_prob()
            self.ssr_upsample = SSR_upsample(num_classes)

    def concat_volume_generator(self, left_input, right_input, disparity_samples):
        right_feature_map, left_feature_map = SpatialTransformer_grid(left_input, right_input, disparity_samples)
        concat_volume = torch.cat((left_feature_map, right_feature_map), dim=1)
        return concat_volume

    def forward(self, left, right):

        features_left = self.feature(left) 
        features_right = self.feature(right)

        features_left, features_right = self.feature_up(features_left, features_right)
    
        if self.seg_if:
            pred_label = self.head_l(features_left[0])
            pred_label_r = self.head_r(features_right[0])
        
        if self.stereo_if:
            features_left[0] = self.chal_0(features_left[0])
            features_left[1] = self.chal_1(features_left[1])
            features_left[2] = self.chal_2(features_left[2])
            features_left[3] = self.chal_3(features_left[3])
            features_left[4] = self.chal_4(features_left[4])

            features_right[1] = self.chal_1(features_right[1])
            features_right[2] = self.chal_2(features_right[2])

            xspx = self.spx32_16(features_left[4], features_left[3]) 
            xspx = self.spx16_8(xspx, features_left[2]) 
            xspx = self.spx8_4(xspx, features_left[1]) 
            xspx = self.spx4_2(xspx, features_left[0])
            spx_pred = self.spx2(xspx) 

            corr_volume = build_gwc_volume_norm(features_left[2], features_right[2], self.maxdisp//8, self.chans2[2]//8)
            corr_volume = self.patch(corr_volume)

            cost_att = self.corr_feature_att_8(corr_volume, features_left[2])
            cost_att = self.hourglass_att(cost_att) 
            cost_att = self.classif_att_(cost_att)
            att_weights = F.interpolate(cost_att, [self.maxdisp//4*2, left.size()[2]//4, left.size()[3]//4], mode='trilinear')
            
            pred_att = torch.squeeze(att_weights, 1)
            pred_att_prob = F.softmax(pred_att, dim=1)
            pred_att = disparity_regression(pred_att_prob, self.maxdisp//4)
            
            pred_variance = disparity_variance(pred_att_prob, self.maxdisp//4, pred_att.unsqueeze(1)) 
            pred_variance = self.beta + self.gamma * pred_variance 
            pred_variance = torch.sigmoid(pred_variance)
            pred_variance_samples = self.propagation(pred_variance) 
            disparity_samples = self.propagation(pred_att.unsqueeze(1)) 
            
            right_feature_x4, left_feature_x4 = SpatialTransformer_grid(features_left[1], features_right[1], disparity_samples) 
            disparity_sample_strength = (left_feature_x4 * right_feature_x4).mean(dim=1)
            disparity_sample_strength = torch.softmax(disparity_sample_strength*pred_variance_samples, dim=1)
        
            att_weights = self.propagation_prob(att_weights)
            att_weights = att_weights * disparity_sample_strength.unsqueeze(2)
            att_weights = torch.sum(att_weights, dim=1, keepdim=True)
            att_weights_prob = F.softmax(att_weights, dim=2)
            _, ind = att_weights_prob.sort(2, True)

            k = 24 # 32
            ind_k = ind[:, :, :k]
            ind_k = ind_k.sort(2, False)[0] 
            att_topk = torch.gather(att_weights_prob, 2, ind_k)
            disparity_sample_topk = ind_k.squeeze(1).float() - self.maxdisp//4

            att_prob = torch.gather(att_weights, 2, ind_k).squeeze(1)
            att_prob = F.softmax(att_prob, dim=1)
            pred_att = att_prob*disparity_sample_topk
            pred_att = torch.sum(pred_att, dim=1)
            pred_att_up = self.ssr_upsample(pred_att.unsqueeze(1), spx_pred, pred_label)

            if not self.att_weights_only:
                concat_features_left = self.concat_feature(features_left[1]) 
                concat_features_right = self.concat_feature(features_right[1])
                concat_volume = self.concat_volume_generator(concat_features_left, concat_features_right, disparity_sample_topk) 
                
                volume = att_topk * concat_volume
                volume = self.concat_stem(volume)
                volume = self.concat_feature_att_4(volume, features_left[1]) 
                cost = self.hourglass(volume) 
                cost = self.classif(cost)
                pred = regression_topk(cost.squeeze(1), disparity_sample_topk, 2)
                pred_up = self.ssr_upsample(pred, spx_pred, pred_label)

        if self.seg_if and not self.stereo_if:
            return pred_label

        if self.training:
            if self.att_weights_only:
                if not self.seg_if:
                    return [pred_att_up*4, pred_att*4]
                return [pred_att_up*4, pred_att*4], pred_label, pred_label_r
            else:
                if not self.seg_if:
                    return [pred_up*4, pred.squeeze(1)*4, pred_att_up*4, pred_att*4]
                return [pred_up*4, pred.squeeze(1)*4, pred_att_up*4, pred_att*4], pred_label, pred_label_r
        else:
            if self.att_weights_only:
                if not self.seg_if:
                    return [pred_att_up*4]
                return [pred_att_up*4], pred_label
            else:
                if not self.seg_if:
                    return [pred_up*4]
                return [pred_up*4], pred_label
