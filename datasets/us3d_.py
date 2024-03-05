import os
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from datasets.data_io import get_transform, read_all_lines, pfm_imread
import cv2
import torchvision
import scipy.signal as sig
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


def SpatialTransformer_grid(y, disp_range_samples):

    b, c, height, width = y.size()

    mh, mw = torch.meshgrid([torch.arange(0, height, dtype=y.dtype, device=y.device),
                                 torch.arange(0, width, dtype=y.dtype, device=y.device)]) # (H *W)

    mh = mh.reshape(height, width).repeat(1, 1, 1, 1)
    mw = mw.reshape(height, width).repeat(1, 1, 1, 1)  # (B, D, H, W)

    cur_disp_coords_y = mh
    cur_disp_coords_x = mw - disp_range_samples

    coords_x = cur_disp_coords_x / ((width - 1.0) / 2.0) - 1.0  # trans to -1 - 1
    coords_y = cur_disp_coords_y / ((height - 1.0) / 2.0) - 1.0
    grid = torch.stack([coords_x, coords_y], dim=4) #(B, D, 2)

    y_warped = F.grid_sample(y, grid.view(1, height, width, 2), mode='bilinear',
                               padding_mode='zeros', align_corners=True).view(height, width)  #(B, C, D, H, W)

    return y_warped

class Us3dDataset(Dataset):
    def __init__(self, datapath, list_filename, training):
        self.datapath = datapath
        self.left_filenames, self.right_filenames, self.disp_filenames, self.label_filenames = self.load_path(list_filename)
        self.training = training

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        # lines = lines[0:10]
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        disp_images = [x[2] for x in splits]
        label_images = [x[3] for x in splits]
        return left_images, right_images, disp_images, label_images

    def load_image(self, filename):
        return Image.open(filename).convert('RGB')        

    def load_disp(self, filename):
        data = Image.open(filename)
        data = np.ascontiguousarray(data, dtype=np.float32)
        return data

    def load_label(self, filename):
        data = Image.open(filename)
        data = np.ascontiguousarray(data, dtype=np.float32)
        return data

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        left_img = self.load_image(os.path.join(self.datapath, self.left_filenames[index]))
        right_img = self.load_image(os.path.join(self.datapath, self.right_filenames[index]))
        disparity = self.load_disp(os.path.join(self.datapath, self.disp_filenames[index]))
        label = self.load_label(os.path.join(self.datapath, self.label_filenames[index]))

        label_disparity_dict = {
            0: [],
            1: [],
            2: [],
            3: [],
            4: [],
            5: []
        }

        height, width = label.shape

        semantic_flat = label.reshape(-1)
        disparity_flat = disparity.reshape(-1)

        # 遍历每个像素的语义标签和视差值
        # for label, disparity in zip(semantic_flat, disparity_flat):
        #     if disparity < -800:
        #         continue
        #     # 将视差值添加到相应标签类别的列表中
        #     if label in label_disparity_dict:
        #         label_disparity_dict[label].append(disparity)

        img = (left_img.convert("L") - np.mean(left_img.convert("L"))) / np.std(left_img.convert("L"))
        kx = np.array([[-1, 0, 1]])
        ky = np.array([[-1], [0], [1]])
        # print(img.shape)

        # gx = np.zeros_like(img)
        # gy = np.zeros_like(img)
        # for channel in range(3):
        #     gx[:, :, channel] = sig.convolve2d(img[:, :, channel], kx, 'same')
        #     gy[:, :, channel] = sig.convolve2d(img[:, :, channel], ky, 'same')
        gx = torch.Tensor(sig.convolve2d(img, kx, 'same')).unsqueeze(0)
        gy = torch.Tensor(sig.convolve2d(img, ky, 'same')).unsqueeze(0)
        # print(gx.shape, gy.shape)

        if self.training:

            w, h = left_img.size
            # label_r = SpatialTransformer_grid(torch.Tensor(label).unsqueeze(0).unsqueeze(0), torch.Tensor(disparity).unsqueeze(0).unsqueeze(0))
            height, width = label.shape[0], label.shape[1]
            # right_x = np.arange(width) - disparity
            # right_x = right_x[mask]
            # print(right_x.shape)
            # valid_mask = (right_x >= 0) & (right_x < width) &(disparity < 64) & (disparity >= -64)
            # right_semantic = np.full_like(label, 5)
            # print(right_semantic[valid_mask].shape, label[right_x.astype(int)][valid_mask].shape)

            # for y in range(height):
            #     for x in range(width):
            #         disparity_ = disparity[y, x]
            #         if disparity_ < -64 or disparity_ >=64:
            #             continue
            #         right_x = np.round(x - disparity_)
            #         # print(right_x)
            #         if right_x >= 0 and right_x < width:
            #             right_semantic[y, right_x.astype(int)] = label[y, x]
            # right_semantic[valid_mask] = label[right_x[valid_mask].astype(int)] 
            # print(np.max(right_semantic))
            # edges = np.zeros((1024,1024))

            # edges = np.zeros(label.shape, dtype=np.uint8)

            # for class_id in range(6):
            #     # 创建一个二值掩膜，将当前类别的像素设置为255，其他类别的像素设置为0
            #     mask = np.uint8(label == class_id)

            #     # 应用Canny边缘检测到当前类别的二值掩膜
            #     class_edges = cv2.Canny(mask, threshold1=0.5, threshold2=1)

            #     # print(class_edges.shape, edges.shape)
            #     # 将当前类别的边缘图像与总边缘图像进行合并
            #     edges = cv2.bitwise_or(edges, class_edges)
            # # kernel_size = (5, 5)
            # # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
            # edges = edges /255.
            # 进行闭运算
            # edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            # cv2.imwrite('edges.jpg', edges)
            # cv2.imwrite('eroded_edges.jpg', eroded_edges)

            # 定义颜色映射
            # colors = ['red', 'green', 'blue', 'yellow', 'purple', 'cyan']
            # labels = [0, 1, 2, 3, 4, 5]
            # cmap = ListedColormap(colors)
            # norm = BoundaryNorm(labels, len(labels))
            # right_semantic_rgb = cmap(norm(right_semantic))
            # # # cv2.imwrite('label_r.jpg', right_semantic_rgb)
            # plt.imsave('right_semantic_rgb.png', right_semantic_rgb)
            # right_semantic_rgb2 = cmap(norm(label))
            # plt.imsave('right_semantic_rgb2.png', right_semantic_rgb2)
            # plt.imsave('left.png', left_img)
            # plt.imsave('right.png', right_img)

            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)
            # print(left_img.shape)

            # gx = processed(gx)
            # gy = processed(gy)

            disparity_4 = cv2.resize(disparity, (w//4, h//4), interpolation=cv2.INTER_NEAREST)
            disparity_8 = cv2.resize(disparity, (w//8, h//8), interpolation=cv2.INTER_NEAREST)
            disparity_16 = cv2.resize(disparity, (w//16, h//16), interpolation=cv2.INTER_NEAREST)
            label_2 = cv2.resize(label, (w//2, h//2), interpolation=cv2.INTER_NEAREST)
            label_4 = cv2.resize(label, (w//4, h//4), interpolation=cv2.INTER_NEAREST)

            return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity,
                    "disparity_4":disparity_4,
                    "disparity_8":disparity_8,
                    "disparity_16":disparity_16,
                    "label": label,
                    "label_2": label_2,
                    "label_4": label_4,#}
                    # "label_r": right_semantic,
                    # "edge": edges}
                    "gx": gx.float(),
                    "gy": gy.float()}
        else:
            w, h = left_img.size

            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)
            # gx = processed(gx)
            # gy = processed(gy)

            return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity,
                    "label": label,
                    "top_pad": 0,
                    "right_pad": 0,
                    # "disparity_low":disparity_low,
                    "left_filename": self.left_filenames[index],#}
                    "gx": gx.float(),
                    "gy": gy.float()}
