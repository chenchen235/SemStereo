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


class WhuDataset(Dataset):
    def __init__(self, datapath, list_filename, training):
        self.datapath = datapath
        self.left_filenames, self.right_filenames, self.disp_filenames = self.load_path(list_filename)
        self.training = training

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        # lines = lines[0:10]
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        disp_images = [x[2] for x in splits]
        return left_images, right_images, disp_images

    def load_image(self, filename):
        return Image.open(filename).convert('RGB')        

    def load_disp(self, filename):
        data = Image.open(filename)
        data = np.ascontiguousarray(data, dtype=np.float32)/256.
        return data

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        left_img = self.load_image(os.path.join(self.datapath, self.left_filenames[index]))
        right_img = self.load_image(os.path.join(self.datapath, self.right_filenames[index]))
        disparity = self.load_disp(os.path.join(self.datapath, self.disp_filenames[index]))

        img = (left_img.convert("L") - np.mean(left_img.convert("L"))) / np.std(left_img.convert("L"))
        kx = np.array([[-1, 0, 1]])
        ky = np.array([[-1], [0], [1]])
        gx = np.zeros_like(img)
        gy = np.zeros_like(img)
        gx = torch.Tensor(sig.convolve2d(img, kx, 'same')).unsqueeze(0)
        gy = torch.Tensor(sig.convolve2d(img, ky, 'same')).unsqueeze(0)

        # print(disparity.max(), disparity.min())
        if self.training:

            w, h = left_img.size

            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)

            disparity_4 = cv2.resize(disparity, (w//4, h//4), interpolation=cv2.INTER_NEAREST)
            disparity_8 = cv2.resize(disparity, (w//8, h//8), interpolation=cv2.INTER_NEAREST)
            disparity_16 = cv2.resize(disparity, (w//16, h//16), interpolation=cv2.INTER_NEAREST)

            return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity,
                    "disparity_4":disparity_4,
                    "disparity_8":disparity_8,
                    "disparity_16":disparity_16,
                    "gx": gx.float(),
                    "gy": gy.float()}

        else:
            w, h = left_img.size

            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)

            return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity,
                    "top_pad": 0,
                    "right_pad": 0,
                    # "disparity_low":disparity_low,
                    "left_filename": self.left_filenames[index],
                    "gx": gx.float(),
                    "gy": gy.float()}