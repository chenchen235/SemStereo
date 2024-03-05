import os
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from datasets.data_io import get_transform, read_all_lines, pfm_imread
import cv2
import torchvision
import scipy.signal as sig


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

        # img = (left_img - np.mean(left_img)) / np.std(left_img)
        # kx = np.array([[-1, 0, 1]])
        # ky = np.array([[-1], [0], [1]])
        # print(img.shape)
        # # 对每个通道进行卷积操作
        # gx = np.zeros_like(img)
        # gy = np.zeros_like(img)
        # for channel in range(3):
        #     gx[:, :, channel] = sig.convolve2d(img[:, :, channel], kx, 'same')
        #     gy[:, :, channel] = sig.convolve2d(img[:, :, channel], ky, 'same')
        # gx = sig.convolve2d(img, kx, 'same')
        # gy = sig.convolve2d(img, ky, 'same')

        if self.training:

            w, h = left_img.size

            # crop_w, crop_h = 896, 896
            # left_img = left_img.crop((w - crop_w, h - crop_h, w, h))
            # right_img = right_img.crop((w - crop_w, h - crop_h, w, h))
            # disparity = disparity[h - crop_h:h, w - crop_w: w]
            # label = label[h - crop_h:h, w - crop_w: w]
            # disparity_low = cv2.resize(disparity, (crop_w//4, crop_h//4), interpolation=cv2.INTER_NEAREST)

            # agu
            # random_brightness = np.random.uniform(0.5, 2.0, 2)
            # random_gamma = np.random.uniform(0.8, 1.2, 2)
            # random_contrast = np.random.uniform(0.8, 1.2, 2)
            # random_saturation = np.random.uniform(0, 1.4, 2)
            # left_img = torchvision.transforms.functional.adjust_brightness(left_img, random_brightness[0])
            # left_img = torchvision.transforms.functional.adjust_gamma(left_img, random_gamma[0])
            # left_img = torchvision.transforms.functional.adjust_contrast(left_img, random_contrast[0])
            # left_img = torchvision.transforms.functional.adjust_saturation(left_img, random_saturation[0])

            # right_img = torchvision.transforms.functional.adjust_brightness(right_img, random_brightness[1])
            # right_img = torchvision.transforms.functional.adjust_gamma(right_img, random_gamma[1])
            # right_img = torchvision.transforms.functional.adjust_contrast(right_img, random_contrast[1])
            # right_img = torchvision.transforms.functional.adjust_saturation(right_img, random_saturation[1])

            # right_img = np.array(right_img)
            # left_img = np.array(left_img)

            # right_img.flags.writeable = True
            # if np.random.binomial(1,0.2):
            #   sx = int(np.random.uniform(25,100))
            #   sy = int(np.random.uniform(25,100)) 
            #   cx = int(np.random.uniform(sx,right_img.shape[0]-sx))
            #   cy = int(np.random.uniform(sy,right_img.shape[1]-sy))
            #   right_img[cx-sx:cx+sx,cy-sy:cy+sy] = np.mean(np.mean(right_img,0),0)[np.newaxis,np.newaxis]
            
            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)
            # gx = processed(gx)
            # gy = processed(gy)

            disparity_low = cv2.resize(disparity, (w//4, h//4), interpolation=cv2.INTER_NEAREST)
            label_2 = cv2.resize(label, (w//2, h//2), interpolation=cv2.INTER_NEAREST)
            label_4 = cv2.resize(label, (w//4, h//4), interpolation=cv2.INTER_NEAREST)

            return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity,
                    "disparity_low":disparity_low,
                    "label": label,
                    "label_2": label_2,
                    "label_4": label_4,}
                    # "gx": gx,
                    # "gy": gy}
        else:
            w, h = left_img.size
#             crop_w, crop_h = 960, 512
#             # crop_w, crop_h = 960, 960
            
#             left_img = left_img.crop((w - crop_w, h - crop_h, w, h))
#             right_img = right_img.crop((w - crop_w, h - crop_h, w, h))
#             disparity = disparity[h - crop_h:h, w - crop_w: w]
            # disparity_low = cv2.resize(disparity, (w//4, h//4), interpolation=cv2.INTER_NEAREST)

            # crop_w, crop_h = 896, 896
            # left_img = left_img.crop((w - crop_w, h - crop_h, w, h))
            # right_img = right_img.crop((w - crop_w, h - crop_h, w, h))
            # disparity = disparity[h - crop_h:h, w - crop_w: w]
            # label = label[h - crop_h:h, w - crop_w: w]

            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)

            return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity,
                    "label": label,
                    "top_pad": 0,
                    "right_pad": 0,
                    # "disparity_low":disparity_low,
                    "left_filename": self.left_filenames[index],}
                    # "gx": gx,
                    # "gy": gy}
