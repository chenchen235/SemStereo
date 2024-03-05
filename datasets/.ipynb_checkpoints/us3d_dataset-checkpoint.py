import os
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from datasets.data_io import get_transform, read_all_lines, pfm_imread
import cv2

class Us3dDataset(Dataset):
    def __init__(self, datapath, list_filename, training):
        self.datapath = datapath
        self.left_filenames, self.right_filenames, self.disp_filenames, self.label_filenames = self.load_path(list_filename)
        self.training = training

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        #lines = lines[0:10]
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

        if self.training:
            w, h = left_img.size

            # to tensor, normalize
            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)
            disparity_low = cv2.resize(disparity, (w//4, h//4), interpolation=cv2.INTER_NEAREST)
            

            return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity,
                    "disparity_low":disparity_low,
                    "label": label}
        else:
            # w, h = left_img.size
#             crop_w, crop_h = 960, 512
#             # crop_w, crop_h = 960, 960
            
#             left_img = left_img.crop((w - crop_w, h - crop_h, w, h))
#             right_img = right_img.crop((w - crop_w, h - crop_h, w, h))
#             disparity = disparity[h - crop_h:h, w - crop_w: w]
            # disparity_low = cv2.resize(disparity, (w//4, h//4), interpolation=cv2.INTER_NEAREST)

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
                    "left_filename": self.left_filenames[index]}
