import os
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2
from datasets.data_io import get_transform, read_all_lines, pfm_imread
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt
import torchvision
from . import flow_transforms_c



class CityscapesDataset(Dataset):
    def __init__(self, CityscapesDataset, list_filename, training):
        self.cityscapes_c = CityscapesDataset
        self.left_filenames, self.right_filenames, self.disp_filenames, self.label_filenames = self.load_path(list_filename)
        self.training = training
        if self.training:
            assert self.disp_filenames is not None

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        if len(splits[0]) == 2:  # ground truth not available
            return left_images, right_images, None
        else:
            disp_images = [x[2] for x in splits]
            label_images = [x[3] for x in splits]
            # disp_images.split('/')[0] + "/obj_map/" + disp_images.split('/')[-1]
            return left_images, right_images, disp_images, label_images

    def load_image(self, filename):
        return Image.open(filename).convert('RGB')

    def load_disp(self, filename):
        data = Image.open(filename)
        data = np.array(data, dtype=np.float32) / 256.
        return data

    def load_label(self, filename):
        ignore_label = 19
        class_map =  {-1: ignore_label, 0: ignore_label, 
                              1: ignore_label, 2: ignore_label, 
                              3: ignore_label, 4: ignore_label, 
                              5: ignore_label, 6: ignore_label, 
                              7: 0, 8: 1, 9: ignore_label, 
                              10: ignore_label, 11: 2, 12: 3, 
                              13: 4, 14: ignore_label, 15: ignore_label, 
                              16: ignore_label, 17: 5, 18: ignore_label, 
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
                              25: 12, 26: 13, 27: 14, 28: 15, 
                              29: ignore_label, 30: ignore_label, 
                              31: 16, 32: 17, 33: 18}
        data = Image.open(filename)
        data = np.array(data)
        data_ = data.copy() # np.zeros(data.shape, dtype=np.int8)
        for k,v in class_map.items():
            data[data_ == k] = v
        return data

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        
        left_name = self.left_filenames[index].split('/')[1]

        self.datapath = self.cityscapes_c

        left_img = self.load_image(os.path.join(self.datapath, self.left_filenames[index]))
        right_img = self.load_image(os.path.join(self.datapath, self.right_filenames[index]))

        if self.disp_filenames:  # has disparity ground truth
            disparity = self.load_disp(os.path.join(self.datapath, self.disp_filenames[index]))
            label = self.load_label(os.path.join(self.datapath, self.label_filenames[index]))
        else:
            disparity = None
            label = None

        if self.training:
            th, tw = 256, 512
            random_brightness = np.random.uniform(0.5, 2.0, 2)
            random_gamma = np.random.uniform(0.8, 1.2, 2)
            random_contrast = np.random.uniform(0.8, 1.2, 2)
            random_saturation = np.random.uniform(0, 1.4, 2)
            left_img = torchvision.transforms.functional.adjust_brightness(left_img, random_brightness[0])
            left_img = torchvision.transforms.functional.adjust_gamma(left_img, random_gamma[0])
            left_img = torchvision.transforms.functional.adjust_contrast(left_img, random_contrast[0])
            right_img = torchvision.transforms.functional.adjust_brightness(right_img, random_brightness[1])
            right_img = torchvision.transforms.functional.adjust_gamma(right_img, random_gamma[1])
            right_img = torchvision.transforms.functional.adjust_contrast(right_img, random_contrast[1])
            left_img = torchvision.transforms.functional.adjust_saturation(left_img, random_saturation[0])
            right_img = torchvision.transforms.functional.adjust_saturation(right_img, random_saturation[1])
            right_img = np.array(right_img)
            left_img = np.array(left_img)

            # geometric unsymmetric-augmentation
            angle = 0
            px = 0
            if np.random.binomial(1, 0.5):
                # angle = 0.1;
                # px = 2
                angle = 0.05
                px = 1
            co_transform = flow_transforms_c.Compose([
                flow_transforms_c.RandomVdisp(angle, px),
                # flow_transforms_c.Scale(np.random.uniform(self.rand_scale[0], self.rand_scale[1]), order=self.order),
                flow_transforms_c.RandomCrop((th, tw)),
            ])
            
            augmented, disparity, label = co_transform([left_img, right_img], disparity, label)
            left_img = augmented[0]
            right_img = augmented[1]

            right_img.flags.writeable = True
            if np.random.binomial(1,0.2):
              sx = int(np.random.uniform(35,100))
              sy = int(np.random.uniform(25,75))
              cx = int(np.random.uniform(sx,right_img.shape[0]-sx))
              cy = int(np.random.uniform(sy,right_img.shape[1]-sy))
              right_img[cx-sx:cx+sx,cy-sy:cy+sy] = np.mean(np.mean(right_img,0),0)[np.newaxis,np.newaxis]

            # to tensor, normalize
            disparity = np.ascontiguousarray(disparity, dtype=np.float32)
            label = np.ascontiguousarray(label, dtype=np.float32)


            disparity_low = cv2.resize(disparity, (tw//4, th//4), interpolation=cv2.INTER_NEAREST)
            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)

            return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity,
                    "disparity_low": disparity_low,
                    "label": label}

        else:
            w, h = left_img.size

            # normalize
            processed = get_transform()
            left_img = processed(left_img).numpy()
            right_img = processed(right_img).numpy()

            # pad to size 1248x384
            # top_pad = 1024 - h
            # right_pad = 2048 - w
            # assert top_pad > 0 and right_pad > 0
            # pad images
            # left_img = np.lib.pad(left_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
            # right_img = np.lib.pad(right_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant',
            #                        constant_values=0)
            # pad disparity gt
            # if disparity is not None:
            #     assert len(disparity.shape) == 2
            #     disparity = np.lib.pad(disparity, ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
            #     label = np.lib.pad(label, ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)


            if disparity is not None:
                return {"left": left_img,
                        "right": right_img,
                        "disparity": disparity,
                        "label": label,}
                        # "top_pad": top_pad,
                        # "right_pad": right_pad}
            else:
                return {"left": left_img,
                        "right": right_img,
                        "top_pad": top_pad,
                        "right_pad": right_pad,
                        "left_filename": self.left_filenames[index],
                        "right_filename": self.right_filenames[index]}
