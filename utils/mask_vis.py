import numpy as np
import torch


def feature_vis(feats):
    # Define the number of classes
    num_classes = feats.size(1)

    # Define colors for each class (can be changed as per requirement)
    colors = np.array([[0,0,0],       # Class 0: Black
                       [255,0,0],     # Class 1: Red
                       [0,255,0],     # Class 2: Green
                       [0,0,255],     # Class 3: Blue
                       [255,255,0],   # Class 4: Yellow
                       [0,255,255]])   # Class 5: Cyan

    # Read the semantic segmentation output and convert to numpy array
    seg_output = feats.cpu().detach().numpy() # 4 6 1024 1024  [0, :, :, :].unsqueeze(0)
    b, c, w, h =  seg_output.shape

    # Create an empty mask to store the final visualization result
    mask = np.zeros((b, 3, w, h)) # 4 3 1024 1024

    # Use numpy matrix operation for iteration and color assignment
    class_ids = np.argmax(seg_output, axis = 1) # 4 1024 1024
    # class_ids = np.transpose(class_ids, (1, 2, 0)).shape # 1024 1024 4
    a = colors[class_ids]
    # a = class_ids[colors, :, :]
    mask[:, :, :, :] = np.transpose(a, (0, 3, 1, 2))

    return mask
