import numpy as np
import cv2

def linear_mapping(img):
    return (img - img.min()) / (img.max() - img.min())

# pre-processing the image...
def pre_process2(img):
    # get the size of the img...
    height, width = img.shape
    img = np.log(img + 1)
    img = (img - np.mean(img)) / (np.std(img) + 1e-5)
    # use the hanning window...
    window = window_func_2d(height, width)
    img = img * window

    return img

def window_func_2d(height, width):
    win_col = np.hanning(width)
    win_row = np.hanning(height)
    mask_col, mask_row = np.meshgrid(win_col, win_row)

    win = mask_col * mask_row

    return win

def random_warp(img):
    a = -180 / 16
    b = 180 / 16
    r = a + (b - a) * np.random.uniform()
    # rotate the image...
    matrix_rot = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), r, 1)
    img_rot = cv2.warpAffine(np.uint8(img * 255), matrix_rot, (img.shape[1], img.shape[0]))
    img_rot = img_rot.astype(np.float32) / 255
    return img_rot

def pre_process(img):
    img = img/ 255
    img = img - np.mean(img)
    img = img / np.std(img)
    return img

def iou(ground_truth, pred):
    # coordinates of the area of intersection.
    ix1 = np.maximum(ground_truth.xpos, pred.xpos)
    iy1 = np.maximum(ground_truth.ypos, pred.ypos)
    ix2 = np.minimum(ground_truth.xpos+ground_truth.width, pred.xpos+pred.width)
    iy2 = np.minimum(ground_truth.ypos+ground_truth.height, pred.ypos+pred.height)
     
    # Intersection height and width.
    i_height = np.maximum(iy2 - iy1, np.array(0.))
    i_width = np.maximum(ix2 - ix1, np.array(0.))
     
    area_of_intersection = i_height * i_width
     
    area_of_union = ground_truth.height * ground_truth.width + pred.height * pred.width - area_of_intersection
     
    iou = area_of_intersection / area_of_union
     

    return iou