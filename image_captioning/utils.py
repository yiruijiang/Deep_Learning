import os
import queue
import threading
import zipfile
import tqdm
import cv2
import numpy as np
import pickle

# Note: python supports dynamic type inference
def image_center_crop(img):
    h, w = img.shape[0], img.shape[1]
    pad_left = pad_right = pad_top = pad_bottom = 0
    diff = abs(h - w)
    half_diff = diff // 2
    if h > w:
        pad_top = diff - half_diff
        pad_bottom = half_diff
    else:
        pad_left = diff - half_diff
        pad_right = half_diff
    return img[pad_top : h - pad_bottom, pad_left : h - pad_right]

def decode_image_from_buf(buf):
    img = cv2.imdecode(np.asarray(bytearray(buf), dtype=np.uint8), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

