import sys
sys.path.append("..") # add ".." to the system path -> relative path

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

import tqdm
import utils
import time
import zipfile
import json
import re
import random
import os
import constant
import preparation

from tensorflow.contrib import keras
from collections import defaultdict
L = keras.layers
K = keras.backend

# preparation.load_encoder()
train_img_embeds, train_img_fns, val_img_embeds, val_img_fns = preparation.initialize()

print(type(train_img_fns))
c = np.random.choice(len(train_img_fns), 10)
print(np.array(train_img_fns)[c])