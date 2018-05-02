import sys
sys.path.append("..") # add ".." to the system path -> relative path
# import download_utils

#download_utils.link_all_keras_resources()

import tensorflow as tf
from tensorflow.contrib import keras
import numpy as np
import matplotlib.pyplot as plt
L = keras.layers
K = keras.backend
import tqdm
import utils
import time
import zipfile
import json
from collections import defaultdict
import re
import random
from random import choice
import os
from constant import Constant


# we take the last hidden layer of IncetionV3 as an image embedding
def get_cnn_encoder():

    K.set_learning_phase(False) # Sets the learning phase to a fixed value

    model = keras.applications.InceptionV3(include_top = False)

    preprocess_for_model = keras.applications.inception_v3.preprocess_input # a function

    model = keras.models.Model(model.inputs, keras.layers.GlobalAveragePooling2D()(model.output)) # add another layer on top of the current model

    return model, preprocess_for_model

