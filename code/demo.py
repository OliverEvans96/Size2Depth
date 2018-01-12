from data import get_rgbd
import numpy as np
import skimage.io as sio
from CRF_optim import annotation_gradient_optim
import random
from skimage.transform import resize
import scipy
from annotation import get_label, get_dense_label

# Comparing performance of Size-to-depth method and label depth directly
def view_labeled_results(idx, cycle):
    rgb, depth = get_rgbd(idx, 63, 84)
    anno = annotation_gradient_optim(rgb, get_dense_label(cycle))

    return

def label_new_image(idx):
    rgb, depth = get_rgbd(idx, 63, 84)

    label = get_label(rgb)
    rgb = resize(rgb, (63, 84))
    anno = annotation_gradient_optim(rgb, label)