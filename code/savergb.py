from annotation import get_dense_label
from data import get_rgbd
from math import exp
import numpy as np
import skimage.io as sio
from CRF_optim import annotation_gradient_optim
import random
from skimage.transform import resize
import scipy

# Produce and save depth predictions for chosen images

idx_list = [233, 224, 275, 283, 290, 324, 354, 465, 25, 14]
cycle = 0
for idx in idx_list:
    rgb, depth = get_rgbd(idx, 480, 640)
    scipy.misc.imsave('../result_image/'+str(idx)+'rgb.pdf', rgb)

    ma, mi = np.max(depth), np.min(depth)
    depth = (depth - mi)/(ma-mi)
    scipy.misc.imsave('../result_image/'+str(idx)+'gtdepth.pdf', depth)
    anno = annotation_gradient_optim(idx, cycle)
    scipy.misc.imsave('../result_image/'+str(idx)+'preddepth.png', anno)
    cycle = cycle + 1