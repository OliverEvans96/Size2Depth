from data import get_rgbd
import numpy as np
import skimage.io as sio
from CRF_optim import annotation_gradient_optim
import random
from skimage.transform import resize
import scipy
from annotation import get_label, get_dense_label

# Showing sample image #number
def view_labeled_result(number):
    print 'Showing sample image #%d'%number
    idx_list = [233, 224, 275, 283, 290, 324, 354, 465, 25, 14]
    print '-----optimization infos-----'
    rgb, depth = get_rgbd(idx_list[number], 63, 84)
    anno = annotation_gradient_optim(rgb, get_dense_label(number))
    print 'This is the predicted depth map'
    return

# Label #idx image in the dataset
def label_new_image(idx):
    print 'This is the original image'
    rgb, depth = get_rgbd(idx, 63, 84)
    print 'Now please assign size label to each of the 7 by 7 grid like below:\n1 1 1\n1 1 1\n1 1 1\n---------------'

    label = get_label(rgb)
    print '-----optimization infos-----'
    anno = annotation_gradient_optim(rgb, label)
    print 'This is the predicted depth map'


#view_labeled_result(0-9)
#label_new_image(0-1449)