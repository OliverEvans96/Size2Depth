from annotation import get_dense_label
from data import get_rgbd
from math import exp
import numpy as np
import skimage.io as sio

# Transforming rgb image to grayscale
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def get_idx(x,y,len):
    return x*len+y

# Optimizing CRF loss to give predicted depth map
def annotation_gradient_optim(rgb, label):
    lmd = 700
    beta = 0.05  #weights in CRF loss term

    pr, pc = rgb.shape[0:2]
    c = label.flatten()
    len = c.shape[0]
    e = np.eye(len)
    gray = rgb2gray(rgb)
    gray = gray.flatten()
    w = np.zeros([len, len])
    grayr = gray.reshape([1,-1])
    grayc = gray.reshape([-1,1])
    print "calculate w"
    dim0 = []
    dim1 = []
    for i in range(pr):
        for j in range(pc):
            x_idx = get_idx(i,j,pc)
            if i>0:
                y_idx = get_idx(i-1, j, pc)
                dim0.append(x_idx)
                dim1.append(y_idx)
                dim1.append(x_idx)
                dim0.append(y_idx)
            if j>0:
                y_idx = get_idx(i, j-1, pc)
                dim0.append(x_idx)
                dim1.append(y_idx)
                dim1.append(x_idx)
                dim0.append(y_idx)
    w[dim0, dim1] = np.exp(-beta*np.abs(grayr-grayc))[dim0, dim1]
    d = np.sum(w, axis = 1)
    d=np.diag(d)
    wprime = e + lmd * (d - w)
    k = wprime + wprime.T
    print "Inversing"
    y_target = np.dot(2 * c, np.linalg.inv(k))
    y_target = y_target.reshape(rgb.shape[0:2])


    # normalizing output to be within [0, 1]
    max = y_target.max()
    min = y_target.min()
    #print max, min
    y_target = (y_target - min) / (max - min)
    max = y_target.max()
    min = y_target.min()
    #print max, min
    y_target = (y_target - min) / (max - min)
    sio.imshow(y_target)
    sio.show()
    return y_target
