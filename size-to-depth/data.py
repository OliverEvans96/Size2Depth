import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

mat = '/home/wyr/Downloads/nyu_depth_v2_labeled.mat'
data = sio.loadmat(mat)

rgb = mat['images']
depth = mat['depths']

rgb = rgb.transpose(3,0,1,2)
depth = depth.transpose(3,0,1,2)

def get_rgbd(ind):
    return (rgb[ind], depth[ind])

