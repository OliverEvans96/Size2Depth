import skimage.io as sio
import numpy as np
import h5py
from skimage.transform import resize

mat = 'nyu_depth_v2_labeled.mat'
data = h5py.File(mat)
# Function to read rgb and ground truth depth images from dataset
# idx: index of image
# resr, resc: resolution of returned image
def get_rgbd(idx, resr, resc):
    rgb_ = data['images'][idx]
    depth_ = data['depths'][idx]
    #print rgb_.shape
    #print depth_.shape

    rgb = np.empty([480, 640, 3])
    rgb[:,:,0] = rgb_[0,:,:].T
    rgb[:,:,1] = rgb_[1,:,:].T
    rgb[:,:,2] = rgb_[2,:,:].T

    rgb = rgb.astype('float32')
    sio.imshow(rgb / 255)
    sio.show()
    rgb = resize(rgb/255, [resr,resc])*255
    depth = depth_.T
    sio.imshow(depth)
    sio.show()
    return (rgb, depth)

