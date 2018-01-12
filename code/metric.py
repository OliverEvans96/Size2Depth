from annotation import get_dense_label
from data import get_rgbd
from math import exp
import numpy as np
import skimage.io as sio
from CRF_optim import annotation_gradient_optim
import random
from skimage.transform import resize
import scipy


# Comparing performance of Size-to-depth method and label depth directly

idx_list = [233, 224, 275, 283, 290, 324, 354, 465, 25, 14]
cycle = 0
mse_list = []
corr_list = []
rev_list = []
for idx in idx_list:
    rgb, depth = get_rgbd(idx, 63, 84)
    #scipy.misc.imsave('../result_image/'+str(idx)+'rgb.pdf', rgb)

    # normalize ground truth depth to [0, 1] so that it is comparable to predicted depth
    ma, mi = np.max(depth), np.min(depth)
    depth = (depth - mi)/(ma-mi)
    #scipy.misc.imsave('../result_image/'+str(idx)+'gtdepth.pdf', depth)
    anno = annotation_gradient_optim(rgb, get_dense_label(cycle))
    #scipy.misc.imsave('../result_image/'+str(idx)+'preddepth.png', anno)
    gt, pred = ([],[])
    for i in xrange(1000):
        posx = int(random.random()*63)
        posy = int(random.random()*84)
        gt.append(depth[posx, posy])
        pred.append(anno[posx,posy])
    #print gt
    #print pred
    gt, pred = np.array(gt), np.array(pred)

    print "mse = ",np.mean(np.square(gt-pred))
    mse_list.append(np.mean(np.square(gt-pred)))
    corr = np.dot(gt, pred) / np.linalg.norm(gt, 2) / np.linalg.norm(pred,2)
    print "corr = ",corr
    corr_list.append(corr)
    num = 0.0
    for i in range(gt.size):
        for j in range(i,pred.size,1):
           if gt[i] > gt[j] and pred[i]<pred[j]: num = num+1
           if gt[i] < gt[j] and pred[i]>pred[j]: num = num+1

    #print num
    print "rev = ",num/gt.size/(gt.size-1)*2
    rev_list.append(num/gt.size/(gt.size-1)*2)
    cycle = cycle + 1
print "mean_mse = ", np.mean(np.array(mse_list))
print "mean_corr = ", np.mean(np.array(corr_list))
print "mean rev = ", np.mean(np.array(rev_list))
