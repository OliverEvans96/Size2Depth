from data import get_rgbd
import tensorflow as tf
from tf_util import *
from VGG import get_VGG

def get_train_op(ind):
    (rgb, depth) = get_rgbd(ind)

rgb_list = rgb.tolist()

def get_model(rgb, div):
    conv_feature = get_VGG(rgb)
    fc1 = fc(conv_feature, 1024, 'fc1' + str(div))
    fc2 = fc(fc1, 256, 'fc2' + str(div))
    depth = fc(fc2, div*div, 'depth' + str(div))
    return depth

def get_loss(depth, gt_depth, div):
    l2_loss = tf.reduce_mean(tf.square(depth - gt_depth))
    tf.add_to_collection('loss' + str(div), l2_loss)
    return tf.add_n(tf.get_collection('loss' + str(div)), name = 'total_loss'+ str(div))