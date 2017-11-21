from data import get_rgbd
import tensorflow as tf
from tf_util import *
from VGG import get_VGG

ind = 233
(rgb, depth) = get_rgbd(ind)
rgb_list = rgb.tolist()


def get_model(rgb, div):
    depth_list = []
    with tf.variable_scope('patchCNN') as scope:
        for i in range(div):
            for j in range(div):
                patch = tf.slice(rgb, [i*div, j*div], [div, div], 'patch' + str(i*div+j))
                conv_feature = get_VGG(patch)
                fc1 = fc(conv_feature, 1024, 'fc1' + str(div))
                fc2 = fc(fc1, 256, 'fc2' + str(div))
                scope.reuse_variables()
                depth = fc(fc2, div*div, 'depth' + str(i*div+j))
                depth_list.append(depth)
    depths = tf.reshape(tf.stack(depth_list, axis = 1), [div, div])
    return depths

def get_loss(depth, gt_depth, div):
    l2_loss = tf.reduce_mean(tf.square(depth - gt_depth))
    tf.add_to_collection('loss' + str(div), l2_loss)
    return tf.add_n(tf.get_collection('loss' + str(div)), name = 'total_loss'+ str(div))