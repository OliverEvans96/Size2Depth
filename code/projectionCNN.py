import tensorflow as tf
from tf_util import *
#####need Upconv implementation

def get_model(rgb, div):
    proj_list = []
    with tf.variable_scope("projCNN") as scope:
        for i in range(div):
            for j in range(div):
                patch = tf.slice(rgb, [i * div, j * div], [div, div], name = "patch" + str(i * div + j))
                conv_feature = get_inter_conv(patch)
                proj_list.append(conv_feature)
    projs = tf.reshape(tf.stack(proj_list), [div, div, -1])
    return projs
