import numpy as np
import tensorflow as tf


def cpu_var(name, shape, init, dtype = tf.float32):
    """Helper to create a Variable stored on CPU memory.

    Args:
        name: name of the variable
        shape: list of ints
        init: initializer for Variable
        dtype: variable type

    Returns:
        Variable Tensor
    """
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, dtype, init)
    return var

def reg_var(name, shape, stddev, reg, use_xavier = True):
    """Helper to create a regularized variable.

    Args:
        name: name of the variable
        shape: list of ints
        stddev: standard deviation for initialization
        reg: regularization strength
        use_xavier: flag indicating usage of xavier initialization


    Returns:
        Variable Tensor
    """
    if use_xavier:
        init = tf.contrib.layers.xavier_initializer()
    else:
        init = tf.truncated_normal_initializer(stddev = stddev)
    var = cpu_var(name, shape, init)
    if reg is not None:
        reg_loss = tf.multiply(tf.nn.l2_loss(var), reg, 'reg_loss')
        tf.add_to_collection('loss', reg_loss)
    return var


def conv2d(input,
           kernel_num,
           kernel_sz,
           scope_name,
           stride = [1, 1],
           padding='SAME',
           stddev=1e-3,
           reg=0.0,
           activation = tf.nn.relu,
           use_xavier = True,
           bn = False,
           bn_decay = 0.9,
           is_training = None):
    with tf.variable_scope(scope_name):
        kernel_h, kernel_w = kernel_sz
        channel_num = input.get_shape()[-1]
        weight_sz = [kernel_h, kernel_w,
                  channel_num, kernel_num]
        weight = reg_var(name = 'weight',
                         shape = weight_sz,
                         stddev = stddev, reg = reg,
                         use_xavier = use_xavier)
        stride_h, stride_w = stride
        out = tf.nn.conv2d(input, weight,
                           [1, stride_h, stride_w, 1],
                           padding = padding)
        bias = cpu_var('bias', [kernel_num],
                       tf.constant_initializer(0.0))
        out = tf.nn.bias_add(out, bias)
        if bn:
            output = bn_conv2d(out, is_training, bn_decay)
        if activation is not None:
            out=activation(out)
    return out


def conv3d(input,
           kernel_num,
           kernel_sz,
           scope_name,
           stride = [1, 1, 1],
           padding='SAME',
           stddev=1e-3,
           reg=0.0,
           activation = tf.nn.relu,
           use_xavier=True,
           bn = False,
           bn_decay = 0.9,
           is_training = None):
    with tf.variable_scope(scope_name):
        kernel_l, kernel_w, kernel_h = kernel_sz
        channel_num = input.get_shape()[-1]
        weight_sz = [kernel_l, kernel_w, kernel_h,
                  channel_num, kernel_num]
        weight = reg_var(name = 'weight',
                         shape = weight_sz,
                         stddev = stddev, reg = reg,
                         use_xavier = use_xavier)
        stride_l, stride_w, stride_h = stride
        out = tf.nn.conv3d(input, weight,
                           [1, stride_l, stride_w, stride_h, 1],
                           padding = padding)
        bias = cpu_var('bias', [kernel_num],
                       tf.constant_initializer(0.0))
        out = tf.nn.bias_add(out, bias)
        if bn:
            output = bn_conv3d(out, is_training, 'bn_conv3d', bn_decay)
        if activation is not None:
            out=activation(out)
    return out

def fc(input,
       hidden,
       scope_name,
       stddev=1e-3,
       reg=0.0,
       activation = tf.nn.relu,
       use_xavier=True,
       bn = False,
       bn_decay = 0.9,
       is_training = None):
    with tf.variable_scope(scope_name):
        input_num = input.get_shape()[-1]
        weight = reg_var(name = 'weight',
                         shape = [input_num, hidden],
                         stddev = stddev, reg = reg,
                         use_xavier=use_xavier)
        out = tf.matmul(input, weight)
        bias = cpu_var('bias', [hidden],
                       tf.constant_initializer(0.0))
        out = tf.nn.bias_add(out, bias)
        if bn:
            output = bn_fc(out, is_training, 'bn_fc', bn_decay)
        if activation is not None:
            out=activation(out)
    return out

def max_pool2d(input,
               kernel_sz,
               scope_name,
               stride = [2, 2],
               padding = 'VALID'):
    with tf.variable_scope(scope_name):
        kernel_h, kernel_w = kernel_sz
        stride_h, stride_w = stride
        out = tf.nn.max_pool(input,
                             ksize = [1, kernel_h, kernel_w, 1],
                             strides = [1, stride_h, stride_w, 1],
                             padding=padding)
    return out



def max_pool3d(input,
               kernel_sz,
               scope_name,
               stride = [2, 2, 2],
               padding = 'VALID'):
    with tf.variable_scope(scope_name):
        kernel_l, kernel_w, kernel_h = kernel_sz
        stride_l, stride_w, stride_h = stride
        out = tf.nn.max_pool3d(input,
                             ksize = [1, kernel_l, kernel_w, kernel_h, 1],
                             strides = [1, kernel_l, stride_w, stride_h, 1],
                             padding=padding)
    return out

def bn_template(input,
                is_training,
                scope_name,
                dims,
                bn_decay):
    with tf.variable_scope(scope_name):
        channel_num = input.get_shape()[-1]
        beta = tf.Variable(tf.constant(0,0, [channel_num]),
                           name = 'beta', trainable=True)
        gamma = tf.Variable(tf.constant(0,0, [channel_num]),
                           name = 'gamma', trainable=True)
        mean, var = tf.nn.moments(input, dims, name = 'moment')
        movingavg = tf.train.ExponentialMovingAverage(decay=bn_decay)

        avgop = tf.cond(is_training,
                               lambda: avgop.apply([mean, var]),
                               lambda: tf.no_op())

        def mean_var_with_update():
            with tf.control_dependencies([avgop]):
                return tf.identity(mean), tf.identity(var)

        # ema.average returns the Variable holding the average of var.
        mean, var = tf.cond(is_training,
                            mean_var_with_update,
                            lambda: (avgop.average(mean), avgop.average(var)))
        normed = tf.nn.batch_normalization(input, mean, var, beta, gamma, 1e-3)
    return normed

def bn_fc(input,
          is_training,
          scope_name,
          bn_decay):
    return bn_template(input, is_training, scope_name, [0,], bn_decay)

def bn_conv2d(input,
          is_training,
          scope_name,
          bn_decay):
    return bn_template(input, is_training, scope_name, [0,1,2], bn_decay)

def bn_conv3d(input,
          is_training,
          scope_name,
          bn_decay):
    return bn_template(input, is_training, scope_name, [0,1,2,3], bn_decay)