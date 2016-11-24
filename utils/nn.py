import math
import numpy as np
import tensorflow as tf
from tensorflow.python.training import moving_averages

def weight(name, shape, init='he', range=0.1, stddev=0.01, init_val=None, group_id=0):
    """ Get a weight variable. """
    if init_val != None:
        initializer = tf.constant_initializer(init_val)
    elif init == 'uniform':
        initializer = tf.random_uniform_initializer(-range, range)
    elif init == 'normal':
        initializer = tf.random_normal_initializer(stddev=stddev)
    elif init == 'he':
        fan_in, _ = _get_dims(shape)
        std = math.sqrt(2.0 / fan_in)
        initializer = tf.random_normal_initializer(stddev=std)
    elif init == 'xavier':
        fan_in, fan_out = _get_dims(shape)
        range = math.sqrt(6.0 / (fan_in + fan_out))
        initializer = tf.random_uniform_initializer(-range, range)
    else:
        initializer = tf.truncated_normal_initializer(stddev = stddev)

    var = tf.get_variable(name, shape, initializer = initializer)
    tf.add_to_collection('l2_'+str(group_id), tf.nn.l2_loss(var))
    return var

def bias(name, dim, init_val=0.0):
    """ Get a bias variable. """
    dims = dim if isinstance(dim, list) else [dim]
    return tf.get_variable(name, dims, initializer = tf.constant_initializer(init_val))

def nonlinear(x, nl=None):
    """ Apply a nonlinearity layer. """ 
    if nl == 'relu':
        return tf.nn.relu(x)
    elif nl == 'tanh':
        return tf.tanh(x)
    elif nl == 'sigmoid':
        return tf.sigmoid(x)
    else:
        return x

def convolution(x, k_h, k_w, c_o, s_h, s_w, name, init_w='he', init_b=0, stddev=0.01, padding='SAME', group_id=0):
    """ Apply a convolutional layer (with bias). """
    c_i = _get_shape(x)[-1]
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    with tf.variable_scope(name) as scope:
        w = weight('weights', [k_h, k_w, c_i, c_o], init=init_w, stddev=stddev, group_id=group_id)
        z = convolve(x, w)
        b = bias('biases', c_o, init_b)
        z = tf.nn.bias_add(z, b)
    return z

def convolution_no_bias(x, k_h, k_w, c_o, s_h, s_w, name, init_w='he', stddev=0.01, padding='SAME', group_id=0):
    """ Apply a convolutional layer (without bias). """
    c_i = _get_shape(x)[-1]
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    with tf.variable_scope(name) as scope:
        w = weight('weights', [k_h, k_w, c_i, c_o], init=init_w, stddev=stddev, group_id=group_id)
        z = convolve(x, w)
    return z

def fully_connected(x, output_size, name, init_w='he', init_b=0, stddev=0.01, group_id=0):
    """ Apply a fully-connected layer (with bias). """
    x_shape = _get_shape(x)
    input_dim = x_shape[-1]

    with tf.variable_scope(name) as scope:
        w = weight('weights', [input_dim, output_size], init=init_w, stddev=stddev, group_id=group_id)
        b = bias('biases', [output_size], init_b)
        z = tf.nn.xw_plus_b(x, w, b)
    return z

def fully_connected_no_bias(x, output_size, name, init_w='he', stddev=0.01, group_id=0):
    """ Apply a fully-connected layer (without bias). """
    x_shape = _get_shape(x)
    input_dim = x_shape[-1]

    with tf.variable_scope(name) as scope:
        w = weight('weights', [input_dim, output_size], init=init_w, stddev=stddev, group_id=group_id)
        z = tf.matmul(x, w)
    return z

def batch_norm(x, name, is_train, bn=True, nl='relu'):
    """ Apply a batch normalization layer and a nonlinearity layer. """
    if bn:
        x = _batch_norm(x, name, is_train)
    x = nonlinear(x, nl)
    return x

def _batch_norm(x, name, is_train):
    """ Apply a batch normalization layer. """
    with tf.variable_scope(name):
        inputs_shape = x.get_shape()
        axis = list(range(len(inputs_shape) - 1))
        param_shape = int(inputs_shape[-1])

        moving_mean = tf.get_variable('mean', [param_shape], initializer=tf.constant_initializer(0.0), trainable=False)
        moving_var = tf.get_variable('variance', [param_shape], initializer=tf.constant_initializer(1.0), trainable=False)

        beta = tf.get_variable('offset', [param_shape], initializer=tf.constant_initializer(0.0))
        gamma = tf.get_variable('scale', [param_shape], initializer=tf.constant_initializer(1.0))

        control_inputs = []

        def mean_var_with_update():
            mean, var = tf.nn.moments(x, axis)
            update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, 0.99)
            update_moving_var = moving_averages.assign_moving_average(moving_var, var, 0.99)
            control_inputs = [update_moving_mean, update_moving_var]
            return tf.identity(mean), tf.identity(var)

        def mean_var():
            mean = moving_mean
            var = moving_var            
            return tf.identity(mean), tf.identity(var)

        mean, var = tf.cond(is_train, mean_var_with_update, mean_var)

        with tf.control_dependencies(control_inputs):
            normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)

    return normed

def dropout(x, keep_prob, is_train):
    """ Apply a dropout layer. """
    return tf.cond(is_train, lambda: tf.nn.dropout(x, keep_prob), lambda: x)

def max_pool(x, k_h, k_w, s_h, s_w, name, padding='SAME'):
    """ Apply a max pooling layer. """
    return tf.nn.max_pool(x, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding, name=name)

def avg_pool(x, k_h, k_w, s_h, s_w, name, padding='SAME'):
    """ Apply an average pooling layer. """
    return tf.nn.avg_pool(x, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding, name=name)

def _get_dims(shape):
    """ Get the fan-in and fan-out of a Tensor. """
    fan_in = np.prod(shape[:-1])
    fan_out = shape[-1]
    return fan_in, fan_out

def _get_shape(x):
    """ Get the shape of a Tensor. """
    return x.get_shape().as_list()

