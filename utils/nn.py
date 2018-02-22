import tensorflow as tf
import tensorflow.contrib.layers as layers

class NN(object):
    def __init__(self, config):
        self.config = config
        self.is_train = True if config.phase == 'train' else False
        self.train_cnn = self.is_train and config.train_cnn
        self.prepare()

    def prepare(self):
        """ Setup the weight initalizers and regularizers. """
        config = self.config

        self.conv_kernel_initializer = layers.xavier_initializer()

        if self.train_cnn and config.conv_kernel_regularizer_scale > 0:
            self.conv_kernel_regularizer = layers.l2_regularizer(
                scale = config.conv_kernel_regularizer_scale)
        else:
            self.conv_kernel_regularizer = None

        if self.train_cnn and config.conv_activity_regularizer_scale > 0:
            self.conv_activity_regularizer = layers.l1_regularizer(
                scale = config.conv_activity_regularizer_scale)
        else:
            self.conv_activity_regularizer = None

        self.fc_kernel_initializer = tf.random_uniform_initializer(
            minval = -config.fc_kernel_initializer_scale,
            maxval = config.fc_kernel_initializer_scale)

        if self.is_train and config.fc_kernel_regularizer_scale > 0:
            self.fc_kernel_regularizer = layers.l2_regularizer(
                scale = config.fc_kernel_regularizer_scale)
        else:
            self.fc_kernel_regularizer = None

        if self.is_train and config.fc_activity_regularizer_scale > 0:
            self.fc_activity_regularizer = layers.l1_regularizer(
                scale = config.fc_activity_regularizer_scale)
        else:
            self.fc_activity_regularizer = None

    def conv2d(self,
               inputs,
               filters,
               kernel_size = (3, 3),
               strides = (1, 1),
               activation = tf.nn.relu,
               use_bias = True,
               name = None):
        """ 2D Convolution layer. """
        if activation is not None:
            activity_regularizer = self.conv_activity_regularizer
        else:
            activity_regularizer = None
        return tf.layers.conv2d(
            inputs = inputs,
            filters = filters,
            kernel_size = kernel_size,
            strides = strides,
            padding='same',
            activation = activation,
            use_bias = use_bias,
            trainable = self.train_cnn,
            kernel_initializer = self.conv_kernel_initializer,
            kernel_regularizer = self.conv_kernel_regularizer,
            activity_regularizer = activity_regularizer,
            name = name)

    def max_pool2d(self,
                   inputs,
                   pool_size = (2, 2),
                   strides = (2, 2),
                   name = None):
        """ 2D Max Pooling layer. """
        return tf.layers.max_pooling2d(
            inputs = inputs,
            pool_size = pool_size,
            strides = strides,
            padding='same',
            name = name)

    def dense(self,
              inputs,
              units,
              activation = tf.tanh,
              use_bias = True,
              name = None):
        """ Fully-connected layer. """
        if activation is not None:
            activity_regularizer = self.fc_activity_regularizer
        else:
            activity_regularizer = None
        return tf.layers.dense(
            inputs = inputs,
            units = units,
            activation = activation,
            use_bias = use_bias,
            trainable = self.is_train,
            kernel_initializer = self.fc_kernel_initializer,
            kernel_regularizer = self.fc_kernel_regularizer,
            activity_regularizer = activity_regularizer,
            name = name)

    def dropout(self,
                inputs,
                name = None):
        """ Dropout layer. """
        return tf.layers.dropout(
            inputs = inputs,
            rate = self.config.fc_drop_rate,
            training = self.is_train)

    def batch_norm(self,
                   inputs,
                   name = None):
        """ Batch normalization layer. """
        return tf.layers.batch_normalization(
            inputs = inputs,
            training = self.train_cnn,
            trainable = self.train_cnn,
            name = name
        )
