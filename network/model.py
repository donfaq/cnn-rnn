import config
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

FLAGS = tf.app.flags.FLAGS


class Model:
    def __init__(self, inputs, is_training):
        self.inputs = inputs
        self.is_training = is_training
        self.logits = self._init_model()

    def _init_model(self):
        if FLAGS.conv == 'inception':
            print('Using Inception model')
            net = self._inception_cnn(self.inputs)
        elif FLAGS.conv == 'vgg16':
            print('Using VGG16 model')
            net = self._vgg16(self.inputs)
        else:
            print('Using common cnn block')
            net = self._cnn(self.inputs)
        rnn = self._rnn_cell(net)
        return self._dense(rnn)

    def _cnn(self, input):
        with slim.arg_scope([slim.conv2d], stride=1,
                            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                            trainable=self.is_training):
            with tf.variable_scope('Convolution', [input]):
                conv1 = slim.conv2d(input, 32, [1, 1], stride=2, scope='Conv1',
                                    normalizer_fn=slim.batch_norm,
                                    normalizer_params={'is_training': self.is_training})
                pool2 = slim.max_pool2d(conv1, [3, 3], scope='Pool1', stride=1)
                conv2 = slim.conv2d(pool2, 32, [3, 3], scope='Conv2')
                pool3 = slim.max_pool2d(conv2, [3, 3], scope='Pool2', stride=1)
                return slim.conv2d(pool3, 32, [3, 3], stride=2, scope='Conv3')

    def _inception_cnn(self, inputs):
        conv1 = slim.conv2d(inputs, 32, [3, 3], stride=2, padding='VALID', scope='Conv2d_1a_3x3')
        conv2 = slim.conv2d(conv1, 32, [3, 3], stride=2, padding='VALID', scope='Conv2d_2a_3x3')
        inc_inputs = slim.conv2d(conv2, 64, [3, 3], scope='Conv2d_2b_3x3')

        with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d],
                            trainable=self.is_training,
                            stride=1, padding='SAME'):
            with tf.variable_scope('BlockInceptionA', [inc_inputs]):
                with tf.variable_scope('IBranch_0'):
                    ibranch_0 = slim.conv2d(inc_inputs, 96, [1, 1], scope='IConv2d_0a_1x1')
                with tf.variable_scope('IBranch_1'):
                    ibranch_1_conv1 = slim.conv2d(inc_inputs, 64, [1, 1], scope='IConv2d_0a_1x1')
                    ibranch_1 = slim.conv2d(ibranch_1_conv1, 96, [3, 3], scope='IConv2d_0b_3x3')
                with tf.variable_scope('IBranch_2'):
                    ibranch_2_conv1 = slim.conv2d(inc_inputs, 64, [1, 1], scope='IConv2d_0a_1x1')
                    ibranch_2_conv2 = slim.conv2d(ibranch_2_conv1, 96, [3, 3], scope='IConv2d_0b_3x3')
                    ibranch_2 = slim.conv2d(ibranch_2_conv2, 96, [3, 3], scope='IConv2d_0c_3x3')
                with tf.variable_scope('IBranch_3'):
                    ibranch_3_pool = slim.avg_pool2d(inc_inputs, [3, 3], scope='IAvgPool_0a_3x3')
                    ibranch_3 = slim.conv2d(ibranch_3_pool, 96, [1, 1], scope='IConv2d_0b_1x1')
                inception = tf.concat(axis=3, values=[ibranch_0, ibranch_1, ibranch_2, ibranch_3])
            with tf.variable_scope('BlockReductionA', [inception]):
                with tf.variable_scope('RBranch_0'):
                    rbranch_0 = slim.conv2d(inception, 384, [3, 3], stride=2, padding='VALID',
                                            scope='RConv2d_1a_3x3')
                with tf.variable_scope('RBranch_1'):
                    rbranch_1_conv1 = slim.conv2d(inception, 192, [1, 1], scope='RConv2d_0a_1x1')
                    rbranch_1_conv2 = slim.conv2d(rbranch_1_conv1, 224, [3, 3], scope='RConv2d_0b_3x3')
                    rbranch_1 = slim.conv2d(rbranch_1_conv2, 256, [3, 3], stride=2, padding='VALID',
                                            scope='RConv2d_1a_3x3')
                with tf.variable_scope('RBranch_2'):
                    rbranch_2 = slim.max_pool2d(inception, [3, 3], stride=2, padding='VALID',
                                                scope='RMaxPool_1a_3x3')
            return tf.concat(axis=3, values=[rbranch_0, rbranch_1, rbranch_2])

    def _vgg16(self, inputs):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            trainable=self.is_training,
                            weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                            weights_regularizer=slim.l2_regularizer(0.0005)):
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [2, 2], scope='pool5')
            net = slim.fully_connected(net, 4096, scope='fc6')
            net = slim.dropout(net, 0.5, scope='dropout6')
            net = slim.fully_connected(net, 4096, scope='fc7')
            net = slim.dropout(net, 0.5, scope='dropout7')
            net = slim.fully_connected(net, 1000, activation_fn=None, scope='fc8')
        return net

    @staticmethod
    def _rnn_cell(net):
        with tf.variable_scope('RNN_cell'):
            size = np.prod(net.get_shape().as_list()[1:])
            rnn_inputs = tf.reshape(net, (-1, FLAGS.esize, size))
            if FLAGS.rnn == 'LSTM':
                cell = tf.contrib.rnn.LSTMCell(100)
            else:
                cell = tf.contrib.rnn.GRUCell(100)
            init_state = cell.zero_state(1, dtype=tf.float32)
            rnn_outputs, _ = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)
            return tf.reduce_mean(rnn_outputs, axis=1)

    @staticmethod
    def _dense(output):
        with tf.name_scope('Dense'):
            return slim.fully_connected(output, 6, scope="dense")
