import config
import tensorflow as tf
import tensorflow.contrib.slim as slim
from network.model import Model

FLAGS = tf.app.flags.FLAGS


class Network:
    def __init__(self, is_training):
        self.graph = tf.Graph()
        self.is_training = is_training
        self.eval()

    def eval(self):
        with self.graph.as_default():
            self.x = tf.placeholder(dtype=tf.float32, shape=(FLAGS.esize, FLAGS.height, FLAGS.width, 3), name="inputs")
            self.y = tf.placeholder(dtype=tf.int32, shape=(1,), name='label')
            logits = Model(self.x, self.is_training).logits
            self._calc_accuracy(logits, self.y)

            with tf.name_scope('Cost'):
                cross_entropy = slim.losses.sparse_softmax_cross_entropy(logits=logits, labels=self.y,
                                                                         scope='cross_entropy')
                tf.summary.scalar("cross_entropy", cross_entropy)
            with tf.name_scope('Optimizer'):
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                # optimizer = tf.train.AdamOptimizer(FLAGS.lrate)
                # optimizer = tf.train.MomentumOptimizer(FLAGS.lrate, 0.9, use_nesterov=True)
                optimizer = tf.train.GradientDescentOptimizer(FLAGS.lrate)
                # optimizer = tf.train.RMSPropOptimizer(FLAGS.lrate)
                self.train_step = slim.learning.create_train_op(cross_entropy, optimizer, self.global_step,
                                                                aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
            self.summary_op = tf.summary.merge_all()
            self.saver = tf.train.Saver()

    def _calc_accuracy(self, logits, y):
        with tf.name_scope('Accuracy'):
            prediction = tf.cast(tf.arg_max(logits, dimension=1), tf.int32)
            self.accuracy = tf.contrib.metrics.accuracy(labels=y, predictions=prediction)
            tf.summary.scalar("accuracy", self.accuracy)


    @staticmethod
    def print_model():
        def get_nb_params_shape(shape):
            nb_params = 1
            for dim in shape:
                nb_params = nb_params * int(dim)
            return nb_params

        tot_nb_params = 0
        for trainable_variable in slim.get_trainable_variables():
            print(trainable_variable.name, trainable_variable.shape)
            vshape = trainable_variable.get_shape()  # e.g [D,F] or [W,H,C]
            current_nb_params = get_nb_params_shape(vshape)
            tot_nb_params = tot_nb_params + current_nb_params
        print('Total number of trainable params', tot_nb_params)
