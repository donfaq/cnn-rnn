import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from dataset import Reader


class Network:
    def __init__(self, args, pattern="dataset/data/segmented_set1/*.tfr"):
        self.BATCH_SIZE = args.size
        self.STEP = args.step
        self.HEIGHT = args.height
        self.WIDTH = args.width
        self.LRATE = args.lrate
        self.RESTORE = args.restore
        self.UPDATE = args.update
        self.reader = Reader.Reader(args, pattern)
        self.logs_path = 'network/logs'
        self.chkpt_file = self.logs_path + "/model.ckpt"
        self.classes_num = 6
        self.model()
        self.writer = tf.summary.FileWriter(self.logs_path, graph=self.graph)

    def model(self):
        with tf.Graph().as_default() as self.graph:
            self.x = tf.placeholder(dtype=tf.float32,
                                    shape=(self.BATCH_SIZE, self.HEIGHT, self.WIDTH, 3),
                                    name="input")
            self.y = tf.placeholder(dtype=tf.int32, shape=(1,), name='labels')
            # flatten_x = tf.reshape(x, (-1, height, width, 3))
            net = self.cnn(self.x)
            size = np.prod(net.get_shape().as_list()[1:])
            output = self.rnn(net, size)
            logits = self.dense(output)
            self.calc_accuracy(logits, self.y)

            with tf.name_scope('Cost'):
                cross_entropy = slim.losses.sparse_softmax_cross_entropy(logits=logits, labels=self.y, scope='cross_entropy')
                tf.summary.scalar("cross_entropy", cross_entropy)

            with tf.name_scope('Optimizer'):
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.train_step = tf.train.GradientDescentOptimizer(self.LRATE).minimize(loss=cross_entropy,
                                                                                         global_step=self.global_step)
            self.summary_op = tf.summary.merge_all()
            self.saver = tf.train.Saver()

    def cnn(self, input):
        with slim.arg_scope([slim.conv2d], stride=1):
            with tf.variable_scope('Convolution', [input]):
                conv1 = slim.conv2d(input, 32, [1, 1], stride=2, scope='Conv1')
                # dropout = slim.dropout(conv1, 0.8, is_training=is_training)
                pool2 = slim.max_pool2d(conv1, [3, 3], scope='Pool1', stride=1)
                conv2 = slim.conv2d(pool2, 32, [3, 3], scope='Conv2')
                # dropout = slim.dropout(conv2, 0.8, is_training=is_training)
                pool3 = slim.max_pool2d(conv2, [3, 3], scope='Pool2', stride=1)
                return slim.conv2d(pool3, 32, [3, 3], stride=2, scope='Conv3')
                # net = slim.dropout(conv3, 0.7, is_training=is_training, scope='Pool3')

    def rnn(self, net, size):
        with tf.variable_scope('GRU_RNN_cell'):
            rnn_inputs = tf.reshape(net, (-1, self.BATCH_SIZE, size))
            cell = tf.contrib.rnn.GRUCell(100)
            init_state = cell.zero_state(1, dtype=tf.float32)
            rnn_outputs, _ = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)
            return tf.reduce_mean(rnn_outputs, axis=1)


    def dense(self, output):
        with tf.name_scope('Dense'):
            return slim.fully_connected(output,
                                        self.classes_num,
                                        weights_initializer=tf.truncated_normal_initializer(mean=0.0,
                                                                                            stddev=0.05,
                                                                                            dtype=tf.float32),
                                        scope="Fully-connected")

    def calc_accuracy(self, logits, y):
        with tf.name_scope('Accuracy'):
            prediction = tf.cast(tf.arg_max(logits, dimension=1), tf.int32)
            equality = tf.equal(prediction, y)
            self.accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
            tf.summary.scalar("accuracy", self.accuracy)

    def begin_training(self):
        with tf.Session(graph=self.graph) as sess:
            if self.RESTORE:
                self.saver.restore(sess, self.chkpt_file)
                print("Model restored.")
            else:
                sess.run(tf.global_variables_initializer())

            while True:
                label, example = self.reader.get_random_example()
                feed_dict = {self.x: example, self.y: label}
                _, summary, acc, gs = sess.run([self.train_step, self.summary_op, self.accuracy, self.global_step],
                                               feed_dict=feed_dict)
                self.writer.add_summary(summary, gs)
                print("Global step {} - Accuracy: {}".format(gs, acc))
                if gs % 100 == 0:
                    save_path = self.saver.save(sess, self.chkpt_file)
                    print("Model saved in file: %s" % save_path)
