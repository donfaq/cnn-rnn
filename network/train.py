import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from dataset import Reader


class Network:
    def __init__(self, args):
        self.BATCH_SIZE = args.size
        self.train_pattern = "dataset/data/segmented_set1/*.tfr"
        self.test_pattern = "dataset/data/segmented_set2/*.tfr"
        self.train_reader = Reader.Reader(args, self.train_pattern)
        self.test_reader = Reader.Reader(args, self.test_pattern)
        self.STEP = args.step
        self.IS_TRAINING = True
        self.HEIGHT = args.height
        self.WIDTH = args.width
        self.LRATE = args.lrate
        self.RESTORE = args.restore
        self.UPDATE = args.update
        self.cnntype = args.cnn
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
            if self.cnntype == 'inception':
                print('Using inception cnn block')
                net = self.inception_cnn(self.x)
            else:
                print('Using common cnn block')
                net = self.cnn(self.x)
            size = np.prod(net.get_shape().as_list()[1:])
            output = self.rnn(net, size)
            logits = self.dense(output)
            self.calc_accuracy(logits, self.y)

            with tf.name_scope('Cost'):
                self.cross_entropy = slim.losses.sparse_softmax_cross_entropy(logits=logits, labels=self.y,
                                                                              scope='cross_entropy')
                tf.summary.scalar("cross_entropy", self.cross_entropy)

            with tf.name_scope('Optimizer'):
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer(self.LRATE)
                self.train_step = slim.learning.create_train_op(self.cross_entropy, self.optimizer, self.global_step)
            self.summary_op = tf.summary.merge_all()
            self.saver = tf.train.Saver()

    def cnn(self, input):
        with slim.arg_scope([slim.conv2d], stride=1, weights_initializer=tf.contrib.layers.xavier_initializer_conv2d()):
            with tf.variable_scope('Convolution', [input]):
                conv1 = slim.conv2d(input, 32, [1, 1], stride=2, scope='Conv1',
                                    normalizer_fn=slim.batch_norm,
                                    normalizer_params={'is_training': self.IS_TRAINING})
                # dropout = slim.dropout(conv1, 0.8, is_training=is_training)
                pool2 = slim.max_pool2d(conv1, [3, 3], scope='Pool1', stride=1)
                conv2 = slim.conv2d(pool2, 32, [3, 3], scope='Conv2')
                # dropout = slim.dropout(conv2, 0.8, is_training=is_training)
                pool3 = slim.max_pool2d(conv2, [3, 3], scope='Pool2', stride=1)
                return slim.conv2d(pool3, 32, [3, 3], stride=2, scope='Conv3')
                # net = slim.dropout(conv3, 0.7, is_training=is_training, scope='Pool3')

    def inception_cnn(self, inputs):
        conv1 = slim.conv2d(inputs, 32, [3, 3], stride=2, padding='VALID', scope='Conv2d_1a_3x3')
        conv2 = slim.conv2d(conv1, 32, [3, 3], stride=2, padding='VALID', scope='Conv2d_2a_3x3')
        inc_inputs = slim.conv2d(conv2, 64, [3, 3], scope='Conv2d_2b_3x3')

        with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d], stride=1, padding='SAME'):
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
                rbranch_0 = slim.conv2d(inception, 384, [3, 3], stride=2, padding='VALID', scope='RConv2d_1a_3x3')
            with tf.variable_scope('RBranch_1'):
                rbranch_1_conv1 = slim.conv2d(inception, 192, [1, 1], scope='RConv2d_0a_1x1')
                rbranch_1_conv2 = slim.conv2d(rbranch_1_conv1, 224, [3, 3], scope='RConv2d_0b_3x3')
                rbranch_1 = slim.conv2d(rbranch_1_conv2, 256, [3, 3], stride=2, padding='VALID', scope='RConv2d_1a_3x3')
            with tf.variable_scope('RBranch_2'):
                rbranch_2 = slim.max_pool2d(inception, [3, 3], stride=2, padding='VALID', scope='RMaxPool_1a_3x3')
            return tf.concat(axis=3, values=[rbranch_0, rbranch_1, rbranch_2])

    def rnn(self, net, size):
        with tf.variable_scope('GRU_RNN_cell'):
            rnn_inputs = tf.reshape(net, (-1, self.BATCH_SIZE, size))
            cell = tf.contrib.rnn.GRUCell(100)
            init_state = cell.zero_state(1, dtype=tf.float32)
            rnn_outputs, _ = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)
            return tf.reduce_mean(rnn_outputs, axis=1)

    def dense(self, output):
        with tf.name_scope('Dense'):
            return slim.fully_connected(output, self.classes_num, scope="Fully-connected")

    def calc_accuracy(self, logits, y):
        with tf.name_scope('Accuracy'):
            prediction = tf.cast(tf.arg_max(logits, dimension=1), tf.int32)
            self.accuracy = tf.contrib.metrics.accuracy(labels=y, predictions=prediction)
            tf.summary.scalar("accuracy", self.accuracy)

    def begin_training(self):
        with tf.Session(graph=self.graph) as sess:
            if self.RESTORE:
                self.saver.restore(sess, self.chkpt_file)
                print("Model restored.")
            else:
                sess.run(tf.local_variables_initializer())
                sess.run(tf.global_variables_initializer())

            right_answers = []
            total = []
            epoch_count = 0
            while True:
                label, example = self.train_reader.get_random_example()
                feed_dict = {self.x: example, self.y: label}

                _, summary, acc, g_step = sess.run(
                    [self.train_step,
                     self.summary_op,
                     self.accuracy,
                     self.global_step], feed_dict=feed_dict)
                self.writer.add_summary(summary, g_step)

                print("[train] Global step {}: Current step accuracy = {}".format(g_step, acc))
                if acc == 1:
                    right_answers.append(1)
                if g_step % 10 == 0:
                    acc10 = len(right_answers) / 10
                    right_answers = []
                    print("10 steps accuracy = {}".format(acc10))
                    total.append(acc10)
                if g_step % 100 == 0:
                    save_path = self.saver.save(sess, self.chkpt_file)
                    print("Model saved in file: %s" % save_path)
                    epoch_count += 1
                    print("[EPOCH {}] TOTAL ACCURACY AFTER 100 STEPS: {}".format(epoch_count, sum(total) / 100))
                    self.begin_test()

    def begin_test(self):
        with tf.Session(graph=self.graph) as sess:
            self.saver.restore(sess, self.chkpt_file)
            print("Model restored")
            self.IS_TRAINING = False

            right_answers = []
            total = []

            for i in range(100):  # 100 steps == 1 epoch
                label, example = self.test_reader.get_random_example()
                feed_dict = {self.x: example, self.y: label}
                summary, acc, g_step = sess.run(
                    [self.summary_op,
                     self.accuracy,
                     self.global_step], feed_dict=feed_dict)
                if acc == 1:
                    right_answers.append(1)
                if g_step % 10 == 0:
                    acc10 = len(right_answers) / 10
                    right_answers = []
                    print("[test] 10 test steps accuracy = {}".format(acc10))
                    total.append(acc10)
                if g_step % 100 == 0:
                    print("TOTAL TEST ACCURACY: {}".format(sum(total) / 100))
