import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from dataset.dataset import read_tfrecord, batch_size, height, width

restore = False
logdir = 'network/train_logs'
chkpt_dir = 'network/chkpt'
learning_rate = 0.001

with tf.Graph().as_default():
    x = tf.placeholder(dtype=tf.float32, shape=(batch_size, height, width, 3))  # (2, 240, 320, 3)
    y = tf.placeholder(dtype=tf.int32, shape=1)

    with slim.arg_scope([slim.conv2d], stride=1):
        with tf.variable_scope('BlockA', [x]):
            conv1 = slim.conv2d(x, 32, [1, 1], stride=2, scope='conv1')
            conv2 = slim.conv2d(conv1, 32, [3, 3], scope='Conv2')
            pool = slim.max_pool2d(conv2, [3, 3], scope='Pool', stride=1)
            conv3 = slim.conv2d(pool, 32, [3, 3], stride=2, scope='Conv3')

    size = np.prod(conv3.get_shape().as_list()[1:])  # 307200

    # rnn cell
    rnn_inputs = tf.reshape(conv3, (-1, batch_size, size))
    cell = tf.contrib.rnn.GRUCell(100)
    init_state = cell.zero_state(1, dtype=tf.float32)
    rnn_outputs, _ = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)
    output = tf.reduce_mean(rnn_outputs, axis=1)

    # full-connected
    dense_w = tf.Variable(tf.truncated_normal_initializer(mean=0.0, stddev=0.05, dtype=tf.float32)(shape=(100, 6)),
                          tf.float32, name="w")
    dense_b = tf.Variable(tf.zeros([6]), tf.float32, name="b")
    score = tf.add(tf.matmul(output, dense_w), dense_b, name="out")

    prediction = tf.cast(tf.arg_max(score, dimension=1), tf.int32)
    loss = slim.losses.sparse_softmax_cross_entropy(logits=score, labels=y)

    total_loss = slim.losses.get_total_loss()

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

    global_step = tf.Variable(0, name='global_step', trainable=False)  # learning rate

    train_op = slim.learning.create_train_op(total_loss, optimizer)
    train = slim.learning.train(train_op,
                                logdir,
                                number_of_steps=1000,
                                save_summaries_secs=300,
                                save_interval_secs=600)


if __name__ == '__main__':
    print("Net0: ", conv1)
    print("Net: ", conv2)
    print("Pool: ", pool)
    print("Conv1: ", conv3)
    print("RNN inputs ", rnn_inputs)
    print("RNN outputs: ", rnn_outputs)
    print("Output: ", output)
    with tf.Session() as sess:

        # if restore:
        #     restorer.restore(sess, chkpt_dir + "/data.chkp")
        #     print("model restored")

        for video in tf.gfile.Glob("dataset/SDHA2010Interaction/segmented_set2/?_11_?.tfr"):
            label, batches = read_tfrecord(video)
            for batch in batches:
                feed_dict = {x: batch, y: label}
                sess.run(slim.variables.global_variables_initializer())
                op = sess.run([train])
                # _, l = sess.run([train, loss], feed_dict=feed_dict)

                print(op)
                # saver.save(sess, chkpt_dir + "/data.chkp")
