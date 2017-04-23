import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from dataset.dataset import read_tfrecord, batch_size, height, width

restore = True
is_training = True
logs_path = 'network/logs'
chkpt_file = logs_path + "/model.ckpt"
learning_rate = 1e-4
group_size = 6
classes_num = 6

with tf.Graph().as_default() as graph:
    x = tf.placeholder(dtype=tf.float32, shape=(None, batch_size, height, width, 3), name="input")  # (2, 240, 320, 3)
    y = tf.placeholder(dtype=tf.int32, shape=(None, 1), name='labels')
    flatten_x = tf.reshape(x, (-1, height, width, 3))

    with slim.arg_scope([slim.conv2d], stride=1):
        with tf.variable_scope('Convolution', [flatten_x]):
            conv1 = slim.conv2d(flatten_x, 32, [1, 1], stride=2, scope='Conv1')
            dropout = slim.dropout(conv1, 0.8, is_training=is_training)
            pool2 = slim.max_pool2d(dropout, [3, 3], scope='Pool1', stride=1)
            conv2 = slim.conv2d(pool2, 32, [3, 3], scope='Conv2')
            dropout = slim.dropout(conv2, 0.8, is_training=is_training)
            pool3 = slim.max_pool2d(dropout, [3, 3], scope='Pool2', stride=1)
            conv3 = slim.conv2d(pool3, 32, [3, 3], stride=2, scope='Conv3')
            net = slim.dropout(conv3, 0.7, is_training=is_training, scope='Pool3')

    size = np.prod(net.get_shape().as_list()[1:])  # 307200

    with tf.variable_scope('GRU_RNN_cell', [flatten_x]):
        rnn_inputs = tf.reshape(net, (-1, batch_size, size))
        cell = tf.contrib.rnn.GRUCell(100)
        init_state = cell.zero_state(tf.shape(x)[0], dtype=tf.float32)
        rnn_outputs, _ = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)
        output = tf.reduce_mean(rnn_outputs, axis=1)

    # full-connected
    with tf.name_scope('Dense'):
        logits = slim.fully_connected(output,
                                      classes_num,
                                      weights_initializer=tf.truncated_normal_initializer(mean=0.0,
                                                                                          stddev=0.05,
                                                                                          dtype=tf.float32),
                                      scope="Fully-connected")
    with tf.name_scope('Accuracy'):
        prediction = tf.cast(tf.arg_max(logits, dimension=1), tf.int32)
        equality = tf.equal(prediction, y)
        accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
        tf.summary.scalar("accuracy", accuracy)
    with tf.name_scope('Cost'):
        cross_entropy = slim.losses.sparse_softmax_cross_entropy(logits=logits, labels=y, scope='cross_entropy')
        tf.summary.scalar("cost", cross_entropy)
    with tf.name_scope('Optimizer'):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss=cross_entropy,
                                                                               global_step=global_step)
    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver()

if __name__ == '__main__':

    print("X: ", x)
    print("FlattenX: ", flatten_x)
    print("conv1: ", conv1)
    print("dropout: ", dropout)
    print("conv2: ", conv2)
    print("pool2: ", pool2)
    print("Conv1: ", net)
    print("RNN inputs ", rnn_inputs)
    print("RNN outputs: ", rnn_outputs)
    print("Output: ", output)
    print("Logits: ", logits)

    with tf.Session(graph=graph) as sess:
        if restore:
            saver.restore(sess, chkpt_file)
            print("Model restored.")

        # sess.run(slim.variables.global_variables_initializer())
        writer = tf.summary.FileWriter(logs_path, graph=graph)

        while True:

            for video in tf.gfile.Glob("dataset/SDHA2010Interaction/segmented_set1/?_1_?.tfr"):
                label, video_frames = read_tfrecord(video)
                for group in np.array_split(video_frames, group_size):
                    label_arr = np.array([label] * len(group))

                    feed_dict = {x: group, y: label_arr}
                    _, summary, acc, gs = sess.run([train_step, summary_op, accuracy, global_step], feed_dict=feed_dict)
                    writer.add_summary(summary, gs)
                    print("Num epochs {} - Accuracy: {}".format(gs, acc))
                    if gs % 10:
                        save_path = saver.save(sess, chkpt_file)
                        print("Model saved in file: %s" % save_path)
