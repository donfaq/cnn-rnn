import tensorflow as tf
import numpy as np

from dataset.dataset import read_tfrecord, batch_size, height, width

x = tf.placeholder(dtype=tf.float32, shape=(batch_size, height, width, 3))  # (2, 240, 320, 3)
y = tf.placeholder(dtype=tf.int32, shape=1)

with tf.name_scope("conv1"):
    conv1_w = tf.Variable(tf.truncated_normal_initializer(mean=0.0, stddev=0.05, dtype=tf.float32)(shape=(5, 5, 3, 16)),
                          tf.float32, name="w")
    conv1_b = tf.Variable(tf.zeros(shape=(16)), name="b")
    conv1 = tf.nn.relu(tf.nn.conv2d(x, conv1_w, strides=[1, 2, 2, 1], padding="SAME") + conv1_b,
                       name="conv")  # (2, 120, 160, 16)

size = np.prod(conv1.get_shape().as_list()[1:]) # 307200

rnn_inputs = tf.reshape(conv1, (-1, batch_size, size))
cell = tf.contrib.rnn.GRUCell(100)
init_state = cell.zero_state(1, dtype=tf.float32)

rnn_outputs, _ = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)
output = tf.reduce_mean(rnn_outputs, axis=1)

dense_w = tf.Variable(tf.truncated_normal_initializer(mean=0.0, stddev=0.05, dtype=tf.float32)(shape=(100, 6)),
                      tf.float32, name="w")
dense_b = tf.Variable(tf.zeros([6]), tf.float32, name="b")
score = tf.add(tf.matmul(output, dense_w), dense_b, name="out")

prediction = tf.cast(tf.arg_max(score, dimension=1), tf.int32)
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=score, labels=y)


with tf.Session() as sess:
    for video in tf.gfile.Glob("dataset/SDHA2010Interaction/segmented_set2/*.tfr"):
        label, batches = read_tfrecord(video)
        for batch in batches:
            sess.run(tf.global_variables_initializer())
            feed_dict = {x: batch, y: label}
            R_loss, R_prediction = sess.run([loss, prediction], feed_dict=feed_dict)
            print("Loss: ", R_loss)
            print("Prediction: ", R_prediction)
