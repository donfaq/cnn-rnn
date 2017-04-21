import tensorflow as tf

from dataset.dataset import read_tfrecord, BATCH_SIZE, HEIGHT, WIDTH

x = tf.placeholder(dtype=tf.float32, shape=(BATCH_SIZE, HEIGHT, WIDTH, 3))
y = tf.placeholder(dtype=tf.int32, shape=1)

with tf.Session() as sess:
    for video in tf.gfile.Glob("dataset/SDHA2010Interaction/segmented_set2/*.tfr"):
        label, batches = read_tfrecord(video)
        for batch in batches:
            sess.run(tf.global_variables_initializer())
            feed_dict = {x: batch, y: label}
            x_r, y_r = sess.run([x, y], feed_dict=feed_dict)
            print(x_r.shape)
            print(y_r)