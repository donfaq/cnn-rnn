import tensorflow as tf
import numpy as np
from PIL import Image

from dataset.tfrecord_reader import read_binary_video

N_SAMPLES = 1000
NUM_THREADS = 4


queue = tf.FIFOQueue(capacity=10000, dtypes=[tf.int64, tf.uint8])
for file in tf.gfile.Glob('dataset/SDHA2010Interaction/segmented_set2/*.avi.tfrecord'):
    label, frame = read_binary_video(file)
    enqueue_op = queue.enqueue([label, tf.reverse(frame, axis=[-1])])

dequeue_op = queue.dequeue()
#
qr = tf.train.QueueRunner(queue, [enqueue_op] * NUM_THREADS)
#
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    enqueue_threads = qr.create_threads(sess, coord=coord, start=True)
    for step in range(5):  # do to 100 iterations
        if coord.should_stop():
            break
        label, frame = sess.run(dequeue_op)
    count = 0
    for img in frame:
        Image.fromarray(img, 'RGB').save('tmp/tmp_{}.jpg'.format(count))
        count += 1
    coord.request_stop()
    coord.join(enqueue_threads)
