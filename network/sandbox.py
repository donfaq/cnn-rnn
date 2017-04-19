import tensorflow as tf
import logging as log
from dataset.tfrecord_reader import read_binary_video

log.basicConfig(level=log.INFO)

N_SAMPLES = 1000
NUM_THREADS = 1

queue = tf.FIFOQueue(capacity=1000, dtypes=[tf.int64, tf.uint8])
log.info('Start queue')
for file in tf.gfile.Glob('dataset/SDHA2010Interaction/segmented_set1/*_1_*.avi.tfrecord'):
    global enqueue_op
    log.info('Reading file {}'.format(file))
    # label, frames = read_binary_video(file)
    # enqueue_op = queue.enqueue([label, tf.reverse(frames, axis=[-1])])
    enqueue_op = queue.enqueue(read_binary_video(file))

dequeue_op = queue.dequeue()
#
qr = tf.train.QueueRunner(queue, [enqueue_op] * NUM_THREADS)
#
if __name__ == '__main__':
    from PIL import Image
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        enqueue_threads = qr.create_threads(sess, coord=coord, start=True)
        stuff = []
        for step in range(5):  # do to 100 iterations
            if coord.should_stop():
                break
            stuff.append(sess.run(dequeue_op))
        Image.fromarray(stuff[0][1][0], 'rgb').save('stuff.jpg')
        # Image.fromarray(stuff[1][0], 'rgb').show()
        count = 0

        coord.request_stop()
        coord.join(enqueue_threads)
