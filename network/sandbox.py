import tensorflow as tf
import logging as log
from dataset.tfrecord_reader import read

log.basicConfig(level=log.INFO)

NUM_THREADS = 1

queue = tf.FIFOQueue(capacity=1000, dtypes=[tf.int64, tf.uint8])
filenames = tf.gfile.Glob('dataset/SDHA2010Interaction/segmented_set1/0_1_4.avi_part*.tfrecord')
filename_queue = tf.train.string_input_producer(filenames, shuffle=False)

enqueue_op = queue.enqueue(read(filename_queue))

dequeue_op = queue.dequeue()
#
qr = tf.train.QueueRunner(queue, [enqueue_op] * NUM_THREADS)
#
if __name__ == '__main__':
    from PIL import Image

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        enqueue_threads = qr.create_threads(sess, coord=coord, start=True)
        stuff = {'labels': [], 'videos': []}
        for step in range(2):  # do to 100 iterations
            if coord.should_stop():
                break
            label_video = sess.run(dequeue_op)
            stuff['labels'].append(label_video[0])
            stuff['videos'].append(label_video[1])
        # print(len(stuff[0][1]))
        count = 0
        for video in stuff['videos']:
            Image.fromarray(video[0], 'RGB').save('vid_{}.jpg'.format(count))
            count += 1
        # Image.fromarray(stuff['videos'][1][0], 'RGB').save('stuff.jpg')
        # Image.fromarray(stuff[1][0], 'rgb').show()
        count = 0

        coord.request_stop()
        coord.join(enqueue_threads)
