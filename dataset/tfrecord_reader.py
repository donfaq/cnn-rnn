import base64

import tensorflow as tf
from dataset.Frames import Frames

context_features = {
    "label": tf.FixedLenFeature([], dtype=tf.int64)
}

sequence_features = {
    "frame": tf.FixedLenSequenceFeature([], dtype=tf.string)
}


def read(filename_queue):
    reader = tf.TFRecordReader()
    key, record_string = reader.read(filename_queue)
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=record_string,
        context_features=context_features,
        sequence_features=sequence_features
    )
    # context_data.append(context_parsed)
    # sequence_data.append(sequence_parsed)
    # return context_data[0]["label"], sequence_data[0]["frame"]
    return context_parsed, sequence_parsed


def base64_decode_op(x):
    return tf.py_func(lambda x: base64.decodestring(x), [x], [tf.string])[0]


# def read_binary_video(filename_queue):
#     context, sequence = read(filename_queue)
#     label = context
#     byte_videoframes = tf.Session().run(context)
#     return label, byte_videoframes


if __name__ == '__main__':
    filenames = tf.gfile.Glob('SDHA2010Interaction/segmented_set1/1_1_2.avi_part?.tfrecord')
    fqueue = tf.train.string_input_producer(filenames, shuffle=False, capacity=len(filenames))
    context, sequence = read(fqueue)

    sess = tf.Session()
    coord = tf.train.Coordinator()
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    threads = tf.train.start_queue_runners(sess=sess)
    # count = 0
    # for bimg in sess.run(sequence["frame"]):
    #     with open('test{}.jpg'.format(count), 'wb') as f_output:
    #         f_output.write(sess.run(base64_decode_op(bimg)))
    #         print('writing done')
    #     f_output.close()
    #     count += 1

    coord.request_stop(threads)
    sess.close()
