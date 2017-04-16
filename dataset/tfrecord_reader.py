import cv2
import tensorflow as tf
from PIL import Image

context_features = {
    "length": tf.FixedLenFeature([], dtype=tf.int64),
    "label": tf.FixedLenFeature([], dtype=tf.int64),
    "extention": tf.FixedLenFeature([], dtype=tf.string)
}

sequence_features = {
    "frame": tf.FixedLenSequenceFeature([], dtype=tf.string),
    "group": tf.FixedLenSequenceFeature([], dtype=tf.int64),
    "number_in_group": tf.FixedLenSequenceFeature([], dtype=tf.int64),
}


def read(filename):
    context_data, sequence_data = [], []
    for serialized_example in tf.python_io.tf_record_iterator(filename):
        context_parsed, sequence_parsed = tf.parse_single_sequence_example(
            serialized=serialized_example,
            context_features=context_features,
            sequence_features=sequence_features
        )
        context_data.append(context_parsed)
        sequence_data.append(sequence_parsed)

    return context_data, sequence_data


def read_binary_video(filename):
    context, sequence = read(filename)
    label = context[0]['label']
    byte_videoframes = []
    for element in sequence:
        byte_videoframes.append(element['frame'])
    with tf.Session() as sess:
        byte_videoframes = sess.run(byte_videoframes)
    BGR_videoframes = []
    for frame_group in byte_videoframes:
        for frame in frame_group:
            BGR_videoframes.append(tf.image.decode_jpeg(frame, channels=3))
    return label, BGR_videoframes


def BGRtoRGB(bgr_frames):
    RGB_frames = []
    for frame in bgr_frames:
        RGB_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return RGB_frames

if __name__ == '__main__':
    with tf.Session() as session:
        vlabel, frames = session.run(
            read_binary_video('SDHA2010Interaction/segmented_set2/15_13_3.avi.tfrecord'))

    count = 0
    for img in frames:
        Image.fromarray(BGRtoRGB(img), 'RGB').save('tmp/tmp_{}.jpg'.format(count))
        count += 1
