import os

import numpy as np
import tensorflow as tf


class Reader:
    def __init__(self, args, data_pattern):
        self.BATCH_SIZE = args.size
        self.HEIGHT = args.height
        self.WIDTH = args.width
        self.zip_options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
        self.data_pattern = data_pattern
        self.files = np.array(tf.gfile.Glob(self.data_pattern))
        self.init_dataset()

    def get_next_example_group(self):
        label, video = self.read_tfrecord(str(self.iterator.value))
        self.iterator.iternext()
        return self.iterator.finished, label, video

    def get_random_example_group(self, length):
        r_labels, r_examples = [], []
        for i in range(length):
            label, video = self.read_tfrecord(np.random.choice(self.files))
            r_labels.append(label)
            r_examples.append(video[np.random.choice(video.shape[0], 1, replace=False)][0])
        return np.array(r_labels).reshape(length, 1), np.array(r_examples)

    def get_random_example(self):
        label, video = self.read_tfrecord(np.random.choice(self.files))
        return np.array(label), video[np.random.choice(video.shape[0], 1, replace=False)][0]

    def read_tfrecord(self, path):
        batches = []
        filename, label = self.parse_tfr_filename(path)
        for string_record in tf.python_io.tf_record_iterator(path=filename, options=self.zip_options):
            example = tf.train.Example()
            example.ParseFromString(string_record)
            img_string = (example.features.feature['batch'].bytes_list.value[0])
            data = np.fromstring(img_string, dtype=np.uint8).reshape((self.BATCH_SIZE, self.HEIGHT, self.WIDTH, -1))
            batches.append(data)
        return label, np.array(batches)

    def parse_tfr_filename(self, path):
        filename, ext = os.path.splitext(path)
        path, file = os.path.split(filename)
        return "{}/{}.tfr".format(path, file), np.array([int(file.split('_')[-1])])

    def normalize_images(self, data) -> np.ndarray:
        return data / 255

    def init_dataset(self):
        np.random.shuffle(self.files)
        self.iterator = np.nditer(self.files)
