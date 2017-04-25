import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image


class Dataset:
    def __init__(self, args):
        self.zip_options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
        self.BATCH_SIZE = args.size
        self.STEP = args.step
        self.HEIGHT = args.height
        self.WIDTH = args.width

    @staticmethod
    def from_BRG_to_RGB(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @staticmethod
    def resize_image(img, size):
        return np.array(Image.fromarray(img).resize((size[0], size[1])))

    def split_video_into_frames(self, path):
        vidcap = cv2.VideoCapture(path)
        success, image = vidcap.read()
        images = []
        assert success is True
        while success:
            images.append(self.resize_image(self.from_BRG_to_RGB(image), [self.WIDTH, self.HEIGHT]))
            success, image = vidcap.read()
        return np.array(images)

    @staticmethod
    def group_images(x, frame_count):
        images = []
        for shift in range(x.shape[0] - frame_count + 1):
            data = x[shift:shift + frame_count]
            images.append(data)
        return np.array(images)

    @staticmethod
    def group(arr, length, step):
        beg = 0
        end = length
        res = []
        while end < len(arr):
            res.append(arr[beg:end:])
            end += step
            beg += step
        return np.array(res)

    @staticmethod
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def write_tfrecord(self, filename, data):
        writer = tf.python_io.TFRecordWriter(filename, options=self.zip_options)
        for batch in data:
            example = tf.train.Example(features=tf.train.Features(feature={
                'batch': self._bytes_feature(batch.tostring())
            }))
            writer.write(example.SerializeToString())
        writer.close()

    @staticmethod
    def save_frames(batches, foldername="tmp/"):
        num_batch = 0
        for batch in batches:
            num_img = 0
            for image in batch:
                Image.fromarray(image).save("{}img_{}_{}.jpg".format(foldername, num_batch, num_img))
                num_img += 1
            num_batch += 1

    def create_dataset(self, pattern):
        for video in tf.gfile.Glob(pattern):
            print("Processing video: {}".format(video))
            filename, _ = os.path.splitext(video)
            self.write_tfrecord(filename + ".tfr",
                                self.group(self.split_video_into_frames(video), self.BATCH_SIZE, self.STEP))

