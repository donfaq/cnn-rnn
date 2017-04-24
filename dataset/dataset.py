import tensorflow as tf
import cv2
import numpy as np
import os

from PIL import Image

height = 240
width = 320
batch_size = 30
step = 15

zip_options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)


def from_BRG_to_RGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def resize_image(img, size):
    return np.array(Image.fromarray(img).resize((size[0], size[1])))


def split_video_into_frames(path):
    vidcap = cv2.VideoCapture(path)
    success, image = vidcap.read()
    images = []
    assert success is True
    while success:
        images.append(resize_image(from_BRG_to_RGB(image), [width, height]))
        success, image = vidcap.read()
    return np.array(images)


def group_images(x, frame_count):
    images = []
    for shift in range(x.shape[0] - frame_count + 1):
        data = x[shift:shift + frame_count]
        images.append(data)
    return np.array(images)


def group(arr, length, step):
    beg = 0
    end = length
    res = []
    while end < len(arr):
        res.append(arr[beg:end:] )
        end += step
        beg += step
    return np.array(res)


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def write_tfrecord(filename, data):
    writer = tf.python_io.TFRecordWriter(filename, options=zip_options)
    for batch in data:
        example = tf.train.Example(features=tf.train.Features(feature={
            'batch': _bytes_feature(batch.tostring())
        }))
        writer.write(example.SerializeToString())
    writer.close()


def save_frames(batches, foldername="tmp/"):
    num_batch = 0
    for batch in batches:
        num_img = 0
        for image in batch:
            Image.fromarray(image).save("{}img_{}_{}.jpg".format(foldername, num_batch, num_img))
            num_img += 1
        num_batch += 1


def create_dataset(pattern):
    for video in tf.gfile.Glob(pattern):
        print("Processing video: {}".format(video))
        filename, _ = os.path.splitext(video)
        write_tfrecord(filename + ".tfr", group(split_video_into_frames(video), batch_size, step))


if __name__ == '__main__':
    create_dataset("SDHA2010Interaction/segmented_set?/*.avi")
