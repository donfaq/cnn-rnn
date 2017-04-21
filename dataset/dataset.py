import tensorflow as tf
import cv2
import numpy as np
import os

from PIL import Image

HEIGHT = 240
WIDTH = 320
BATCH_SIZE = 2

video_path = 'SDHA2010Interaction/segmented_set1/20_4_2.avi'
zip_options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)


def from_BRG_to_RGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def resize_image(img, size):
    return np.array(Image.fromarray(img).resize((size[0], size[1])))


def normalize_images(data) -> np.ndarray:
    return data / 255


def split_video_into_frames(path):
    vidcap = cv2.VideoCapture(path)
    success, image = vidcap.read()
    images = []
    assert success is True
    while success:
        images.append(resize_image(from_BRG_to_RGB(image), [WIDTH, HEIGHT]))
        success, image = vidcap.read()
    return np.array(images)


def group_images(x, frame_count):
    images = []
    for shift in range(x.shape[0] - frame_count + 1):
        data = x[shift:shift + frame_count]
        images.append(data)
    return np.array(images)


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


def parse_tfr_filename(path):
    filename, ext = os.path.splitext(path)
    path, file = os.path.split(filename)
    return "{}/{}.tfr".format(path, file), np.array([int(file.split('_')[-1])])


def read_tfrecord(path):
    batches = []
    filename, label = parse_tfr_filename(path)
    for string_record in tf.python_io.tf_record_iterator(path=filename, options=zip_options):
        example = tf.train.Example()
        example.ParseFromString(string_record)
        img_string = (example.features.feature['batch'].bytes_list.value[0])
        data = np.fromstring(img_string, dtype=np.uint8).reshape((BATCH_SIZE, HEIGHT, WIDTH, -1))
        batches.append(data)
    return label, np.array(batches)


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
        write_tfrecord(filename + ".tfr", group_images(split_video_into_frames(video), BATCH_SIZE))


def read_dataset(pattern):
    labels, videos = [], []
    for video in tf.gfile.Glob(pattern):
        print("Reading video: {}".format(video))
        label, batches = read_tfrecord(video)
        labels.append(label)
        videos.append(normalize_images(batches))
    return np.array(labels), np.array(videos)



