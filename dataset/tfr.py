import os
import io
import tensorflow as tf
import cv2
from PIL import Image


def video_to_frames(filename):
    vid = cv2.VideoCapture(filename)
    frames = []
    hasNext = True
    while hasNext:
        hasNext, frame = vid.read()
        if not hasNext:
            break
        frames.append(frame)
    assert len(frames) is not 0
    return filename, frames


def group_frames(filename, frames, batch=3, step=1):
    frames_group = 0
    start_group = 0
    end_group = batch
    frames_dict = {}
    while end_group <= len(frames):
        num = 0
        for frame in frames[start_group:end_group]:
            single_frame = {'vfile': filename,
                            'group': frames_group,
                            'num': num,
                            'frame': frame}
            frames_dict[str(frames_group) + '_' + str(num)] = single_frame
            num += 1
        start_group += step
        end_group += step
        frames_group += 1
    return frames_dict


def save_frames(frames_dict):
    for frame in frames_dict:
        cv2.imwrite("{vfile}_frame_{group}_{num}.jpg".format(
            vfile=frames_dict[frame]['vfile'],
            group=frames_dict[frame]['group'],
            num=frames_dict[frame]['num']
        ), frames_dict[frame]['frame'])
        print('done')
    return True


def sample_videofile(vfilename):
    dirname = vfilename + '_frames'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    filename, frames = video_to_frames(vfilename)
    os.chdir(dirname)
    save_frames(group_frames(filename, frames, batch=3, step=1))
    os.chdir("..")


def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_example(filename, group, image_encoded, label, height, width):
    colorspace = b'RGB'
    channels = 3
    image_format = b'JPEG'
    filename = bytes(filename, 'utf-8')
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'filename': _bytes_feature(filename),
                'group': _int64_feature(group),
                'height': _int64_feature(height),
                'width': _int64_feature(width),
                'colorspace': _bytes_feature(colorspace),
                'channels': _int64_feature(channels),
                'format': _bytes_feature(image_format),
                'label': _int64_feature(label),
                'image': _bytes_feature(image_encoded),
            }
        )
    )
    return example


def decode_jpeg(image_data):
    _decode_jpeg_data = tf.placeholder(dtype=tf.string)
    _decoded_jpeg = tf.image.decode_jpeg(_decode_jpeg_data, channels=3)
    return tf.Session().run(_decoded_jpeg, feed_dict={_decode_jpeg_data: image_data})


def process_image(filename):
    with tf.gfile.FastGFile(filename, 'r') as f:
        image_data = f.read()

    image = decode_jpeg(image_data)

    assert len(image.shape) == 3
    image_height = image.shape[0]
    image_width = image.shape[1]
    assert image.shape[2] == 3

    return image_data, image_height, image_width


def write_to_tfr(filename, output_filename):
    writer = tf.python_io.TFRecordWriter(output_filename)
    image_buffer, height, width = process_image(filename)
    example = convert_to_example(filename, 1, image_buffer, 1, height, width)
    writer.write(example.SerializeToString())
    writer.close()


def write_batch(path, video):
    output_path = '{video}.tfrecords'.format(video=video)
    writer = tf.python_io.TFRecordWriter(output_path)
    for file in tf.gfile.Glob(path):
        image_buffer, height, width = process_image(file)
        example = convert_to_example(file, 1, image_buffer, 1, height, width)
        writer.write(example.SerializeToString())
    writer.close()


def read_from_tfr(filename_queue, image_number):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'filename': tf.FixedLenFeature([], tf.string),
            'group': tf.FixedLenFeature([], tf.int64),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'colorspace': tf.FixedLenFeature([], tf.string),
            'channels': tf.FixedLenFeature([], tf.int64),
            'format': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
            'image': tf.FixedLenFeature([], tf.string),
        })
    sequence = tf.contrib.learn.run_n(features, n=image_number, feed_dict=None)
    return sequence


if __name__ == '__main__':
    video = '1.avi'
    sample_videofile(video)

    write_batch('{video}_frames/*.jpg'.format(video=video), video)
    queue = tf.train.string_input_producer(['11_2_3.tfrecords'])

    output = read_from_tfr(queue, 140)
    print(output[1]['height'], output[1]['width'])
    image_bytes = output[133]['image']
    image = Image.open(io.BytesIO(image_bytes))  # тут картинка с песиком
    image.show()
