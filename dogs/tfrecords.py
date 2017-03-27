import numpy as np
import tensorflow as tf
import skimage.io as io
from PIL import Image

# dog_img = io.imread('src/dog.jpg')
# io.imshow(dog_img)
# io.show()
#
# dog_string = dog_img.tostring()
# print(dog_img.shape)
# # print(dog_string)
#
# reconstructed_dog_1d = np.fromstring(dog_string, dtype=np.uint8)
# print(reconstructed_dog_1d)
#
# reconstructed_dog_img = reconstructed_dog_1d.reshape(dog_img.shape)
#
# io.imshow(reconstructed_dog_img)
# io.show()

files = ['dog_img/dog1.jpg', 'dog_img/dog2.jpg', 'dog_img/dog3.jpg', 'dog_img/dog4.jpg']
tfrecords_filename = 'dogs.tfrecords'
originals = []


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def write():
    global files
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)
    for img_path in files:
        img = np.array(Image.open(img_path))

        height = img.shape[0]
        width = img.shape[1]

        originals.append(img)

        img_raw = img.tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'image_raw': _bytes_feature(img_raw)
        }))

        writer.write(example.SerializeToString())
    writer.close()


def read():
    reconstructed_images = []
    record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)

    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)

        height = int(example.features.feature['height'].int64_list.value[0])
        width = int(example.features.feature['width'].int64_list.value[0])
        img_string = (example.features.feature['image_raw'].bytes_list.value[0])

        img_1d = np.fromstring(img_string, dtype=np.uint8)
        reconstructed_image = img_1d.reshape((height, width, -1))
        reconstructed_images.append(reconstructed_image)
    record_iterator.close()
    return reconstructed_images


if __name__ == '__main__':
    write()
    for img in read():
        io.imshow(img)
        io.show()
    # Оно живое!
