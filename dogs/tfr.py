import tensorflow as tf


def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_sequence_example(image_encoded, label, height, width):
    colorspace = 'RGB'
    channels = 3
    image_format = 'JPEG'

    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'height': _int64_feature(height),
                'weight': _int64_feature(width),
                'colorspace': _bytes_feature(colorspace),
                'channels': _int64_feature(channels),
                'format': _bytes_feature(image_format),
                'label': _int64_feature(label),
                'image': _bytes_feature(image_encoded)
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


if __name__ == '__main__':
    _, h, w = process_image("dogs_img/dog1.jpg")
    print(h, w)



