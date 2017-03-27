import tensorflow as tf
import skimage.io as io

IMAGE_HEIGHT = 720
IMAGE_WEIGHT = 1024

tfrecords_filename = 'dogs.tfrecords'


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serializer_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serializer_example,
        features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string)
        })

    image = tf.decode_raw(features['image_raw'], tf.uint8)

    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    image_shape = tf.stack([height, width, 3])
    image = tf.reshape(image, image_shape)

    image_size_const = tf.constant((IMAGE_HEIGHT, IMAGE_WEIGHT, 3), dtype=tf.int32)

    resized_image = tf.image.resize_image_with_crop_or_pad(image=image,
                                                           target_height=IMAGE_HEIGHT,
                                                           target_width=IMAGE_WEIGHT)
    images = tf.train.shuffle_batch([resized_image],
                                    batch_size=2,
                                    capacity=30,
                                    num_threads=2,
                                    min_after_dequeue=10)
    return images


if __name__ == '__main__':
    filename_queue = tf.train.string_input_producer([tfrecords_filename], num_epochs=10)

    image = read_and_decode(filename_queue)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(2):
            img = sess.run([image])
            #print(img[0, :, :, :].shape)
            print('Current batch')
            io.imshow(img[0][0])
            io.show()
            io.imshow(img[0][1])
            io.show()
        coord.request_stop()
        coord.join(threads)
