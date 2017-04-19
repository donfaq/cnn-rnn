import tensorflow as tf
import tensorflow.contrib.slim as slim
from network.model import cnnrnn
from dataset.tfrecord_reader import read_binary_video

train_log_dir = 'train_logs/'
if not tf.gfile.Exists(train_log_dir):
    tf.gfile.MakeDirs(train_log_dir)
videofiles_dir = 'dataset/SDHA2010Interaction/segmented_set1/15_13_3.avi.tfrecord'


def load_videofile(path):
    label, bgr_frames = read_binary_video(path)
    with tf.Session() as sess:
        rgb_frames = []
        for frame in bgr_frames:
            # tf.reverse(frame, axis=[-1]) convert from BGR into RGB format
            rgb_frames.append(tf.to_float(tf.reverse(frame, axis=[-1])))
        return label, rgb_frames


if __name__ == '__main__':
    with tf.Graph().as_default():
        label, frames = load_videofile('dataset/SDHA2010Interaction/segmented_set2/0_11_4.avi.tfrecord')
        # TODO: predictions = model
        frames = tf.image.resize_image_with_crop_or_pad(frames, 299, 299)
        print(frames)
        print(label)
        oh_labels = tf.one_hot(label, depth=3)
        print(tf.Session().run(label))
        # 299 x 299 x 3
        # net = slim.conv2d(frames, 32, [3, 3], stride=2,
        #                   padding='VALID', scope='Conv2d_1a_3x3')
        # predictions = cnnrnn(frames)
        # oh_labels = tf.one_hot(label, depth=3)
        # tf.losses.softmax_cross_entropy(predictions, oh_labels)
        # total_loss = slim.losses.get_total_loss()
        # tf.summary.scalar('losses/total_loss', total_loss)
        #
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=.001)
        #
        # train_tensor = slim.learning.create_train_op(total_loss, optimizer)
        #
        # slim.learning.train(train_tensor, train_log_dir)
