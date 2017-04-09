import tensorflow as tf
import tensorflow.contrib.slim as slim
from network.model import cnnrnn

train_log_dir = 'train_logs/'
if not tf.gfile.Exists(train_log_dir):
    tf.gfile.MakeDirs(train_log_dir)


def load_data():
    pass

with tf.Graph().as_default():
    # TODO: load_data
    frames, labels = load_data()
    # TODO: predictions = model
    predictions = cnnrnn(frames)

    slim.losses.softmax_cross_entropy(predictions, labels)
    total_loss = slim.losses.get_total_loss()
    tf.summary.scalar('losses/total_loss', total_loss)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=.001)

    train_tensor = slim.learning.create_train_op(total_loss, optimizer)

    slim.learning.train(train_tensor, train_log_dir)
