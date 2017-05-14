import tensorflow as tf
from dataset import Reader
from network.network import Network

import config

FLAGS = tf.app.flags.FLAGS


class Learning:
    def __init__(self):
        self.train_reader = Reader.Reader("dataset/data/segmented_set1/*.tfr")
        self.test_reader = Reader.Reader("dataset/data/segmented_set2/*.tfr")
        self.train_logs_path = 'network/train_logs'
        self.test_logs_path = 'network/test_logs'
        self.chkpt_file = self.train_logs_path + "/model.ckpt"
        self.ten_accuracy = []
        self.epoch_accuracy = []
        if FLAGS.test is True:
            self.is_training = False
            self._eval_test()
        else:
            self.is_training = True
            self.evaluate_model()

    def evaluate_model(self):
        self.net = Network(self.is_training)
        self.train_writer = tf.summary.FileWriter(self.train_logs_path, graph=self.net.graph)
        # self.test_writer = tf.summary.FileWriter(self.test_logs_path, graph=self.net.graph)

        with tf.Session(graph=self.net.graph) as sess:
            if FLAGS.restore:
                self.net.saver.restore(sess, self.chkpt_file)
                print("Model restored.")
            else:
                sess.run(tf.local_variables_initializer())
                sess.run(tf.global_variables_initializer())
                print('Parameters were initialized')
            self.net.print_model()

            step_num = 1
            max_steps = FLAGS.epoch * 100
            while step_num <= max_steps:
                if step_num % 10 == 0:
                    gs, acc = self._train_step(sess,
                                               tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                                               tf.RunMetadata())
                    self._add_accuracy(step_num, gs, acc)
                    save_path = self.net.saver.save(sess, self.chkpt_file)
                    print("[train] Model saved in file: %s" % save_path)
                    if step_num % 100 == 0:
                        self._eval_test()
                else:
                    gs, acc = self._train_step(sess)
                    self._add_accuracy(step_num, gs, acc)
                step_num += 1

    def _train_step(self, sess, run_options=None, run_metadata=None):
        if run_options is not None:
            _, summary, global_step, accuracy = sess.run(
                [self.net.train_step, self.net.summary_op, self.net.global_step, self.net.accuracy],
                feed_dict=self.next_example(), options=run_options, run_metadata=run_metadata)
            self.train_writer.add_run_metadata(run_metadata, 'step{}'.format(global_step), global_step)
            print('[train] Adding run metadata for', global_step)
        else:
            _, summary, global_step, accuracy = sess.run(
                [self.net.train_step, self.net.summary_op, self.net.global_step, self.net.accuracy],
                feed_dict=self.next_example())
        self.train_writer.add_summary(summary, global_step)
        return global_step, accuracy

    def next_example(self):
        if self.is_training is False:
            label, example = self.test_reader.get_random_example()
        else:
            label, example = self.train_reader.get_random_example()
        return {self.net.x: example, self.net.y: label}

    def _add_accuracy(self, step_num, global_step, accuracy):
        print('Accuracy on step {}: {}'.format(global_step, accuracy))
        if accuracy == 1:
            self.ten_accuracy.append(1)
        if step_num % 10 == 0:
            print('Accuracy for 10 steps:', sum(self.ten_accuracy) / 10)
            self.epoch_accuracy.append(sum(self.ten_accuracy) / 10)
            self.ten_accuracy = []
            if step_num % 100 == 0:
                print('Test accuracy:', sum(self.epoch_accuracy) / len(self.epoch_accuracy))
                self.epoch_accuracy = []

    def _eval_test(self):
        pass
