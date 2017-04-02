import cv2
import tensorflow as tf


class Frames:
    videofile_name = ''
    frames = []
    grouped_frames = []
    grouped_frames_decoded = []
    group_numbers_sequence = []
    frames_number_in_group = []
    number_of_frames = int
    number_of_grouped_frames = int

    def __init__(self, videofile_name, group_length=3, step=1):
        self.videofile_name = videofile_name
        self.number_of_frames = 0
        self.video_to_frames()
        self.group_frames(group_length=group_length, step=step)
        self.decode_frames()

    def video_to_frames(self):
        videofile = cv2.VideoCapture(self.videofile_name)
        readSucceded = True
        while readSucceded:
            readSucceded, frame = videofile.read()
            if not readSucceded:
                break
            self.frames.append(frame)
        assert len(self.frames) is not 0
        self.number_of_frames = len(self.frames)

    def group_frames(self, group_length, step):
        current_frame_group = 0
        start_group_index = 0
        end_group_index = group_length
        while end_group_index <= self.number_of_frames:
            number_in_group = 0
            for frame in self.frames[start_group_index:end_group_index]:
                self.grouped_frames.append(frame)
                self.group_numbers_sequence.append(current_frame_group)
                self.frames_number_in_group.append(number_in_group)
                number_in_group += 1
            start_group_index += step
            end_group_index += step
            current_frame_group += 1
        self.number_of_grouped_frames = len(self.grouped_frames)

    def decode_frames(self):
        with tf.Session():
            for frame in self.grouped_frames:
                self.grouped_frames_decoded.append(
                    tf.image.encode_jpeg(frame, format='rgb', quality=100).eval())


class Dataset:
    VIDEOS_PATH_PATTERN = ''
    videofiles_names_list = []

    def __init__(self, input_path):
        self.VIDEOS_PATH_PATTERN = input_path
        self.videofiles_names_list = tf.gfile.Glob(input_path)

    def create_dataset(self, group_length=3, step=1):
        for vname in self.videofiles_names_list:
            with open('{vname}.tfrecord'.format(vname=vname), 'w') as tfrecord:
                writer = tf.python_io.TFRecordWriter(tfrecord.name)
                frames = Frames(vname, group_length=group_length, step=step)
                example = self.make_sequence_example(frames)
                writer.write(example.SerializeToString())
                writer.close()
            tfrecord.close()

    @staticmethod
    def make_sequence_example(frames):
        example_sequence = tf.train.SequenceExample()
        example_sequence.context.feature["length"].int64_list.value.append(frames.number_of_grouped_frames)
        frames_sequence = example_sequence.feature_lists.feature_list["frame"]
        groups = example_sequence.feature_lists.feature_list["group"]
        number_in_group = example_sequence.feature_lists.feature_list["number_in_group"]
        for frame, group, number in zip(frames.grouped_frames_decoded, frames.group_numbers_sequence,
                                        frames.frames_number_in_group):
            if frame is not None:
                frames_sequence.feature.add().bytes_list.value.append(frame)
            if group is not None:
                groups.feature.add().int64_list.value.append(group)
            if number is not None:
                number_in_group.feature.add().int64_list.value.append(number)
        return example_sequence

    @staticmethod
    def parse_single_example(filename):
        filename_queue = tf.train.string_input_producer([filename], num_epochs=1)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        context_features = {
            "length": tf.FixedLenFeature([], dtype=tf.int64)
        }

        sequence_features = {
            "frame": tf.FixedLenSequenceFeature([], dtype=tf.string),
            "group": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "number_in_group": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        }

        context_parsed, sequence_parsed = tf.parse_single_sequence_example(
            serialized=serialized_example,
            context_features=context_features,
            sequence_features=sequence_features
        )

        return context_parsed, sequence_parsed


if __name__ == '__main__':
    dataset = Dataset('src/*.avi')
    # dataset.create_dataset()
    context, sequence = dataset.parse_single_example('src/1.avi.tfrecord')
    context_features = tf.contrib.learn.run_n(context, n=1, feed_dict=None)
    sequence_features_length = context_features[0]['length']
    sequence_features = tf.contrib.learn.run_n(sequence, n=1, feed_dict=None)
    print(sequence_features[0]['frame'][1])
