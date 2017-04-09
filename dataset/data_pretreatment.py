import cv2
import tensorflow as tf
from PIL import Image


class Frames:
    def __init__(self, session, videofile_name, group_length=3, step=1):
        self.frames = []
        self.grouped_frames = []
        self.grouped_frames_decoded = []
        self.group_numbers_sequence = []
        self.frames_number_in_group = []
        self.number_of_frames = 0
        self.number_of_grouped_frames = 0
        self.videofile_name = videofile_name
        self.video_to_frames()
        self.group_frames(group_length=group_length, step=step)
        self.session = session
        self.decode_frames()

    def video_to_frames(self):
        print('splitting video into frames')
        videofile = cv2.VideoCapture(self.videofile_name)
        readSucceded = True
        while readSucceded:
            readSucceded, frame = videofile.read()
            if not readSucceded:
                break
            self.frames.append(frame)
        self.number_of_frames = len(self.frames)
        assert self.number_of_frames is not 0
        print('done splitting')

    def group_frames(self, group_length, step):
        print('strarting grouping')
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
        print('done grouping')

    def decode_frames(self):
        print(self.number_of_grouped_frames)
        for frame in self.grouped_frames:
            self.grouped_frames_decoded.append(
                self.session.run(tf.image.encode_jpeg(frame, format='rgb', quality=70)))


class Dataset:
    VIDEOS_PATH_PATTERN = ''
    videofiles_names_list = []

    def __init__(self, input_path=str()):
        self.VIDEOS_PATH_PATTERN = input_path
        self.videofiles_names_list = tf.gfile.Glob(input_path)

    @staticmethod
    def parse_filename(file_path):
        path_parts = file_path.split('/')
        file_name = path_parts[-1]
        fname_parts = file_name.split('.')
        name = fname_parts[0]
        extention = str.encode(fname_parts[1])
        label = int(name.split('_')[-1])
        return label, extention

    def create_dataset(self, group_length=3, step=1):
        with tf.Session() as sess:
            for vname in self.videofiles_names_list:
                print('\n' + vname)
                with open('{vname}.tfrecord'.format(vname=vname), 'w') as tfrecord:
                    # videofile name like 21_2_3.avi
                    label, extention = self.parse_filename(vname)
                    writer = tf.python_io.TFRecordWriter(tfrecord.name)
                    print('obtaining frames')
                    frames = Frames(sess, vname, group_length=group_length, step=step)
                    print('creating example')
                    example = self.make_sequence_example(frames, label, extention)
                    print('writing example')
                    writer.write(example.SerializeToString())
                    print('done writing')
                    writer.close()
                    del writer, frames, example
                tfrecord.close()
        sess.close()
        del sess

    @staticmethod
    def make_sequence_example(frames, label, extention):
        example_sequence = tf.train.SequenceExample()
        # context features
        example_sequence.context.feature["length"].int64_list.value.append(frames.number_of_grouped_frames)
        example_sequence.context.feature["label"].int64_list.value.append(label)
        example_sequence.context.feature["extention"].bytes_list.value.append(extention)
        # sequence features
        frames_sequence = example_sequence.feature_lists.feature_list["frame"]
        groups = example_sequence.feature_lists.feature_list["group"]
        number_in_group = example_sequence.feature_lists.feature_list["number_in_group"]
        # 'fill' sequence features
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
    def read(filename_queue):

        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        context_features = {
            "length": tf.FixedLenFeature([], dtype=tf.int64),
            "label": tf.FixedLenFeature([], dtype=tf.int64),
            "extention": tf.FixedLenFeature([], dtype=tf.string)
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
    queue = tf.train.string_input_producer(['SDHA2010Interaction/segmented_set1/0_1_4.avi.tfrecord'])
    context, sequence = Dataset().read(queue)
    my_img = tf.image.decode_jpeg(sequence['frame'][1], channels=3)
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        image = my_img.eval()
        print(image.shape)

        RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(RGB_img, 'RGB')
        img.show()

        coord.request_stop()
        coord.join(threads)
