import cv2
import tensorflow as tf


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
            self.frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
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
                self.session.run(tf.image.encode_jpeg(frame, format='rgb', quality=80)))


class Dataset:
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

    def create_dataset(self, batch_size=30, group_length=3, step=1):
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
                    for example in self.make_examples_list(frames, label, extention, batch_size):
                        print('writing example')
                        writer.write(example.SerializeToString())
                    print('done writing')
                    writer.close()
                    del writer, frames
                tfrecord.close()
        sess.close()
        del sess

    @staticmethod
    def make_example(frames_list, groups_list, numbers_list, label, extention, length):
        example = tf.train.SequenceExample()
        # context features
        example.context.feature["length"].int64_list.value.append(length)
        example.context.feature["label"].int64_list.value.append(label)
        example.context.feature["extention"].bytes_list.value.append(extention)
        # sequence features
        frames_sequence = example.feature_lists.feature_list["frame"]
        groups = example.feature_lists.feature_list["group"]
        number_in_group = example.feature_lists.feature_list["number_in_group"]
        # 'fill' sequence features
        for frame, group, number in zip(frames_list, groups_list, numbers_list):
            frames_sequence.feature.add().bytes_list.value.append(frame)
            groups.feature.add().int64_list.value.append(group)
            number_in_group.feature.add().int64_list.value.append(number)
        return example

    @staticmethod
    def make_examples_list(frames: Frames, label, extention, batch_size) -> list:
        batch_counter = 0
        tmp_frames = []
        tmp_groups = []
        tmp_numbers = []
        examples_list = []
        for frame, group, number in zip(frames.grouped_frames_decoded, frames.group_numbers_sequence,
                                        frames.frames_number_in_group):
            if batch_counter < batch_size:
                if frame is not None:
                    tmp_frames.append(frame)
                if group is not None:
                    tmp_groups.append(group)
                if number is not None:
                    tmp_numbers.append(number)
                batch_counter += 1
            else:
                examples_list.append(Dataset.make_example(tmp_frames, tmp_groups, tmp_numbers,
                                                          label, extention, len(tmp_frames)))
                batch_counter = 0
                tmp_frames = []
                tmp_groups = []
                tmp_numbers = []
                if frame is not None:
                    tmp_frames.append(frame)
                if group is not None:
                    tmp_groups.append(group)
                if number is not None:
                    tmp_numbers.append(number)
                batch_counter += 1
        return examples_list
