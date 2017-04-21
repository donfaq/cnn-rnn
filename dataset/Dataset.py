from dataset.Frames import *
import logging as LOG
LOG.basicConfig(level=LOG.INFO)


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
        label = int(name.split('_')[-1])
        return label

    @staticmethod
    def write_tfrecord_file(filename, example):
        with open(filename, 'w') as tfrecord:
            writer = tf.python_io.TFRecordWriter(tfrecord.name)
            writer.write(example.SerializeToString())
        writer.close()
        tfrecord.close()

    def create_dataset(self, batch_size=30, group_length=3, step=1):
        with tf.Session() as sess:
            for vname in self.videofiles_names_list:
                LOG.info('\n' + vname)
                label = self.parse_filename(vname)
                LOG.info('obtaining frames')
                frames = Frames(sess, vname, group_length=group_length, step=step)
                LOG.info('creating example')
                part_count = 0
                for example in self.make_examples_list(frames, label, batch_size):
                    filename = '{vname}_part{part}.tfrecord'.format(vname=vname, part=part_count)
                    self.write_tfrecord_file(filename, example)
                    part_count += 1
                    LOG.info('writing example')
                LOG.info('done writing')
                del frames
        sess.close()
        del sess

    @staticmethod
    def make_example(frames_list, label):
        example = tf.train.SequenceExample()
        # context features
        example.context.feature["label"].int64_list.value.append(label)
        # sequence features
        frames_sequence = example.feature_lists.feature_list["frame"]
        # 'fill' sequence features
        for frame in frames_list:
            frames_sequence.feature.add().bytes_list.value.append(frame)
        return example

    @staticmethod
    def make_examples_list(frames: Frames, label, batch_size) -> list:
        batch_counter = 0
        tmp_frames = []
        examples_list = []
        for frame in frames.grouped_frames_decoded:
            if batch_counter < batch_size:
                if frame is not None:
                    tmp_frames.append(frame)
                batch_counter += 1
            else:
                examples_list.append(Dataset.make_example(tmp_frames, label))
                batch_counter = 0
                tmp_frames = []
                if frame is not None:
                    tmp_frames.append(frame)
                batch_counter += 1
        return examples_list


if __name__ == '__main__':
    Dataset('SDHA2010Interaction/segmented_set?/*.avi').create_dataset()