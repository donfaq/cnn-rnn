import cv2
import logging as LOG
import tensorflow as tf
import base64
LOG.basicConfig(level=LOG.INFO)


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
        LOG.info('splitting video into frames')
        videofile = cv2.VideoCapture(self.videofile_name)
        readSucceded = True
        while readSucceded:
            readSucceded, frame = videofile.read()
            if not readSucceded:
                break
            self.frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.number_of_frames = len(self.frames)
        assert self.number_of_frames is not 0
        LOG.info('done splitting')

    def group_frames(self, group_length, step):
        LOG.info('strarting grouping')
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
        LOG.info('done grouping')

    def decode_frames(self):
        LOG.info(self.number_of_grouped_frames)
        for frame in self.grouped_frames:
            self.grouped_frames_decoded.append(self.base64_encoder(frame))

    @staticmethod
    def base64_encoder(rgb_image):
        retval, buffer = cv2.imencode('.jpg', cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
        return base64.b64encode(buffer)

    @staticmethod
    def base64_decoder(byte_image):
        return base64.b64decode(byte_image)