import cv2
from matplotlib import pyplot as plt
import numpy as np


def group(data, batch=3, step=1):
    start = 0
    end = batch
    result = []
    while end <= len(data):
        result.append(data[start:end])
        start += step
        end += step
    return result


def video_to_frames(filename, step):
    vid = cv2.VideoCapture(filename)
    elements = []
    hasNext = True
    while hasNext:
        hasNext, image = vid.read()
        if not hasNext:
            break
        image = image.tobytes()
        elements.append(image)
    return elements


if __name__ == '__main__':
    data = group(video_to_frames("src/11_2_3.avi", 2))
    assert isinstance(data[0][1], bytes)


    # assert data[0][1] is data[1][0]
    #
    # plt.imshow(data[1][1])
    # plt.show()
