import cv2
from matplotlib import pyplot as plt


def split_video_frames(filename, step):
    vid = cv2.VideoCapture(filename)
    hasNext, image = vid.read()
    element = []
    result = []
    count = 0
    hasNext = True
    while hasNext:
        hasNext, image = vid.read()
        if not hasNext:
            break
        count += 1
        element.append(tuple((count, image)))
        if count % step == 0:
            result.append(element)
            element = []
    return result


if __name__ == '__main__':
    frames = split_video_frames("11_2_3.avi", 2)
    plt.imshow(frames[1][1][1], cmap='gray', interpolation='bicubic')
    plt.show()
