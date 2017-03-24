import cv2
from matplotlib import pyplot as plt


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
    count = 0
    hasNext = True
    while hasNext:
        hasNext, image = vid.read()
        if not hasNext:
            break
        count += 1
        elements.append(tuple((count, image)))
    return elements


if __name__ == '__main__':
    data = group(video_to_frames("11_2_3.avi", 2))
    print(data[0][1][1] is data[1][0][1])
    # True [[[1, img1], [2,img2], [3,img3], [[2,img2], [..], [..]]]

    plt.imshow(data[1][1][1], cmap='gray', interpolation='bicubic')
    plt.show()
    print(_int64_feature(data))