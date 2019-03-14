import math
from scipy import ndimage
import numpy as np


def rotate_data(img_data, lines):
    lines_info = []
    for line in lines:
        first_line_point = line[0]
        last_line_point = line[-1]
        if last_line_point[0] != first_line_point[0]:
            rotation = math.degrees(math.atan((last_line_point[0] - first_line_point[0])
                                              / (last_line_point[1] - first_line_point[1])))
            length = math.sqrt(math.pow(last_line_point[1] - first_line_point[1],2)
                               + math.pow(last_line_point[0]- first_line_point[0],2))
            lines_info.append((length,rotation))
        else:
            lines_info.append((0,0))

    lines_info = sorted(lines_info, key=lambda x: x[0], reverse=True)
    lines_info = lines_info[0:int(len(lines_info)/2)]
    rotation_sum = sum(item[1] for item in lines_info)
    rotation = 0

    if len(lines_info) > 0:
        rotation = rotation_sum/len(lines_info)

    rotated_data = ndimage.interpolation.rotate(img_data, rotation, reshape=False)
    return rotated_data, - rotation


def get_1d(data):
    return np.sum(data, axis=0)


def nonmaxima_suppression(input):
    dilated = ndimage.morphology.grey_dilation(input, size=(2))
    return input * (input == dilated)


def get_separator_img(img, separators, value):
    for separator in separators:
        img[:, separator] = value

    return img


def find_separators(img):
    paragraph_img = img.copy()
    line_img = img.copy()

    np.putmask(paragraph_img, paragraph_img == 2, 0)
    np.putmask(line_img, line_img == 1, 0)

    paragraph1d = get_1d(paragraph_img)
    line1d = get_1d(line_img)

    is_separator_1d = (line1d > paragraph1d) * line1d
    is_separator_1d = nonmaxima_suppression(is_separator_1d)
    return np.nonzero(is_separator_1d)[0]