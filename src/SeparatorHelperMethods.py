#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.


import math
from scipy import ndimage
import numpy as np
import cv2
from skimage.draw import line
import random

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
    dilated = ndimage.morphology.grey_dilation(input, size=2)
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


def rotate_line(line, rotation, height = 0):
    M = cv2.getRotationMatrix2D((0,0),rotation,1)
    baseline = np.array([line])
    rotated_line = cv2.transform(baseline,M)[0]
    return rotated_line


def fit_separators_into_segmentation_mask(segmentation_mask, line_coords):
    height, width = segmentation_mask.shape
    rotated_mask, rotation = rotate_data(segmentation_mask, line_coords)

    separators = find_separators(rotated_mask)
    lines = np.zeros((height, width))

    for separator in separators:
        rotated_line = rotate_line([(separator, 0), (separator, height)], rotation)
        rr, cc = line(rotated_line[0][1],rotated_line[0][0], rotated_line[1][1], rotated_line[1][0])
        rr = np.clip(rr, 0, height-1)
        cc = np.clip(cc, 0, width-1)
        lines[rr, cc] = 1
        segmentation_mask[rr, cc] = 2

    return segmentation_mask
