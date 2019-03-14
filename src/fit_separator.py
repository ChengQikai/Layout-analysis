import sys
from scipy import misc
import matplotlib.pyplot as plt
import lxml.etree as ET
import math
from scipy import ndimage
import numpy as np
from DocumentAnalyzer import DocumentAnalyzer
import os
from PIL import Image, ImageDraw


def get_img_coords(img, coords):
    res = img.copy()
    label_img = Image.new("L", (img.shape[1], img.shape[0]), 0)
    for rect in coords:
        rect.append(rect[0])
        ImageDraw.Draw(label_img).line(rect, width=4, fill=1)
    label_img = np.array(label_img)
    for y in range(res.shape[0]):
        for x in range(res.shape[1]):
            if label_img[y][x] == 1:
                res[y][x] = (0, 255, 0)
    return res

def rotate_data(img_data, lines):
    lines_info = []
    for line in lines:
        first_line_point = line[0]
        last_line_point = line[-1]
        if last_line_point[0] != first_line_point[0]:
            rotation = math.degrees(math.atan((last_line_point[0] - first_line_point[0]) / (last_line_point[1] - first_line_point[1])))
            length = math.sqrt(math.pow(last_line_point[1] - first_line_point[1],2) + math.pow(last_line_point[0]-first_line_point[0],2))
            lines_info.append((length,rotation))
        else:
            lines_info.append((0,0))
    lines_info = sorted(lines_info,key = lambda x: x[0],reverse = True)
    lines_info = lines_info[0:int(len(lines_info)/2)]
    rotation_sum = sum(item[1] for item in lines_info)
    rotation = 0
    if len(lines_info) > 0:
        rotation = rotation_sum/len(lines_info)
    rotated_data = ndimage.interpolation.rotate(img_data, rotation, reshape=False)
    return rotated_data, -rotation


def get_line_coords(path):
    root = ET.parse(path).getroot()
    baselines = []
    for baseline in root.iter('{http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}Baseline'):
        baseline_coord = list()
        coords_string = baseline.get('points').split(' ')
        for point_string in coords_string:
            point = point_string.split(',')
            baseline_coord.append((int(point[1]), int(point[0])))
        baselines.append(baseline_coord)
    return baselines

def get_1d(data):
    return np.sum(data, axis=0)

def nonmaxima_suppression(input):
    dilated = ndimage.morphology.grey_dilation(input, size=(2))
    return(input * (input==dilated))

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

    isSeparator1D = (line1d > paragraph1d) * line1d
    nonmaxima_separator = nonmaxima_suppression(isSeparator1D)
    # nonmaxima_separator = isSeparator1D
    return np.nonzero(nonmaxima_separator)[0]

def main():
    analyzer = DocumentAnalyzer(scale=1)
    output = '../separator2'
    path = '../data/vertical1/test/'
    inputs_path = path + 'inputs/'
    labels_path = path + 'labels/'
    xmls_path ='../test/test/'

    inputs_names = os.listdir(inputs_path)
    inputs_names = [filename for filename in inputs_names if filename.endswith('.png')]
    labels_names = os.listdir(labels_path)
    labels_names = [filename for filename in labels_names if filename.endswith('.png')]
    xml_names = os.listdir(xmls_path)
    xml_names = [filename for filename in xml_names if filename.endswith('.xml')]

    for input_name in inputs_names:
        con_name = input_name.replace('.png', '.xml')
        if input_name in labels_names and con_name in xml_names:
            img_path = inputs_path + input_name
            path = labels_path + input_name
            xml_path = xmls_path + con_name

            img_o = misc.imread(img_path, mode="RGB")
            img = misc.imread(path)
            line_coords = get_line_coords(xml_path)

            probability_maps = analyzer.get_probability_mask(img_o)
            segmentation_mask = analyzer.get_segmentation_map(probability_maps)

            rotated_img, rotation = rotate_data(img, line_coords)
            rotated_img_o, rotation = rotate_data(img_o, line_coords)
            rotated_img1, rotation1 = rotate_data(segmentation_mask, line_coords)

            sep = find_separators(rotated_img)
            sep_len = len(sep)
            sep1 = find_separators(rotated_img1)
            sep1_len = len(sep1)

            fig = plt.figure(dpi=300)
            plt.subplot(2, 3, 1)
            plt.imshow(img_o)
            plt.subplot(2, 3, 2)
            plt.imshow(segmentation_mask)
            plt.subplot(2, 3, 3)
            plt.title('Ground {}: {}'.format(sep_len, 0))
            plt.imshow(get_separator_img(rotated_img_o.copy(), sep, (0, 255, 0)))
            plt.subplot(2, 3, 4)
            plt.title('CNN {}: {}'.format(sep1_len, 0))
            plt.imshow(get_separator_img(rotated_img_o.copy(), sep1, (0, 255, 0)))

            rotated_img1 = get_separator_img(rotated_img1.copy(), sep1, 2)
            # labels = analyzer.label_clustering(rotated_img1)
            labels = analyzer.clustering(rotated_img1)
            plt.subplot(2, 3, 5)
            plt.imshow(labels)
            coords = analyzer.get_coordinates(labels, rotated_img1.shape[0], rotated_img1.shape[1])
            plt.subplot(2, 3, 6)
            result = get_img_coords(rotated_img_o, coords)
            plt.imshow(result)

            plt.savefig('{}/{}'.format(output, input_name))
            plt.close(fig)
            print('Complete one')


    print("Complete all")
    return 0


if __name__ == "__main__":
    sys.exit(main())