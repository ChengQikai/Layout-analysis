import lxml.etree as ET
import sys
from HelperMethods import get_coordinates_from_xml, create_page_xml
import matplotlib.pyplot as plt
from scipy import misc
from PIL import Image, ImageDraw
import numpy as np
import os

x_treshold = 50
y_treshold = 70

def get_img_coords(img, coords):
    res = img.copy()
    label_img = Image.new("L", (img.shape[1], img.shape[0]), 0)
    for rect in coords:
        rect.append(rect[0])
        ImageDraw.Draw(label_img).line(rect, width=15, fill=1)
    label_img = np.array(label_img)
    for y in range(res.shape[0]):
        for x in range(res.shape[1]):
            if label_img[y][x] == 1:
                res[y][x] = (0, 255, 0)
    return res

def merge_paragraphs(coordinates):
    new_coords = []
    for index1 in range(0, len(coordinates)):
        rect1 = coordinates[index1]
        r1min = np.amin(rect1, axis=0)
        r1max = np.amax(rect1, axis=0)

        if rect1 is None:
            continue

        # index2 = index1 + 1
        index2 = 0
        while index2 < len(coordinates):
            rect2 = coordinates[index2]
            if index2 == index1:
                index2 += 1
                continue
            if rect2 is None:
                index2 += 1
                continue

            r2min = np.amin(rect2, axis=0)
            r2max = np.amax(rect2, axis=0)

            if ((abs(r1min[0] - r2min[0]) < x_treshold) and (abs(r1max[0] - r2max[0]) < x_treshold) and \
                    (abs(r1min[1] - r2max[1]) < y_treshold or abs(r1max[1] - r2min[1]) < y_treshold)):
                coordinates[index2] = None
                xs = min(r1min[0], r2min[0])
                xe = max(r1max[0], r2max[0])
                ys = min(r1min[1], r2min[1])
                ye = max(r1max[1], r2max[1])
                rect1 = [(xs, ys), (xe, ys), (xe, ye), (xs, ye)]
                r1min = np.amin(rect1, axis=0)
                r1max = np.amax(rect1, axis=0)
                index2 = 0
            elif r1min[0] > r2min[0] and r1max[0] < r2max[0] and r1min[1] > r2min[1] and r1max[1] < r2max[1]:
                coordinates[index2] = None
                xs = min(r1min[0], r2min[0])
                xe = max(r1max[0], r2max[0])
                ys = min(r1min[1], r2min[1])
                ye = max(r1max[1], r2max[1])
                rect1 = [(xs, ys), (xe, ys), (xe, ye), (xs, ye)]
                r1min = np.amin(rect1, axis=0)
                r1max = np.amax(rect1, axis=0)
                index2 = 0
            elif r2min[0] > r1min[0] and r2max[0] < r1max[0] and r2min[1] > r1min[1] and r2max[1] < r1max[1]:
                coordinates[index2] = None
                xs = min(r1min[0], r2min[0])
                xe = max(r1max[0], r2max[0])
                ys = min(r1min[1], r2min[1])
                ye = max(r1max[1], r2max[1])
                rect1 = [(xs, ys), (xe, ys), (xe, ye), (xs, ye)]
                r1min = np.amin(rect1, axis=0)
                r1max = np.amax(rect1, axis=0)
                index2 = 0
            else:
                index2 += 1
        new_coords.append([(r1min[0], r1min[1]), (r1max[0], r1min[1]), (r1max[0], r1max[1]), (r1min[0], r1max[1])])

    for i in range(len(new_coords)-1):
        rect1 = new_coords[i]
        r1min = np.amin(rect1, axis=0)
        r1max = np.amax(rect1, axis=0)
        erased = False

        for j in range(i+1, len(new_coords)):
            rect2 = new_coords[j]
            r2min = np.amin(rect2, axis=0)
            r2max = np.amax(rect2, axis=0)
            if r1min[0] > r2min[0] and r1max[0] < r2max[0] and r1min[1] > r2min[1] and r1max[1] < r2max[1]:
                new_coords[i] = None
                erased = True
                break
            elif r2min[0] > r1min[0] and r2max[0] < r1max[0] and r2min[1] > r1min[1] and r2max[1] < r1max[1]:
                new_coords[j] = None

        if erased:
            break

    new_coords = [cord for cord in new_coords if cord is not None]

    return new_coords


def get_img_coords(img, coords):
    res = img.copy()
    label_img = Image.new("L", (img.shape[1], img.shape[0]), 0)
    for rect in coords:
        rect.append(rect[0])
        ImageDraw.Draw(label_img).line(rect, width=15, fill=1)
    label_img = np.array(label_img)
    for y in range(res.shape[0]):
        for x in range(res.shape[1]):
            if label_img[y][x] == 1:
                res[y][x] = (0, 255, 0)
    return res


def main():
    path = 'F:\\splited_dataset\\train'
    out = 'F:\\merged_xml_train2'

    files = os.listdir(path)
    xml_files = [filename for filename in files if filename.endswith('.xml')]
    img_files = [filename for filename in files if filename.endswith('.jpg')]
    out_files = os.listdir(out)
    out_files = [filename for filename in out_files if filename.endswith('.xml')]

    for xml_file in xml_files:
        if xml_file in out_files:
            continue
        img_file = xml_file.replace('.xml', '.jpg')
        if img_file in img_files:
            xml_coordinates, width, height = get_coordinates_from_xml(path + '\\' + xml_file)
            xml_coordinates = merge_paragraphs(xml_coordinates)
            merged_xml = create_page_xml(xml_coordinates, width, height, xml_file.replace('.xml', ''))
            with open(out + '\\' + xml_file, 'wb') as f:
                f.write(merged_xml)
    return 0


if __name__ == "__main__":
    sys.exit(main())
