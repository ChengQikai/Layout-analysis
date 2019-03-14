import sys
from HelperMethods import get_coordinates_from_xml, create_page_xml
from PIL import Image, ImageDraw
import numpy as np
import os
import random



y_treshold = 100

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

def overlap(min1, max1, min2, max2):
    min_length = min(max1 - min1, max2 - min2)
    return (max(0, min(max1, max2) - max(min1, min2))) / min_length

def is_almost_same_size(min1, max1, min2, max2):
    first_len = max1 - min1
    second_len = max2 - min2

    return max(first_len, second_len)  * 0.6 < min(first_len, second_len)

def merge_paragraphs(coordinates):
    new_coords = []

    for index1 in range(0, len(coordinates)):
        rect1 = coordinates[index1]
        r1min = np.amin(rect1, axis=0)
        r1max = np.amax(rect1, axis=0)

        if rect1 is None:
            continue

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

            if (abs(r1min[1] - r2max[1]) < y_treshold or abs(r1max[1] - r2min[1]) < y_treshold) and overlap(r1min[0], r1max[0], r2min[0], r2max[0]) > 0.7 \
                    and is_almost_same_size(r1min[0], r1max[0], r2min[0], r2max[0]):
                print('Merged')
                coordinates[index2] = None
                xs = min(r1min[0], r2min[0])
                xe = max(r1max[0], r2max[0])
                ys = min(r1min[1], r2min[1])
                ye = max(r1max[1], r2max[1])
                rect1 = [(xs, ys), (xe, ys), (xe, ye), (xs, ye)]
                r1min = np.amin(rect1, axis=0)
                r1max = np.amax(rect1, axis=0)
                index2 = 0
            elif overlap(r1min[0], r1max[0], r2min[0], r2max[0]) > 0.8 and overlap(r1min[1], r1max[1], r2min[1], r2max[1]) > 0.8:
                print('Merged1')
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

        # new_coords.append(rect1)
        r1min = np.amin(rect1, axis=0)
        r1max = np.amax(rect1, axis=0)
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
    color = 0
    for rect in coords:
        rect.append(rect[0])
        ImageDraw.Draw(label_img).line(rect, width=15, fill=color+1)
        color = (color + 1) % 3

    label_img = np.array(label_img)
    for y in range(res.shape[0]):
        for x in range(res.shape[1]):
            if label_img[y][x] == 1:
                res[y][x] = (0, 255, 0)
            elif label_img[y][x] == 2:
                res[y][x] = (255, 255, 0)
            elif label_img[y][x] == 3:
                    res[y][x] = (255, 0, 0)
    return res


def main():
    out_path = 'D:\\smaller_splited_dataset2\\train'
    # out = 'D:\\merged_xml_test5'
    path = 'D:\\splited_xml\\train'

    files = os.listdir(path)
    xml_files = [filename for filename in files if filename.endswith('.xml')]
    # img_files = [filename for filename in files if filename.endswith('.jpg')]
    # out_files = os.listdir(out)
    # out_files = [filename for filename in out_files if filename.endswith('.xml')]

    # random.shuffle(xml_files)

    for xml_file in xml_files:
        # if xml_file in out_files:
        #     continue
        # img_file = xml_file.replace('.xml', '.jpg')

        # if img_file in img_files:
        xml_coordinates, width, height = get_coordinates_from_xml(path + '\\' + xml_file)
        xml_coordinates = merge_paragraphs(xml_coordinates)
        merged_xml = create_page_xml(xml_coordinates, width, height, xml_file.replace('.xml', ''))
        with open(out_path + '\\' + xml_file, 'wb') as f:
            f.write(merged_xml)
    return 0


if __name__ == "__main__":
    sys.exit(main())
