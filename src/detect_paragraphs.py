import sys
from DocumentAnalyzer import DocumentAnalyzer
from scipy import misc
import matplotlib.pyplot as plt
import skimage.draw as skd
import random
import HelperMethods
import argparse
import os
import numpy as np

from PIL import Image, ImageDraw


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--model', help='Path to folder where are input images.',
        default='', required=True)
    parser.add_argument(
        '-i', '--input-path', help='Path to folder where are input images.',
        default='', required=True)
    parser.add_argument(
        '-o', '--output-path', help='Path where save xml files.',
        default='', required=True)
    parser.add_argument(
        '-f', '--footprint-size', help='Footprint size.', required=False)

    args = parser.parse_args()
    return args


def show_graph(input_path, coordinates, filename=''):
    img = misc.imread(input_path, mode="RGB")
    seg_img = img.copy()
    plt.subplot(1, 2, 1)
    plt.imshow(img)

    r = 18
    g = 159
    b = 220
    for rect in coordinates:
        rr, cc = skd.rectangle((rect[0][1], rect[0][0]), (rect[2][1], rect[2][0]))
        seg_img[rr, cc] = (r, g, b)
        r = r + random.randint(10, 160) % 255
        g = g + random.randint(10, 160) % 255
        b = b + random.randint(10, 160) % 255

    plt.subplot(1, 2, 2)
    plt.imshow(seg_img)
    plt.savefig(filename)


import random

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
    args = parse_arguments()
    input_path = args.input_path
    analyzer = DocumentAnalyzer(model=args.model)

    if args.footprint_size:
        analyzer.set_footprint_size(int(args.footprint_size))

    names = os.listdir(input_path)
    images_names = [filename for filename in names if filename.endswith('.jpg')]
    random.shuffle(images_names)

    output_list = os.listdir(args.output_path)
    output_images_names = [filename for filename in output_list if filename.endswith('.jpg')]

    # os.makedirs(args.output_path)

    length = len(images_names)
    count = 0
    for img in images_names:
        filename = img.replace('.jpg', '')
        if img in output_images_names:
            continue
        coordinates, img_height, img_width = analyzer.get_document_paragraphs(input_path + img)
        xml_string = HelperMethods.create_page_xml(coordinates, img_width, img_height, filename)
        in_img = misc.imread(input_path + img, mode="RGB")
        res = get_img_coords(in_img, coordinates)
        fig = plt.figure(dpi=300)
        plt.imshow(res)

        plt.savefig('{}/{}.jpg'.format(args.output_path, filename))
        plt.close(fig)
        # show_graph(input_path + img, coordinates, '{}/{}.jpg'.format(args.output_path, filename))
        with open('{}/{}.xml'.format(args.output_path, filename), 'wb') as f:
            f.write(xml_string)
        count += 1
        print('Completed: {}/{}'.format(count, length))
    return 0


if __name__ == "__main__":
    sys.exit(main())

