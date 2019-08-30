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


import sys
from DocumentAnalyzerSeparators import DocumentAnalyzer
import lxml.etree as ET
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
        default='./model/', required=False)
    parser.add_argument(
        '-i', '--input-path', help='Path to folder where are input images.',
        default='../data/test/', required=False)
    parser.add_argument(
        '-o', '--output-path', help='Path where save xml files.',
        default='./detect_output', required=False)

    args = parser.parse_args()
    return args


def get_img_coords(img, coords):
    res = img.copy()
    label_img = Image.new("L", (img.shape[1], img.shape[0]), 0)
    for rect in coords:
        rect.append(rect[0])
        ImageDraw.Draw(label_img).line(rect, width=20, fill=1)
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
    show_img = True

    names = os.listdir(input_path)
    images_names = [filename for filename in names if filename.endswith('.jpg')]

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    length = len(images_names)
    count = 0
    for img in images_names:
        filename = img.replace('.jpg', '')
        xml_name = img.replace('.jpg', '.xml')


        coordinates, img_height, img_width = analyzer.get_document_paragraphs(input_path + img, line_coords=HelperMethods.get_line_coords(input_path + xml_name))
        xml_string = HelperMethods.create_page_xml(coordinates, img_width, img_height, filename)

        if show_img:
            in_img = misc.imread(input_path + img, mode="RGB")
            res = get_img_coords(in_img, coordinates)
            fig = plt.figure()
            f, axarr = plt.subplots(1, 1, dpi=1000)
            axarr.axis('off')
            axarr.imshow(res)
            plt.savefig('{}/{}.jpg'.format(args.output_path, filename), bbox_inches='tight')
            plt.close(fig)

        with open('{}/{}.xml'.format(args.output_path, filename), 'wb') as f:
            f.write(xml_string)

        count += 1
        print('Completed: {}/{}'.format(count, length))
    return 0


if __name__ == "__main__":
    sys.exit(main())
