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
from DocumentAnalyzer import DocumentAnalyzer
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
import matplotlib.pyplot as plt


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--model', help='Path to folder where are input images.',
        default='./model/', required=False)
    parser.add_argument(
        '-i', '--input-path', help='Path to folder where are input images.',
        default='../test/', required=False)
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
        ImageDraw.Draw(label_img).line(rect, width=15, fill=1)
    label_img = np.array(label_img)
    for y in range(res.shape[0]):
        for x in range(res.shape[1]):
            if label_img[y][x] == 1:
                res[y][x] = (0, 255, 0)
    return res

    
def get_baseline_median(xml_path):
    root = ET.parse(xml_path).getroot()
    baseline_heights = []
    for baseline in root.iter('{http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}TextLine'):
        heights = baseline.get('custom')
        if heights and  heights.startswith('heights'):
            digits = re.findall(r'\d+', heights)
            baseline_heights.append(int(digits[0]))
    if(len(baseline_heights) > 0):
        return statistics.median(baseline_heights)
    else:
        return 0


def main():
    args = parse_arguments()
    input_path = args.input_path
    scale = 0.23

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    analyzer = DocumentAnalyzer(model=args.model)

    names = os.listdir(input_path)
    images_names = [filename for filename in names if filename.endswith('.jpeg')]
    xml_names = [filename for filename in names if filename.endswith('.xml')]

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    length = len(images_names)
    count = 0
    for img in images_names:
        filename = img.replace('.jpeg', '')
        xml_name = img.replace('.jpeg', '.xml')
        
        if xml_name in xml_names:
            baseline_median = get_baseline_median(input_path + xml_name)
            
            if baseline_median > 0:
                median_scale = (32 / baseline_median) * scale
            else:
                median_scale = scale
        else:
            median_scale = scale

        analyzer.__scale = median_scale

        f, axarr = plt.subplots(2, 3, dpi=1000)
        in_img = misc.imread(input_path + img, mode="RGB")
        coordinates, img_height, img_width = analyzer.get_document_paragraphs(input_path + img, axarr, in_img)
        xml_string = HelperMethods.create_page_xml(coordinates, img_width, img_height, filename)

        res = get_img_coords(in_img, coordinates)

        axarr[1][2].axis('off')
        axarr[1][2].imshow(res)
        f.show()
        f.savefig('{}/{}_progress.jpg'.format(args.output_path, filename), bbox_inches='tight')
        fig = plt.figure()
        f, axarr = plt.subplots(1, 1, dpi=1000)
        axarr.axis('off')
        axarr.imshow(res)
        plt.savefig('{}/{}.jpg'.format(args.output_path, filename), bbox_inches='tight')
        plt.close(fig)

        with open('{}/{}.xml'.format(args.output_path, os.path.splitext(filename)[0]), 'wb') as f:
            f.write(xml_string)

        count += 1
        print('Completed: {}/{}'.format(count, length))
    return 0


if __name__ == "__main__":
    sys.exit(main())
