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

allowed_extensions = ['.jpg', '.jpeg', '.png']

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
    parser.add_argument('-n', action='store_true')
    args = parser.parse_args()
    return args 

    
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

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    length = len(names)
    count = 0
    for img in names:
        filename, file_extension = os.path.splitext(img)        

        if file_extension.lower() in allowed_extensions:         
            analyzer.__scale = scale

            coordinates, img_height, img_width = analyzer.get_document_paragraphs(input_path + img, no_layout=args.n)
            xml_string = HelperMethods.create_page_xml(coordinates, img_width, img_height, filename)


            with open('{}/{}.xml'.format(args.output_path, os.path.splitext(filename)[0]), 'wb') as f:
                f.write(xml_string)

            count += 1
        print('Completed: {}/{}'.format(count, length))
    return 0


if __name__ == "__main__":
    sys.exit(main())
