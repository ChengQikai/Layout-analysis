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
import os
from HelperMethods import evaluate_symetric_best_dice, get_coordinates_from_xml
from scipy import misc
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
import numpy as np
import argparse


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input-path', help='Path to analyzed xml files', default='', required=True)
    parser.add_argument(
        '-r', '--original-path', help='Path to original xml files', default='', required=True)
    parser.add_argument(
        '-o', '--output', help='Path to output folder', default='./evaluate_outputh', required=False)
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

def main():
    args = parse_argument()
    detected_path = args.input_path
    original_path = args.original_path
    output_path = args.output

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    names = os.listdir(args.input_path)
    names = [filename for filename in names if filename.endswith('.xml')]

    count = len(names)
    actual = 0
    accurracy_sum = 0
    for name in names:
        img_name = name.replace('.xml','.jpg')
        img = misc.imread(original_path + img_name, mode="RGB")
        fig = plt.figure(dpi=300)
        plt.subplot(1, 2, 1)
        or_coords, _, _ = get_coordinates_from_xml(original_path + name)
        res = get_img_coords(img, or_coords)
        plt.imshow(res)
        plt.title(img_name)
        accurracy = evaluate_symetric_best_dice(detected_path + name, original_path + name)
        accurracy_sum += accurracy
        print('{}:{}'.format(name, accurracy))
        plt.subplot(1, 2, 2)
        det_img = misc.imread(detected_path + img_name, mode="RGB")
        plt.title('{}%'.format(accurracy * 100))
        plt.imshow(det_img)
        plt.savefig('{}/{}_{}'.format(output_path, accurracy, img_name))
        plt.close(fig)
        actual += 1
        print('Complete {}/{}'.format(actual, count))
    print('Complete Accuraccy: ', accurracy_sum/float(count))
    with open('{}/acc.txt'.format(output_path),'w') as f:
        f.write('Complete Accuraccy: {}'.format(accurracy_sum/float(count)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
