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


import argparse
import sys
import os
import shutil
from scipy import misc
import numpy as np
import lxml.etree as ET
from PIL import Image, ImageDraw

baseline_medians = dict()

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input-folder', help='Path to input folder.', default='..\\data\\test\\', required=True)
    parser.add_argument(
        '-p', '--output-folder', help='Path to output folder.', default='.\\output\\', required=True)
    parser.add_argument(
        '-s', '--scale', help='How to scale data.', default=1.0, required=False)
    parser.add_argument(
        '-e', '--edge', help='Options "vertical" otherwise "full".', default='', required=False);
    args = parser.parse_args()
    return args


def clear_folder(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(e)


def prepare_output_folder(path):
    if os.path.exists(path):
        clear_folder(path)
    if not os.path.exists(path + '/inputs'):
        os.makedirs(path + '/inputs')
    if not os.path.exists(path + '/labels'):
            os.makedirs(path + '/labels')
    if not os.path.exists(path + '/weights_map'):
        os.makedirs(path + '/weights_map')


def create_label(image, xml_path, scale, edge):
    img_size = (image.shape[:2])
    root = ET.parse(xml_path).getroot()
    img = Image.new("L", (img_size[1], img_size[0]), 0)
    weights_map = Image.new("L", (img_size[1], img_size[0]), 1)

    region_count = 0
    area = 0

    edge_vertical = edge == 'vertical'

    for region in root.iter('{http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}TextRegion'):
        region_count += 1
        coordinates = region.find('{http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}Coords')
        points_string = coordinates.attrib['points'].split(' ')
        poly_points = []

        for point_string in points_string:
            point = point_string.split(',')
            poly_points.append((int(int(point[1])*scale), int(int(point[0])*scale)))

        line_points = poly_points.copy()
        line_points.append(line_points[0])

        min_coords = np.amin(poly_points, axis=0)
        max_coords = np.amax(poly_points, axis=0)

        area += (max_coords[0] - min_coords[0]) * (max_coords[1] - min_coords[1])

        poly = np.array(poly_points)
        poly = list(map(tuple, poly))
        ImageDraw.Draw(img).polygon(poly, fill=1)

        
        if edge_vertical:
            ImageDraw.Draw(weights_map).line([(min_coords[0], min_coords[1]), (min_coords[0], max_coords[1])], width=3, fill=10)
            ImageDraw.Draw(weights_map).line([(max_coords[0], min_coords[1]), (max_coords[0], max_coords[1])], width=3, fill=10)
        else:
            ImageDraw.Draw(weights_map).line(line_points, width=4, fill=10)

    for region in root.iter('{http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}TextRegion'):
        coordinates = region.find('{http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}Coords')
        points_string = coordinates.attrib['points'].split(' ')
        poly_points = []

        for point_string in points_string:
            point = point_string.split(',')
            poly_points.append((int(int(point[1])*scale), int(int(point[0])*scale)))

        line_points = poly_points.copy()
        line_points.append(line_points[0])

        min_coords = np.amin(poly_points, axis=0)
        max_coords = np.amax(poly_points, axis=0)

        if edge_vertical:
            ImageDraw.Draw(img).line([(min_coords[0], min_coords[1]),(min_coords[0], max_coords[1])], width=4, fill=2)
            ImageDraw.Draw(img).line([(max_coords[0], min_coords[1]), (max_coords[0], max_coords[1])], width=4, fill=2)
        else:
            ImageDraw.Draw(img).line(line_points, width=4, fill=2)

    score = 0

    if region_count > 0 and area > (0.3 * (img_size[1] * img_size[0])):
        score = area/region_count

    return np.array(img), np.array(weights_map), score


def process_data(input_folder, img_name, xml_name, scale, edge):
    img = misc.imread(input_folder + '/' + img_name, mode="RGB")
    img = misc.imresize(img, scale)
    label, weights_map, score = create_label(img, input_folder + xml_name, scale, edge)
    return img, label, weights_map, score


def main():
    args = parse_arguments()
    prepare_output_folder(args.output_folder)

    files = os.listdir(args.input_folder)
    xml_files = [filename for filename in files if filename.endswith('.xml')]
    img_files = [filename for filename in files if filename.endswith('.jpg')]
    scale = float(args.scale)
    scores = []
    actual = 0
    count = len(xml_files)
    for xml_name in xml_files:
        img_name = xml_name.replace('.xml', '.jpg')
        if img_name in img_files:
            img, label, weigths_map, score = process_data(args.input_folder, img_name, xml_name, scale, args.edge)
            new_name = img_name.replace('.jpg', '.png')
            misc.imsave(args.output_folder + '/inputs/' + new_name, img)
            misc.imsave(args.output_folder + '/labels/' + new_name, label)
            misc.imsave(args.output_folder + '/weights_map/' + new_name, weigths_map)

            if score != 0:
                scores.append((new_name, score))

            print('{}/{}: {} : {}'.format(actual,count, img_name, score))

        actual += 1

    scores = sorted(scores, key=lambda x: x[1])
    score_sum = 0

    new_scores = []
    for score in scores:
        new_score = 1/score[1]
        score_sum += new_score
        new_scores.append((score[0], new_score))

    score_string = ''
    for score in new_scores:
        score_string += '{},{};'.format(score[0], score[1]/score_sum)

    with open(args.output_folder + '/scores', 'w') as f:
        f.write(score_string[:-1])


    print('Completed')
    return 0


if __name__ == "__main__":
    sys.exit(main())

