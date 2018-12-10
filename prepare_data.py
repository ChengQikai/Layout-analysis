import argparse
import sys
import os
import shutil
from scipy import misc
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input-folder', help='Path to input folder.', default='D:\BP\output\\', required=False)
    parser.add_argument(
        '-p', '--output-folder', help='Path to output folder.', default='D:\BP\data\\', required=False)
    parser.add_argument(
        '-s', '--scale', help='How to scale data.', default=1.0, required=False)
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
    clear_folder(path)
    if not os.path.exists(path + '/inputs'):
        os.makedirs(path + '/inputs')
    if not os.path.exists(path + '/labels'):
            os.makedirs(path + '/labels')


def create_label(image, xml_path, scale):
    img_size = (image.shape[:2])
    root = ET.parse(xml_path).getroot()
    img = Image.new("L", (img_size[1], img_size[0]), 0)

    for region in root.iter('{http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}TextRegion'):
        coordinates = region.find('{http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}Coords')
        points_string = coordinates.attrib['points'].split(' ')
        poly_points = []

        for point_string in points_string:
            point = point_string.split(',')
            poly_points.append((int(int(point[1])*scale), int(int(point[0])*scale)))

        poly = np.array(poly_points)
        poly = list(map(tuple, poly))

        ImageDraw.Draw(img).polygon(poly, fill=1)

    return np.array(img)


def process_data(input_folder, img_name, xml_name, scale):
    img = misc.imread(input_folder + '/' + img_name, mode="RGB")
    img = misc.imresize(img, scale)
    label = create_label(img, input_folder + xml_name, scale)
    return img, label


def main():
    args = parse_arguments()
    prepare_output_folder(args.output_folder)
    files = os.listdir(args.input_folder)
    xml_files = [filename for filename in files if filename.endswith('.xml')]
    img_files = [filename for filename in files if filename.endswith('.jpg')]
    scale = float(args.scale)

    for xml_name in xml_files:
        img_name = xml_name.replace('.xml', '.jpg')
        if img_name in img_files:
            img, label = process_data(args.input_folder, img_name, xml_name, scale)
            new_name = img_name.replace('.jpg', '.png')
            misc.imsave(args.output_folder + '/inputs/' + new_name, img)
            misc.imsave(args.output_folder + '/labels/' + new_name, label)
    print('Completed')
    return 0


if __name__ == "__main__":
    sys.exit(main())

